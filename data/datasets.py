import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
import albumentations as A

class BaseDataset(Dataset):
    def __init__(self, config: Dict[str, Any], root_path: str, mode: str = 'train', transforms: Optional[A.Compose] = None, subset_names: Optional[List[str]] = None):
        self.config = config
        self.root_path = root_path
        self.mode = mode
        self.subset_names = subset_names
        self.transforms = transforms
        
        self.class_mappings = None
        
        self._get_class_mappings()
        self._load_samples()
        
    def get_all_subset_names(self) -> List[str]:
        """
        Expects the following dataset directory structure:
        root_path/
            train/
                subset_1/
                subset_2/
                ...
            test/
                subset_1/
                ...
        """
        mode_path = os.path.join(self.root_path, self.mode)
        subset_names = sorted([d for d in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, d))])
        return subset_names
    
    def _get_class_mappings(self) -> Optional[Dict[int, str]]:
        """
        Override function should be implemented by the dataset subclass.
        Loads class to pixel value mappings for the dataset.
        """
        raise NotImplementedError('Dataset subclass must implement _load_class_mappings.')
    
    def _get_file_names(self, subset_path: str) -> List[str]:
        """
        Override function should be implemented by the dataset subclass.
        Assumes file names are the same across all data types in a subset.
        Returns a list of file names in the subset.
        """
        raise NotImplementedError('Dataset subclass must implement _get_file_names.')
    
    def _get_sample_paths(self, subset_path: str, file_name: str) -> Dict[str, str]:
        """
        Override function should be implemented by the dataset subclass.
        Returns a dictionary containing paths to the data for a single sample.
        Structure: {'left_image': path, 'segmentation': path, ...}
        """
        raise NotImplementedError('Dataset subclass must implement _get_sample_paths.')
    
    def _load_samples(self) -> None:
        mode_path = os.path.join(self.root_path, self.mode)
        
        if self.subset_names is None: self.subset_names = self.get_all_subset_names()
            
        self.sample_paths = []
        for subset_name in self.subset_names:
            subset_path = os.path.join(mode_path, subset_name)
            file_names = self._get_file_names(subset_path)
            
            for file_name in file_names:
                sample_paths = self._get_sample_paths(subset_path, file_name)
                self.sample_paths.append(sample_paths)
                
    def _extract_intrinsics(self, intrinsics_path: str) -> Tuple[float, float]:
        with open(intrinsics_path, 'r') as f:
            intrinsics = json.load(f)
            Q = intrinsics['Q']
            focal_length = Q[2][3]
            baseline = abs(1.0 / Q[3][2])
        return baseline, focal_length
    
    def __len__(self) -> int:
        return len(self.sample_paths)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample_paths = self.sample_paths[idx]
        
        data = {}
        if 'left_image' in sample_paths:
            data['image'] = np.array(Image.open(sample_paths['left_image']).convert('RGB'))
        if 'right_image' in sample_paths:
            data['right_image'] = np.array(Image.open(sample_paths['right_image']).convert('RGB'))
            
        if 'segmentation' in sample_paths:
            data['segmentation'] = np.array(Image.open(sample_paths['segmentation']))
        if 'disparity' in sample_paths:
            disparity_map = np.array(Image.open(sample_paths['disparity']))
            
            valid_mask = disparity_map > 0
            disparity_map[valid_mask] = disparity_map[valid_mask] / 128
            disparity_map[~valid_mask] = 0
            
            data['disparity'] = np.expand_dims(disparity_map, axis=-1)
        
        if self.transforms: data = self.transforms(**data)
        
        if 'segmentation' in data: 
            data['segmentation'] = data['segmentation'].unsqueeze(0)
            if 'teacher_segmentation' in sample_paths:
                teacher_segmentation = torch.load(sample_paths['teacher_segmentation'], weights_only=True)
                data['teacher_segmentation'] = teacher_segmentation.float()
        if 'disparity' in data:
            baseline, focal_length = self._extract_intrinsics(sample_paths['intrinsics'])
            data['baseline'] = torch.tensor(baseline).view(1, 1, 1)
            data['focal_length'] = torch.tensor(focal_length).view(1, 1, 1)
            if 'teacher_disparity' in sample_paths:
                teacher_disparity = torch.load(sample_paths['teacher_disparity'], weights_only=True)
                data['teacher_disparity'] = teacher_disparity.float()
        
        
        return data
    
class OverfitDataset(BaseDataset):
    def __init__(self, mode: str = 'train', transforms: Optional[A.Compose] = None, tasks: Optional[Dict[str, Any]] = None, subset_names: Optional[list[str]] = None):
        root_path = '/data/Zeitler/SIDED/OverfitDataset'
        super().__init__(root_path, mode, transforms, tasks, subset_names)
        
    def _get_class_mappings(self) -> None:
        if self.config['training']['tasks']['segmentation']['enabled']:
            class_mapping_path = os.path.join(self.root_path, 'mapping.json')
            with open(class_mapping_path, 'r') as f:
                name2id = json.load(f)
                
            self.class_mappings = {v: k for k, v in name2id.items()}
        
    def _get_file_names(self, subset_path: str) -> List[str]:
        left_images_path = os.path.join(subset_path, 'input', 'left_images')
        return sorted(os.listdir(left_images_path))
    
    def _get_sample_paths(self, subset_path: str, file_name: str) -> Dict[str, str]:
        sample_paths = {}
        sample_paths['left_image'] = os.path.join(subset_path, 'input', 'left_images', file_name)
        
        if self.config['training']['tasks']['segmentation']['enabled']:
            sample_paths['segmentation'] = os.path.join(subset_path, 'ground_truth', 'segmentation', file_name)
            
            if self.config['training']['tasks']['segmentation']['knowledge_distillation']['enabled']:
                sample_paths['teacher_segmentation'] = os.path.join(subset_path, 'teacher', 'segmentation', file_name.replace('.png', '.pt'))
            
        if self.config['training']['tasks']['disparity']['enabled']:
            sample_paths['right_image'] = os.path.join(subset_path, 'input', 'right_images', file_name)
            sample_paths['disparity'] = os.path.join(subset_path, 'ground_truth', 'disparity', file_name)
            sample_paths['intrinsics'] = os.path.join(subset_path, 'calibration', 'rectified_calibration.json')

            if self.config['training']['tasks']['disparity']['knowledge_distillation']['enabled']:
                sample_paths['teacher_disparity'] = os.path.join(subset_path, 'teacher', 'disparity', file_name.replace('.png', '.pt'))
            
        return sample_paths

class EndoVis17(BaseDataset):
    def __init__(self, mode: str = 'train',  transforms: Optional[A.Compose] = None, tasks: Optional[Dict[str, Any]] = None, subset_names: Optional[list[str]] = None):
        root_path = '/data/Zeitler/SIDED/EndoVis17/processed'
        super().__init__(root_path, mode, transforms, tasks, subset_names)
        
    def _get_class_mappings(self) -> None:
        if self.config['training']['tasks']['segmentation']['enabled']:
            class_mapping_path = os.path.join(self.root_path, 'mapping.json')
            with open(class_mapping_path, 'r') as f:
                name2id = json.load(f)
                
            self.class_mappings = {v: k for k, v in name2id.items()}
        
    def _get_file_names(self, subset_path: str) -> List[str]:
        left_images_path = os.path.join(subset_path, 'input', 'left_images')
        return sorted(os.listdir(left_images_path))
    
    def _get_sample_paths(self, subset_path: str, file_name: str) -> Dict[str, str]:
        sample_paths = {}
        sample_paths['left_image'] = os.path.join(subset_path, 'input', 'left_images', file_name)
        
        if self.config['training']['tasks']['segmentation']['enabled']:
            sample_paths['segmentation'] = os.path.join(subset_path, 'ground_truth', 'segmentation', file_name)
            
            if self.config['training']['tasks']['segmentation']['knowledge_distillation']['enabled'] and self.config['training']['tasks']['segmentation']['knowledge_distillation']['name'] == 'offline':
                sample_paths['segmentation_teacher'] = os.path.join(subset_path, 'teacher', 'segmentation', file_name.replace('.png', '.pt'))
            
        if self.config['training']['tasks']['disparity']['enabled']:
            sample_paths['right_image'] = os.path.join(subset_path, 'input', 'right_images', file_name)
            sample_paths['disparity'] = os.path.join(subset_path, 'ground_truth', 'disparity', file_name)
            sample_paths['intrinsics'] = os.path.join(subset_path, 'calibration', 'rectified_calibration.json')

            if self.config['training']['tasks']['disparity']['knowledge_distillation']['enabled'] and self.config['training']['tasks']['disparity']['knowledge_distillation']['name'] == 'offline':
                sample_paths['disparity_teacher'] = os.path.join(subset_path, 'teacher', 'disparity', file_name.replace('.png', '.pt'))
            
        return sample_paths
    
class Scared(BaseDataset):
    def __init__(self, mode: str = 'train',  transforms: Optional[A.Compose] = None, tasks: Optional[Dict[str, Any]] = None, subset_names: Optional[list[str]] = None):
        root_path = '/data/Zeitler/SIDED/SCARED/processed'
        super().__init__(root_path, mode, transforms, tasks, subset_names)
        
    def _get_file_names(self, subset_path: str) -> List[str]:
        return sorted(os.listdir(subset_path))
    
    def _get_sample_paths(self, subset_path: str, file_name: str) -> Dict[str, str]:
        sample_paths = {}
        sample_paths['left_image'] = os.path.join(subset_path, file_name, 'left_rectified.png')
        sample_paths['right_image'] = os.path.join(subset_path, file_name, 'right_rectified.png')
        sample_paths['disparity'] = os.path.join(subset_path, file_name, 'disparity.png')
        
        return sample_paths
    
    
            
        