import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A

from criterions.disparity import EntropyConfidence

class BaseDataset(Dataset):
    """
    Expects the following dataset directory structure:
    root_path/
        train/
            subset_1/
            subset_2/
            ...
        val/
            subset_3/
            ...
        test/
            subset_4/
            ...
    """
    
    def __init__(self, config: Dict[str, Any], root_path: str, mode: str = 'train', transforms: Optional[A.Compose] = None):
        self.config = config
        self.root_path = root_path
        self.mode = mode
        self.transforms = transforms
        
        self.class_mappings = None
        self.n_segmentation_classes = self.config['data']['num_of_classes']['segmentation']
        self.focal_length_scale_factor = self.config['data']['focal_length_scale_factor']
        
        self.entropy_confidence = EntropyConfidence(c_min=0.3900, c_max=0.8981) # Numbers derived by dataset stats, TODO: move to config
        
        self._get_class_mappings()
        self._load_samples()
    
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
        
        transform_mode = 'train' if self.mode == 'train' else 'test'
        for t in self.config['data']['transforms'][transform_mode]:
            if t['name'] == 'Resize':
                self.target_width = t['params']['width']
                self.target_height = t['params']['height']
                break
            
        self.sample_paths = []
        for subset_name in sorted(os.listdir(mode_path)):
            subset_path = os.path.join(mode_path, subset_name)
            file_names = self._get_file_names(subset_path)
            
            for file_name in file_names:
                sample_paths = self._get_sample_paths(subset_path, file_name)
                self.sample_paths.append(sample_paths)
                
    def _extract_intrinsics(self, intrinsics_path: str) -> Tuple[float, float]:
        with open(intrinsics_path, 'r') as f:
            intrinsics = json.load(f)
            Q = intrinsics['Q']
            focal_length = Q[2][3] * self.focal_length_scale_factor
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
            
            if 'teacher_disparity_confidence' in sample_paths:
                teacher_disparity_confidence_raw = np.array(Image.open(sample_paths['teacher_disparity_confidence']))
                teacher_disparity_confidence = torch.from_numpy(teacher_disparity_confidence_raw.astype(np.int32)).float() / 65535.0
                data['teacher_disparity_confidence'] = teacher_disparity_confidence.unsqueeze(0)
                
                if data['teacher_disparity_confidence'].mean() < 0.6:
                    data['disparity'] = torch.zeros_like(data['disparity'])
                    data['teacher_disparity'] = torch.zeros_like(data['teacher_disparity'])
        
        return data
    
class OverfitDataset(BaseDataset):
    def __init__(self, config: Dict[str, Any], mode: str = 'train', transforms: Optional[A.Compose] = None, tasks: Optional[Dict[str, Any]] = None):
        root_path = '/data/Zeitler/SIDED/OverfitDataset'
        super().__init__(config=config, root_path=root_path, mode=mode, transforms=transforms)
        
    def _get_class_mappings(self) -> None:
        if self.config['training']['tasks']['segmentation']['enabled']:
            class_mapping_path = os.path.join(self.root_path, 'mapping.json')
            with open(class_mapping_path, 'r') as f:
                name2id = json.load(f)
                
            self.segmentation_class_mappings = {v: k for k, v in name2id.items()}
        
    def _get_file_names(self, subset_path: str) -> List[str]:
        left_images_path = os.path.join(subset_path, 'input', 'left_images')
        return sorted(os.listdir(left_images_path))
    
    def _get_sample_paths(self, subset_path: str, file_name: str) -> Dict[str, str]:
        sample_paths = {}
        sample_paths['left_image'] = os.path.join(subset_path, 'input', 'left_images', file_name)
        
        if self.config['training']['tasks']['segmentation']['enabled']:
            sample_paths['segmentation'] = os.path.join(subset_path, 'target', 'segmentation', file_name)
            
            if self.config['training']['tasks']['segmentation']['distillation']['enabled'] and self.config['training']['tasks']['segmentation']['distillation']['name'] == 'offline':
                sample_paths['teacher_segmentation'] = os.path.join(subset_path, 'teacher', 'segmentation_2_256_256', file_name.replace('.png', '.pt'))
            
        if self.config['training']['tasks']['disparity']['enabled']:
            sample_paths['right_image'] = os.path.join(subset_path, 'input', 'right_images', file_name)
            sample_paths['disparity'] = os.path.join(subset_path, 'target', 'disparity', file_name)
            sample_paths['intrinsics'] = os.path.join(subset_path, 'calibration', 'rectified_calibration.json')

            if self.config['training']['tasks']['disparity']['distillation']['enabled'] and self.config['training']['tasks']['disparity']['distillation']['name'] == 'offline':
                sample_paths['teacher_disparity'] = os.path.join(subset_path, 'teacher', 'disparity_128_256_256', file_name.replace('.png', '.pt'))
            
        return sample_paths

class EndoVis17(BaseDataset):
    def __init__(self, config: Dict[str, Any], mode: str = 'train',  transforms: Optional[A.Compose] = None):
        root_path = '/data/Zeitler/SIDED/EndoVis17/processed'
        super().__init__(config=config, root_path=root_path, mode=mode, transforms=transforms)
        
    def _get_class_mappings(self) -> None:
        if self.config['training']['tasks']['segmentation']['enabled']:
            class_mapping_path = os.path.join(self.root_path, f'mapping_{self.n_segmentation_classes}.json')
            with open(class_mapping_path, 'r') as f:
                name2id = json.load(f)
                
            self.segmentation_class_mappings = {v: k for k, v in name2id.items()}
        
    def _get_file_names(self, subset_path: str) -> List[str]:
        left_images_path = os.path.join(subset_path, 'input', 'left_images')
        return sorted(os.listdir(left_images_path))
    
    def _get_sample_paths(self, subset_path: str, file_name: str) -> Dict[str, str]:
        sample_paths = {}
        sample_paths['left_image'] = os.path.join(subset_path, 'input', 'left_images', file_name)
        
        if self.config['training']['tasks']['segmentation']['enabled']:
            
            sample_paths['segmentation'] = os.path.join(subset_path, 'target', f'segmentation_{self.n_segmentation_classes}', file_name)
            
            if self.config['training']['tasks']['segmentation']['distillation']['enabled'] and self.config['training']['tasks']['segmentation']['distillation']['name'] == 'offline':
                segmentation_logit_resolution = self.target_width // 4
                sample_paths['teacher_segmentation'] = os.path.join(subset_path, 'teacher', f'segmentation_{self.n_segmentation_classes}_{segmentation_logit_resolution}_{segmentation_logit_resolution}', file_name.replace('.png', '.pt'))
            
        if self.config['training']['tasks']['disparity']['enabled']:
            sample_paths['right_image'] = os.path.join(subset_path, 'input', 'right_images', file_name)
            sample_paths['disparity'] = os.path.join(subset_path, 'target', 'disparity', file_name)
            sample_paths['intrinsics'] = os.path.join(subset_path, 'calibration', 'rectified_calibration.json')

            #if self.config['training']['tasks']['disparity']['distillation']['enabled'] and self.config['training']['tasks']['disparity']['distillation']['name'] == 'offline':
            disparity_logit_resolution = self.target_width // 4
            n_classes_disparity = self.config['data']['max_disparity'] // 4
            sample_paths['teacher_disparity'] = os.path.join(subset_path, 'teacher', f'disparity_{n_classes_disparity}_{disparity_logit_resolution}_{disparity_logit_resolution}', file_name.replace('.png', '.pt'))
            sample_paths['teacher_disparity_confidence'] = os.path.join(subset_path, 'teacher', 'disparity_confidence', file_name)
            
            
        return sample_paths
    
    
            
        