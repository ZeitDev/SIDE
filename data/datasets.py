import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root_path, mode='train', tasks=None, subset_names=None, transform=None):
        self.root_path = root_path
        self.mode = mode
        self.tasks = tasks if tasks is not None else []
        self.subset_names = subset_names
        self.transform = transform
        
        self._load_samples()
        
    def get_all_subset_names(self):
        mode_path = os.path.join(self.root_path, self.mode)
        subset_names = sorted([d for d in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, d))])
        return subset_names
    
    def _get_file_names(self, subset_path):
        """
        Override function should be implemented by the dataset subclass.
        Assumes file names are the same across all data types in a subset.
        Returns a list of file names in the subset.
        """
        raise NotImplementedError('Dataset subclass must implement _get_file_names.')
    
    def _get_sample_path(self, subset_path, file_name):
        """
        Override function should be implemented by the dataset subclass.
        Returns a dictionary containing paths to the data for a single sample.
        e.g., {'left_image': path, 'segmentation': path, ...}
        """
        raise NotImplementedError('Dataset subclass must implement _get_sample_paths.')
    
    def _load_samples(self):
        mode_path = os.path.join(self.root_path, self.mode)
        
        if self.subset_names is None: self.subset_names = self.get_all_subset_names()
            
        self.sample_paths = []
        for subset_name in self.subset_names:
            subset_path = os.path.join(mode_path, subset_name)
            file_names = self._get_file_names(subset_path)
            
            for file_name in file_names:
                sample_path = self._get_sample_path(subset_path, file_name)
                self.sample_paths.append(sample_path)
                
    def __len__(self):
        return len(self.sample_paths)
    
    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        
        data = {}
        if 'left_image' in sample_path:
            data['left_image'] = np.array(Image.open(sample_path['left_image']).convert('RGB'))
        if 'right_image' in sample_path:
            data['right_image'] = np.array(Image.open(sample_path['right_image']).convert('RGB'))
            
        if 'segmentation' in sample_path:
            mask = np.array(Image.open(sample_path['segmentation']))
            if len(mask.shape) == 2: mask = np.expand_dims(mask, axis=-1)
            data['segmentation'] = mask
        if 'disparity' in sample_path:
            pass # TODO: load disparity map
        
        if self.transform: data = self.transform(**data) # ! NOT TESTED
        else: data = {k: torch.from_numpy(v).permute(2,0,1).float() for k, v in data.items()} # ! NOT TESTED
        
        image = data['left_image']
        if 'right_image' in data:
            image = torch.cat((data['left_image'], data['right_image']), dim=0)
            
        targets = {task: data[task] for task in self.tasks if task in data}
        
        return image, targets
    
class OverfitDataset(BaseDataset):
    def __init__(self, mode='train', tasks=None, subset_names=None, transform=None):
        root_path = '/data/Zeitler/SIDE/OverfitDataset/'
        super().__init__(root_path, mode, tasks, subset_names, transform)
        
    def _get_file_names(self, subset_path):
        left_images_path = os.path.join(subset_path, 'left_images')
        return sorted(os.listdir(left_images_path))
    
    def _get_sample_path(self, subset_path, file_name):
        sample_path = {}
        sample_path['left_image'] = os.path.join(subset_path, 'left_images', file_name)
        
        if self.mode == 'train':
            if 'segmentation' in self.tasks:
                sample_path['segmentation'] = os.path.join(subset_path, 'ground_truth', 'segmentation_masks_instrument_type', file_name)
                
            if 'disparity' in self.tasks:
                sample_path['right_image'] = os.path.join(subset_path, 'right_images', file_name)
                sample_path['disparity'] = os.path.join(subset_path, 'ground_truth', 'disparity_maps', file_name)
                
        return sample_path

class EndoVis17(BaseDataset):
    def __init__(self, mode='train', tasks=None, subset_names=None, transform=None):
        root_path = '/data/Zeitler/SIDE/EndoVis17/processed/'
        super().__init__(root_path, mode, tasks, subset_names, transform)
        
    def _get_file_names(self, subset_path):
        left_images_path = os.path.join(subset_path, 'left_images')
        return sorted(os.listdir(left_images_path))
    
    def _get_sample_path(self, subset_path, file_name):
        sample_path = {}
        sample_path['left_image'] = os.path.join(subset_path, 'left_images', file_name)
        
        if self.mode == 'train':
            if 'segmentation' in self.tasks:
                sample_path['segmentation'] = os.path.join(subset_path, 'ground_truth', 'segmentation_masks_instrument_type', file_name)
                
            if 'disparity' in self.tasks:
                sample_path['right_image'] = os.path.join(subset_path, 'right_images', file_name)
                sample_path['disparity'] = os.path.join(subset_path, 'ground_truth', 'disparity_maps', file_name)
                
        return sample_path
    
            
        