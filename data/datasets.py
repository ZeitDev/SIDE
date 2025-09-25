
import os
import re

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

class CholecSeg8k(Dataset):
    def __init__(self, mode='train', transform=None):
        self.data_path = os.path.join('/data/Zeitler/segmentation/processed/CholecSeg8k', mode)
        self.images_path = os.path.join(self.data_path, 'images')
        self.masks_path = os.path.join(self.data_path, 'masks')
        self.transform = transform
        
        self._generate_data_paths()
        
    def _generate_data_paths(self):
        assert len(os.listdir(self.images_path)) == len(os.listdir(self.masks_path))
        
        self.image_paths = sorted(os.listdir(self.images_path))
        self.mask_paths = sorted(os.listdir(self.masks_path))
        self.data_paths = list(zip(self.image_paths, self.mask_paths))
        
        for image_path, mask_path in self.data_paths:
            assert re.findall(r'\d+', image_path) == re.findall(r'\d+', mask_path)
        
    def __len__(self):
        return len(os.listdir(self.images_path))
    
    def __getitem__(self, index):
        image_path = os.path.join(self.images_path, self.data_paths[index][0])
        mask_path = os.path.join(self.masks_path, self.data_paths[index][1])
        
        image = decode_image(image_path).to(torch.float32) / 255.0
        mask = decode_image(mask_path).to(torch.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask