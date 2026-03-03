# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml
import torch
import cv2
from utils import helpers
from utils.helpers import load, logits2disparity
from torch.utils.data import DataLoader
from data.transforms import build_transforms
from models.teachers.foundation_stereo_wrapper import FoundationStereoWrapper

from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from PIL import Image

from tqdm import tqdm

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment()


# %%
# load debug config
with open(os.path.join('configs', 'base.yaml'), 'r') as f: config = yaml.safe_load(f)
#with open(os.path.join('configs', 'debug.yaml'), 'r') as f: experiment_config = yaml.safe_load(f)
#config = helpers.deep_merge(experiment_config, base_config)

# %%
# Custom EndoVis Dataloader

class EndoVisTeacherDataset(Dataset):
    def __init__(self, mode, transforms=None):
        self.root_dir = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}'
        self.transforms = transforms
        
        self.subsets = sorted(os.listdir(self.root_dir))
        self.file_names = sorted(os.listdir(os.path.join(self.root_dir, self.subsets[0], 'input', 'left_images')))
        
        self.left_image_paths = []
        self.right_image_paths = []
        for subset in self.subsets:
            for file_name in self.file_names:
                self.left_image_paths.append(os.path.join(self.root_dir, subset, 'input', 'left_images', file_name))
                self.right_image_paths.append(os.path.join(self.root_dir, subset, 'input', 'right_images', file_name))
        
        # convert list of left_image_paths to list of dict with 'left_image' as key
        self.sample_paths = []
        for left_path, right_path in zip(self.left_image_paths, self.right_image_paths):
            self.sample_paths.append({'left_image': left_path, 'right_image': right_path})
        
    def __len__(self):
        return len(self.left_image_paths)
    
    def __getitem__(self, idx):
        data = {}
        left_image_path = self.left_image_paths[idx]
        right_image_path = self.right_image_paths[idx]
        data['image'] = np.array(Image.open(left_image_path).convert('RGB'))
        data['right_image'] = np.array(Image.open(right_image_path).convert('RGB'))
        
        if self.transforms: data = self.transforms(**data)
        
        data['image_path'] = left_image_path
        
        return data


# %%
model = FoundationStereoWrapper()
model.to('cuda')
model.eval()

# %% Save Disparity Maps
if False:
    mode = 'train'  # ! Do both train and test
    config['data']['transforms'][mode] = [
        {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
    ]
    transforms_disparity = build_transforms(config, mode=mode)

    dataset_disparity = EndoVisTeacherDataset(
        mode=mode,
        transforms=transforms_disparity)

    dataloader_disparity = DataLoader(
        dataset_disparity,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    helpers.check_dataleakage(mode, dataset_disparity)

    for data in tqdm(dataloader_disparity):
        with torch.cuda.amp.autocast(True) and torch.no_grad():
            left_image = data['image'].to('cuda')
            right_image = data['right_image'].to('cuda')
            
            disparity = model.get_disparity(left_image, right_image)
            
            image_path = data['image_path'][0]
            save_path = image_path.replace('input', 'ground_truth').replace('left_images', 'disparity')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            disp = disparity.squeeze().cpu().numpy()
            disp_scaled = disp * 128.0
            disp_scaled = np.clip(disp_scaled, 0, 65535)
            disp_scaled = disp_scaled.astype(np.uint16)
                
            cv2.imwrite(save_path, disp_scaled)

# %% Save Disparity Teacher Logits
if True:
    mode = 'test'
    
    config['data']['transforms'][mode] = [
        {'name': 'CenterCrop', 'params': {'height': 1024, 'width': 1024}},
        {'name': 'Resize', 'params': {'height': 1024, 'width': 1024}},
        {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
    ]
    train_transform = build_transforms(config, mode=mode)

    dataset_train = EndoVisTeacherDataset(
        mode=mode,
        transforms=train_transform)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    helpers.check_dataleakage(mode, dataset_train)

    FP16_MAX = torch.finfo(torch.float16).max
    for data in tqdm(dataloader_train):
        with torch.cuda.amp.autocast(True) and torch.no_grad():
            left_image = data['image'].to('cuda')
            right_image = data['right_image'].to('cuda')
            
            logit = model.get_logits(left_image, right_image)

            logit_save = logit.squeeze().half().cpu()
            assert logit_save.abs().max() < FP16_MAX, f'Logit values exceed FP16 max value: {logit_save.abs().max()}'
            assert not torch.isnan(logit_save).any(), 'Logit contains NaN values'

            # raw_disparity = logits2disparity(logit, size=left_image.shape[2:]) * 512.0
            
            image_path = data['image_path'][0]
            save_path = image_path.replace('input', 'teacher').replace('left_images', 'disparity_128_256_256').replace('.png', '.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(logit_save, save_path)


# %% Saving Right Disparity Maps
def left_right_consistency_check(left_disp, right_disp, threshold=1.0):
    left_disp = left_disp.detach().cpu()
    right_disp = right_disp.detach().cpu()
    
    B, C, H, W = left_disp.shape
    x_grid = torch.linspace(-1, 1, W, device=left_disp.device)
    y_grid = torch.linspace(-1, 1, H, device=left_disp.device)
    y, x = torch.meshgrid(y_grid, x_grid, indexing='ij')
    grid = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    
    normalized_disp = (left_disp.squeeze(1) / (W / 2)).unsqueeze(-1)
    shifted_grid = grid.clone()
    shifted_grid[..., 0] -= normalized_disp[..., 0]
    
    warped_disp_right = F.grid_sample(right_disp, shifted_grid, align_corners=True)
    diff = torch.abs(left_disp - warped_disp_right)
    
    valid_mask = (diff < threshold).float()
    
    return valid_mask

if False:
    mode = 'train' # ! Do both train and test
    
    config['data']['transforms'][mode] = [
        {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
    ]
    transforms_disparity = build_transforms(config, mode=mode)

    dataset_disparity = EndoVisTeacherDataset(
        mode=mode,
        transforms=transforms_disparity)

    dataloader_disparity = DataLoader(
        dataset_disparity,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    helpers.check_dataleakage(mode, dataset_disparity)

    valid_percentages = []
    for data in tqdm(dataloader_disparity):
        with torch.cuda.amp.autocast(True) and torch.no_grad():
            left_image = data['image'].to('cuda')
            right_image = data['right_image'].to('cuda')
            #left_disparity = model.get_disparity(left_image, right_image)
            
            left_image_flipped = torch.flip(left_image, dims=[3])
            right_image_flipped = torch.flip(right_image, dims=[3])
            right_disparity_flipped = model.get_disparity(right_image_flipped, left_image_flipped)
            right_disparity = torch.flip(right_disparity_flipped, dims=[3])
            
            # valid_mask = left_right_consistency_check(left_disparity, right_disparity)
            # valid_percentage = (valid_mask.sum() / valid_mask.numel()).item() * 100
            # valid_percentages.append(valid_percentage)
            #print((valid_mask.sum() / valid_mask.numel()).item() * 100)
            
            image_path = data['image_path'][0]
            save_path = image_path.replace('input', 'ground_truth').replace('left_images', 'disparity_right')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            disp = right_disparity.squeeze().cpu().numpy()
            disp_scaled = disp * 128.0
            disp_scaled = np.clip(disp_scaled, 0, 65535)
            disp_scaled = disp_scaled.astype(np.uint16)
                
            cv2.imwrite(save_path, disp_scaled)
            
    # avg_valid_percentage = sum(valid_percentages) / len(valid_percentages)
    # print(f'Average Left-Right Consistency Valid Percentage: {avg_valid_percentage:.2f}%')