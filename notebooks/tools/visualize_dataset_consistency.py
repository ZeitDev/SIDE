# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml
import random

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import helpers, visualization

import mlflow
from tqdm import tqdm
import plotly.express as px

from PIL import Image
import numpy as np

from data.transforms import build_transforms

from utils import helpers

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)


import matplotlib.pyplot as plt

# %%
# Settings
dataset_name = 'EndoVis17'

# %%
# Load data

with open(os.path.join('configs', 'base.yaml'), 'r') as f: config = yaml.safe_load(f)
config['data']['dataset'] = f'data.datasets.{dataset_name}'
config['training']['tasks']['segmentation']['enabled'] = True
config['training']['tasks']['segmentation']['knowledge_distillation']['enabled'] = True
config['training']['tasks']['disparity']['enabled'] = True
config['training']['tasks']['disparity']['knowledge_distillation']['enabled'] = True

data_config = config['data']
dataset_class = helpers.load(data_config['dataset'])

train_transforms = build_transforms(config, mode='train')
dataset_train = dataset_class(
    mode='train',
    config=config,
    transforms=train_transforms,
)
helpers.check_dataleakage('train', dataset_train)

test_transforms = build_transforms(config, mode='test')
dataset_test = dataset_class(
    mode='test',
    config=config,
    transforms=test_transforms,
)
helpers.check_dataleakage('test', dataset_test)

# %%
# Visualize
# step into the middle of each 225 image subset of 8
subset = 6
#index = subset * (len(dataset_test) // 8) + (len(dataset_test) // 16)
index = 180 #random.randint(0, len(dataset_test) - 1)
print(f"Visualizing index {index}")

sample = dataset_test[index]
targets = {
    'segmentation': sample['segmentation'],
    'disparity': sample['disparity']
    }

teacher_segmentation_logits = helpers.upsample_logits(sample['teacher_segmentation'].unsqueeze(0), size=sample['image'].shape[1:]).squeeze()
teacher_disparity = helpers.logits2disparity(sample['teacher_disparity'].unsqueeze(0), size=sample['image'].shape[1:]).squeeze()

outputs = {
    'segmentation': teacher_segmentation_logits,
    'disparity': teacher_disparity
    }

visualization.get_multitask_visuals(
    image=sample['image'],
    targets=targets,
    outputs=outputs,
    num_of_segmentation_classes=2,
    max_disparity=512,
    epoch='Visualization',
    index=index
)


# %%
# Valid pixels check

def load_disparity(path):
    raw_disp = np.array(Image.open(path))
    
    valid_mask = raw_disp > 0
    raw_disp[valid_mask] = raw_disp[valid_mask] / 128
    raw_disp[~valid_mask] = 0
    
    raw_disp = np.expand_dims(raw_disp, axis=-1)
    
    return torch.from_numpy(raw_disp).float().permute(2, 0, 1)
            
    
def left_right_consistency_check(left_disp, right_disp, threshold=1.0):
    left_disp_valid = left_disp > 0
    
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
    valid_mask[~left_disp_valid] = torch.nan
    
    return valid_mask

def get_valid_percentage(valid_mask):
    agreed_count = valid_mask.nansum()
    possible_count = (~torch.isnan(valid_mask)).sum()
    return (agreed_count / possible_count).item() * 100


image_dict = {
    'left_paths': [],
    'right_paths': [],
    'valid_percentages': [],
    'stds': []
}

dataset_path = '/data/Zeitler/SIDED/EndoVis17/processed/train'
for subset in sorted(os.listdir(dataset_path)):
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.isdir(subset_path):
        continue
    
    for image_name in tqdm(sorted(os.listdir(os.path.join(subset_path, 'ground_truth', 'disparity')))):
        left_disparity_path = os.path.join(subset_path, 'ground_truth', 'disparity', image_name)
        right_disparity_path = os.path.join(subset_path, 'ground_truth', 'disparity_right', image_name)
        
        left_disp = load_disparity(left_disparity_path).unsqueeze(0)
        right_disp = load_disparity(right_disparity_path).unsqueeze(0)
        
        valid_mask = left_right_consistency_check(left_disp, right_disp, threshold=3.0)
        valid_percentage = get_valid_percentage(valid_mask)
        image_dict['valid_percentages'].append(valid_percentage)
        image_dict['left_paths'].append(left_disparity_path)
        image_dict['right_paths'].append(right_disparity_path)
        
        left_disp_valid = left_disp[left_disp > 0]
        std = left_disp_valid.std().item()
        image_dict['stds'].append(std)
    
print(f"Average valid percentage: {sum(image_dict['valid_percentages']) / len(image_dict['valid_percentages']):.2f}%")
print(f"STD of valid percentage: {torch.tensor(image_dict['valid_percentages']).std().item():.2f}%")
print(f"Average std: {sum(image_dict['stds']) / len(image_dict['stds']):.2f}")

# %%
# Histo

fig = px.histogram(image_dict['valid_percentages'], nbins=50, title="Histogram of 3 Pixel Agree Percentage for Trainset", labels={'value': 'Valid Percentage'})
fig.update_layout(xaxis_title='3 Pixel Agree Percentage (%)', yaxis_title='Count')
fig.show()

# std histogram
fig = px.histogram(image_dict['stds'], nbins=30, title="Histogram of Disparity STD", labels={'value': 'STD'})
fig.update_layout(xaxis_title='STD of Disparity', yaxis_title='Count')
fig.show()

# %%
# Plot images with low valid percentages
max_plots = 10
validity_threshold_range = (0, 10)
i = 0
for left_path, right_path, valid_percentage in zip(image_dict['left_paths'], image_dict['right_paths'], image_dict['valid_percentages']):
    if validity_threshold_range[0] <= valid_percentage <= validity_threshold_range[1] and i <= max_plots:
        left_disp = load_disparity(left_path).squeeze().numpy()
        right_disp = load_disparity(right_path).squeeze().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Left Disparity (Valid: {valid_percentage:.2f}%)")
        plt.imshow(left_disp, cmap='plasma')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.title("Right Disparity")
        plt.imshow(right_disp, cmap='plasma')
        plt.colorbar()
        
        plt.show()
        i += 1
        

# %%
# Plot images with low std
std_threshold_range = (0, 5)
i = 0
for left_path, right_path, std in zip(image_dict['left_paths'], image_dict['right_paths'], image_dict['stds']):
    if std_threshold_range[0] <= std <= std_threshold_range[1] and i <= max_plots:
        left_disp = load_disparity(left_path).squeeze().numpy()
        right_disp = load_disparity(right_path).squeeze().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Left Disparity (STD: {std:.2f})")
        plt.imshow(left_disp, cmap='plasma')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.title("Right Disparity")
        plt.imshow(right_disp, cmap='plasma')
        plt.colorbar()
        
        plt.show()
        i += 1

# %%