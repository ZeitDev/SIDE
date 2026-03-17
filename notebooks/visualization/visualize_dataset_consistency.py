# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml

import torch
import torch.nn.functional as F
from utils import helpers, visualization

from tqdm import tqdm
import plotly.express as px

from PIL import Image
import numpy as np

from data.transforms import build_transforms


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
            
    
def left_right_consistency_check(left_disp, right_disp, threshold=3.0):
    
    B, C, H, W = left_disp.shape
    x_grid = torch.linspace(-1, 1, W, device=left_disp.device)
    y_grid = torch.linspace(-1, 1, H, device=left_disp.device)
    y, x = torch.meshgrid(y_grid, x_grid, indexing='ij')
    grid = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    
    normalized_disp = (left_disp.squeeze(1) / (W / 2)).unsqueeze(-1)
    shifted_grid = grid.clone()
    shifted_grid[..., 0] -= normalized_disp[..., 0]
    
    warped_disp_right = F.grid_sample(right_disp, shifted_grid, align_corners=True, padding_mode='zeros')
    
    diff = torch.abs(left_disp - warped_disp_right)
    
    valid_mask = (diff < threshold).float()
    
    left_disp_valid = left_disp > 0
    valid_mask[~left_disp_valid] = torch.nan
    valid_warped_right = warped_disp_right > 0
    valid_mask[~valid_warped_right] = torch.nan
    
    return valid_mask

def get_valid_percentage(valid_mask):
    agreed_count = valid_mask.nansum()
    possible_count = (~torch.isnan(valid_mask)).sum()
    return (agreed_count / possible_count).item() * 100

# %%
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
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import torch

DATA_PATH = Path('/data/Zeitler/SIDED/EndoVis17/processed')
MODES = ['train', 'val', 'test']

# Unwrapped execution with aggregation to prevent OOM
num_bins = 50
pixel_bins = torch.linspace(0, 1.0, num_bins + 1)
bin_centers = (pixel_bins[:-1] + pixel_bins[1:]) / 2.0

image_records = []
pixel_hist_accum = {} # (mode, seq) -> count tensor

for mode in MODES:
    mode_path = DATA_PATH / mode
    if not mode_path.exists():
        print(f"Path does not exist: {mode_path}")
        continue
        
    for seq_path in mode_path.iterdir():
        if not seq_path.is_dir():
            continue
            
        seq = seq_path.name
        left_dir = seq_path / 'ground_truth' / 'disparity'
        right_dir = seq_path / 'ground_truth' / 'disparity_right'
        
        if not left_dir.exists() or not right_dir.exists():
            print(f"Disparity path not found for: {seq_path}")
            continue
            
        images = sorted(list(left_dir.glob('*.png')))
        pixel_hist_accum[(mode, seq)] = torch.zeros(num_bins, dtype=torch.float64)
        
        for img_path in tqdm(images, desc=f"Consistency {mode}/{seq}"):
            right_path = right_dir / img_path.name
            if not right_path.exists():
                continue
            
            left_disp = load_disparity(str(img_path)).unsqueeze(0)
            right_disp = load_disparity(str(right_path)).unsqueeze(0)
            
            valid_mask = left_right_consistency_check(left_disp, right_disp, threshold=3.0)
            
            agreed_count = valid_mask.nansum()
            possible_count = (~torch.isnan(valid_mask)).sum()
            if possible_count > 0:
                img_mean_consistency = (agreed_count / possible_count).item()
            else:
                img_mean_consistency = float('nan')
                
            image_records.append({
                'Mean Consistency': img_mean_consistency,
                'Sequence': seq,
                'Mode': mode
            })
            
            valid_mask_flat = valid_mask.flatten()
            valid_mask_flat = valid_mask_flat[~torch.isnan(valid_mask_flat)]
            
            if len(valid_mask_flat) > 0:
                counts = torch.histogram(valid_mask_flat.type(torch.float32).cpu(), bins=pixel_bins).hist
                pixel_hist_accum[(mode, seq)] += counts

pixel_agg_records = []
for (mode, seq), counts in pixel_hist_accum.items():
    for bin_ctr, count in zip(bin_centers.tolist(), counts.tolist()):
        if count > 0:
            pixel_agg_records.append({
                'Consistency': bin_ctr,
                'Count': count,
                'Sequence': seq,
                'Mode': mode
            })

# %%
df_pixels = pd.DataFrame(pixel_agg_records)
df_images = pd.DataFrame(image_records)

# if not df_pixels.empty:
#     fig1 = px.bar(
#         df_pixels, 
#         x='Consistency', 
#         y='Count',
#         color='Sequence', 
#         facet_col='Mode',
#         barmode='overlay',
#         title='Pixel-level Consistency Distribution per Sequence (Aggregated histograms)'
#     )
#     fig1.update_layout(bargap=0)
#     fig1.show()
# else:
#     print("No pixel data.")

if not df_images.empty:
    fig2 = px.histogram(
        df_images, 
        x='Mean Consistency', 
        color='Sequence', 
        facet_col='Mode',
        barmode='overlay',
        nbins=50,
        title='Image-level Mean 3px-Consistency Distribution per Sequence'
    )
    fig2.show()
else:
    print("No image data.")

# %%