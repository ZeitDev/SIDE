# %%
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
from PIL import Image
from pathlib import Path
import plotly.graph_objects as go
import plotly.subplots as sp

import cv2

sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))
os.chdir('/data/Zeitler/code/SIDE')

from utils.helpers import logits2disparity
from notebooks.figures.helpers import save_figure

import math

def entropy_confidence(logits, c_min=0, c_max=1, prob_dim=1):
    probs = F.softmax(logits, dim=prob_dim)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=prob_dim)
    num_bins = logits.shape[prob_dim]
    max_entropy = math.log2(num_bins)
    normalized_entropy = entropy / max_entropy
    base_confidence = 1.0 - normalized_entropy
    
    denominator = max(c_max - c_min, 1e-6)
    
    scaled_conf = (base_confidence - c_min) / denominator
    final_confidence = torch.clamp(scaled_conf, min=0.0, max=1.0)
    
    return final_confidence

if False:
    image_records = []
    # Number of random pixels to sample per image to prevent RAM crashes
    SAMPLES_PER_IMAGE = 2000 

    for mode_name in ['train', 'val', 'test']:
        mode_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode_name}')
        if not mode_path.exists():
            continue
            
        for seq_path in mode_path.iterdir():
            if not seq_path.is_dir():
                continue
                
            seq = seq_path.name
            logits_path = seq_path / 'teacher' / 'disparity_128_256_256'
            
            if not logits_path.exists():
                continue
                
            pt_files = list(logits_path.glob('*.pt'))
            
            for pt_file in tqdm(pt_files, desc=f"Collecting pixels {mode_name}/{seq}"):
                logits = torch.load(pt_file, map_location='cpu', weights_only=True).float()
                
                if logits.dim() == 4:
                    conf = entropy_confidence(logits, prob_dim=1).squeeze(0)
                elif logits.dim() == 3:
                    conf = entropy_confidence(logits, prob_dim=0).squeeze(0)
                else:
                    continue
                
                # --- THE FIX: Random Subsampling ---
                flat_conf = conf.flatten()
                
                # Generate random indices and select the pixels
                # (If an image is somehow smaller than our sample size, we take what we can)
                actual_samples = min(SAMPLES_PER_IMAGE, flat_conf.numel())
                indices = torch.randperm(flat_conf.numel())[:actual_samples]
                sampled_pixels = flat_conf[indices].tolist()
                
                # Append the sampled pixels to our records
                for pixel_val in sampled_pixels:
                    image_records.append({
                        'Pixel Confidence': pixel_val,
                        'Sequence': seq,
                        'Mode': mode_name
                    })
                    
    df_pixels = pd.DataFrame(image_records)

    # --- CALCULATE YOUR GLOBAL LIMITS ---
    # Let Pandas calculate the exact dataset-wide percentiles!
    dataset_c_min = df_pixels['Pixel Confidence'].quantile(0.05)
    dataset_c_max = df_pixels['Pixel Confidence'].quantile(0.95)

    print("\n" + "="*40)
    print(f"STATISTICS COMPLETE (Sampled {len(df_pixels):,} pixels)")
    print(f"Calculated c_min (5th Percentile) : {dataset_c_min:.4f}")
    print(f"Calculated c_max (95th Percentile): {dataset_c_max:.4f}")
    print("="*40 + "\n")
    

# %%
# Settings
mode = 'val' # 'train', 'val', or 'test'
dataset_name = 'instrument_dataset_5'
frame_idx = 199 # Specific frame to select
max_disparity = 512.0

# Paths
dataset_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}/{dataset_name}')
left_dir = dataset_path / 'input' / 'left_images'
right_dir = dataset_path / 'input' / 'right_images'
teach_disp_dir = dataset_path / 'teacher' / 'disparity_128_256_256'

left_images = sorted(list(left_dir.glob('*.png')))
img_name = left_images[frame_idx].name
pt_filename = img_name.replace('.png', '.pt')

left_image_path = left_dir / img_name
right_image_path = right_dir / img_name
teach_disp_path = teach_disp_dir / pt_filename

# Load Images
left_img = Image.open(left_image_path)
right_img = Image.open(right_image_path)

w, h = left_img.size
crop_w, crop_h = 1024, 1024
top = max(0, (h - crop_h) // 2)
left_offset = max(0, (w - crop_w) // 2)

# Crop images
left_img_cropped = np.array(left_img)[top:top+crop_h, left_offset:left_offset+crop_w]
right_img_cropped = np.array(right_img)[top:top+crop_h, left_offset:left_offset+crop_w]

# Load Teacher
teach_disp_pt = torch.load(teach_disp_path, weights_only=True).float()
if teach_disp_pt.dim() == 3: 
    teach_disp_pt = teach_disp_pt.unsqueeze(0)

# Disparity Map
teach_disp_up = logits2disparity(teach_disp_pt, (crop_h, crop_w)) * 512.0
teach_disp_np = np.nan_to_num(teach_disp_up.squeeze().cpu().numpy(), nan=0.0)

# Confidence Map
teach_disp_logits_up = F.interpolate(teach_disp_pt, size=(crop_h, crop_w), mode='bilinear', align_corners=False)
#conf_pt = F.softmax(teach_disp_logits_up, dim=1).max(dim=1)[0]
conf_pt = entropy_confidence(teach_disp_logits_up, prob_dim=1).squeeze(0)
conf_np = conf_pt.squeeze().cpu().numpy()
mean_conf = conf_np.mean()

# %%
# Confidence of all samples per sequence
resolution = 512 # 1024 / 512
logit_resolution = resolution // 4
channels = 256 # 512 / 256
logit_channels = channels // 4

image_records = []
for mode_name in ['train', 'val', 'test']:
    mode_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode_name}')
    if not mode_path.exists():
        continue
        
    for seq_path in mode_path.iterdir():
        if not seq_path.is_dir():
            continue
            
        seq = seq_path.name
        logits_path = seq_path / 'teacher' / f'disparity_{logit_channels}_{logit_resolution}_{logit_resolution}'
        
        if not logits_path.exists():
            continue
            
        pt_files = list(logits_path.glob('*.pt'))
        
        for pt_file in tqdm(pt_files, desc=f"Collecting confidence {mode_name}/{seq}"):
            logits = torch.load(pt_file, map_location='cpu', weights_only=True).float()
            
            if logits.dim() == 4:
                #probs = F.softmax(logits, dim=1)
                conf = entropy_confidence(logits, c_min=0.39, c_max=0.8981, prob_dim=1).squeeze(0)
            elif logits.dim() == 3:
                #probs = F.softmax(logits, dim=0)
                conf = entropy_confidence(logits, c_min=0.39, c_max=0.8981, prob_dim=0).squeeze(0)
            else:
                continue
            
            conf_np = conf.cpu().numpy()
            conf_scaled = np.round(conf_np * 65535.0)
            conf_scaled = conf_scaled.astype(np.uint16)
            
            save_path = seq_path / 'teacher' / f'disparity_confidence_1_{logit_resolution}_{logit_resolution}' / pt_file.name.replace('.pt', '.png')
            os.makedirs(save_path.parent, exist_ok=True)
            cv2.imwrite(str(save_path), conf_scaled)
            
            img_mean_conf = conf.mean().item()
            image_records.append({
                'Mean Confidence': img_mean_conf,
                'Sequence': seq,
                'Mode': mode_name
            })
