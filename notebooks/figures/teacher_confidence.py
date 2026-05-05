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
# Plotting
font_size = 15.5
family = 'Latin Modern Roman, Computer Modern Roman, serif'

fig = sp.make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Left Image", 
        "Right Image", 
        "Teacher's Logits<br>as Disparity Map", 
        f"Teacher Pixel Confidence<br>(Mean: {mean_conf:.1%})"
    ),
    horizontal_spacing=0.15,
    vertical_spacing=0.08
)

# 1. Left Image
fig.add_trace(go.Image(z=left_img_cropped), row=1, col=1)

# 2. Right Image
fig.add_trace(go.Image(z=right_img_cropped), row=1, col=2)

# 3. Disparity Map (Magma)
disp_for_plot = teach_disp_np.copy()
disp_for_plot[disp_for_plot <= 0] = 0

fig.add_trace(
    go.Heatmap(
        z=disp_for_plot, 
        colorscale='Magma', 
        zmin=0, zmax=max_disparity,
        showscale=True,
        colorbar=dict(
            title=dict(text="Disparity [px]", side="right", font=dict(size=font_size)),
            thickness=12,
            x=0.425, len=0.47, y=0.23, yanchor='middle'
        )
    ), 
    row=2, col=1
)

# 4. Confidence Map
# Red-Yellow-Green scale
ryg_scale = [
    [0.0, 'rgb(255, 0, 0)'],    # Red
    [0.5, 'rgb(255, 255, 0)'],  # Yellow
    [1.0, 'rgb(0, 200, 0)']     # Green
]

fig.add_trace(
    go.Heatmap(
        z=conf_np, 
        colorscale=ryg_scale, 
        zmin=0, zmax=1.0,
        showscale=True,
        colorbar=dict(
            title=dict(text="Confidence [%]", side="right", font=dict(size=font_size)),
            thickness=12,
            x=1.01, len=0.47, y=0.23, yanchor='middle',
            tickvals=[0, 0.5, 1.0],
            ticktext=['0', '50', '100']
        )
    ),
    row=2, col=2
)

fig.update_xaxes(showticklabels=False, visible=False)
fig.update_yaxes(showticklabels=False, visible=False, autorange='reversed')

for i in range(1, 5): 
    axis_suffix = "" if i == 1 else str(i)
    fig.layout[f"yaxis{axis_suffix}"].update(scaleanchor=f"x{axis_suffix}", scaleratio=1)

fig.update_layout(
    font=dict(family=family, size=font_size, color='black'),
    plot_bgcolor='white',
    paper_bgcolor='white',
)

# Fix standoff: Decrease annotation Y positions slightly to pull titles closer to plots
for annot in fig.layout.annotations:
    annot.update(yshift=-4)

save_figure(fig, height=520, name='teacher_confidence_example', lrtb_margin=(10, 80, 20, 10), standoff=0)

# %%
# Figure 2: Confidence of all samples per sequence

image_records = []
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
            
            save_path = seq_path / 'teacher' / 'disparity_confidence' / pt_file.name.replace('.pt', '.png')
            os.makedirs(save_path.parent, exist_ok=True)
            cv2.imwrite(str(save_path), conf_scaled)
            
            img_mean_conf = conf.mean().item()
            image_records.append({
                'Mean Confidence': img_mean_conf,
                'Sequence': seq,
                'Mode': mode_name
            })
            

# %%
df_images = pd.DataFrame(image_records)

# Consistency with left_right_consistency.py labeling
df_images['SequenceLabel'] = df_images['Sequence'].str.extract(r'(\d+)')[0].astype(int).map(lambda x: f"{x:02d}")
mode_map = {'train': 'Train Set', 'val': 'Validation Set', 'test': 'Test Set'}
df_images['ModeLabel'] = df_images['Mode'].map(mode_map)

# Color mapping matching left_right_consistency.py
sequences = sorted(df_images['SequenceLabel'].unique())
colors = px.colors.qualitative.Plotly
color_map = {seq: colors[i % len(colors)] for i, seq in enumerate(sequences)}
color_map['10'] = 'darkcyan'
color_map['07'] = 'goldenrod'

fig2 = px.histogram(
    df_images,
    x='Mean Confidence',
    color='SequenceLabel',
    facet_col='ModeLabel',
    barmode='overlay',
    histnorm='percent',
    #nbins=50,
    category_orders={
        'SequenceLabel': sequences,
        'ModeLabel': ['Train Set', 'Validation Set', 'Test Set']
    },
    color_discrete_map=color_map,
    labels={'SequenceLabel': 'Sequence', 'Mean Confidence': 'Mean Teacher Confidence', 'count': 'Frames'},
    orientation='v'
)
fig2.update_xaxes(tickvals=[0, 0.5, 1.0], range=[0, 1.0])
fig2.update_yaxes(range=[0, 100])

# Remove facet title prefix
for annotation in fig2.layout.annotations:
    if annotation.text.startswith('ModeLabel='):
        annotation.text = annotation.text.replace('ModeLabel=', '')

# Formatting
fig2.update_layout(
    font=dict(family=family, size=font_size, color='black'),
    title=None,
    legend_title_text='Sequence',
    legend=dict(
        title_font_size=font_size,
        font_size=font_size,
        orientation="h",
        y=-0.2,
        entrywidth=0.2,
        entrywidthmode='fraction'
    ),
    yaxis_title='Percent of Sequence (%)'
)
fig2.update_traces(xbins=dict(start=0.0, end=1.0, size=0.02))

save_figure(fig2, height=320, name='teacher_confidence', lrtb_margin=(45, 10, 20, 0), standoff=0)

# %%
