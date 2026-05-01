# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

os.chdir('/data/Zeitler/code/SIDE')

# %%
# Functions needed for the consistency check
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
    
    normalized_disp = (left_disp.squeeze(1) / ((W - 1) / 2)).unsqueeze(-1)
    shifted_grid = grid.clone()
    shifted_grid[..., 0] -= normalized_disp[..., 0]
    
    # Warp disparity
    warped_disp_right = F.grid_sample(right_disp, shifted_grid, align_corners=True, padding_mode='zeros')
    
    right_valid_mask = (right_disp > 0).float()
    warped_right_valid_mask = F.grid_sample(right_valid_mask, shifted_grid, mode='nearest', align_corners=True, padding_mode='zeros')
    
    diff = torch.abs(left_disp - warped_disp_right)
    valid_mask = (diff <= threshold).float()
    
    # Apply invalid masks
    left_disp_valid = left_disp > 0
    valid_mask[~left_disp_valid] = torch.nan
    
    # Use the nearest-interpolated mask to reject blended border zeros
    valid_mask[warped_right_valid_mask == 0] = torch.nan 
    
    return valid_mask

# %%
# Main processing
DATA_PATH = Path('/data/Zeitler/SIDED/EndoVis17/processed')
MODES = ['train', 'val', 'test']

image_records = []

for mode in MODES:
    mode_path = DATA_PATH / mode
    if not mode_path.exists():
        print(f"Path does not exist: {mode_path}")
        continue
        
    for seq_path in mode_path.iterdir():
        if not seq_path.is_dir():
            continue
            
        seq = seq_path.name
        left_dir = seq_path / 'target' / 'disparity'
        right_dir = seq_path / 'target' / 'disparity_right'
        
        if not left_dir.exists() or not right_dir.exists():
            continue
            
        images = sorted(list(left_dir.glob('*.png')))
        
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

# %%
df_images = pd.DataFrame(image_records)

# Use regex to extract the trailing number from "instrument_dataset_X" and format it as 2-digit "0X"
df_images['SequenceLabel'] = df_images['Sequence'].str.extract(r'(\d+)')[0].astype(int).map(lambda x: f"{x:02d}")

# Map Mode to readable names
mode_map = {'train': 'Train Set', 'val': 'Validation Set', 'test': 'Test Set'}
df_images['ModeLabel'] = df_images['Mode'].map(mode_map)

# Explicitly map colors
sequences = sorted(df_images['SequenceLabel'].unique())
colors = px.colors.qualitative.Plotly
color_map = {seq: colors[i % len(colors)] for i, seq in enumerate(sequences)}
color_map['10'] = 'darkcyan'
color_map['07'] = 'goldenrod'

# Create Plot
fig2 = px.histogram(
    df_images,
    x='Mean Consistency',
    color='SequenceLabel',
    facet_col='ModeLabel',
    barmode='overlay',
    nbins=50,
    category_orders={
        'SequenceLabel': sequences,
        'ModeLabel': ['Train Set', 'Validation Set', 'Test Set']
    },
    color_discrete_map=color_map,
    labels={'SequenceLabel': 'Sequence', 'Mean Consistency': 'Mean 3px Agreement', 'count': 'Frames'},
    orientation='v'
)

# Remove facet title prefix
for annotation in fig2.layout.annotations:
    if annotation.text.startswith('ModeLabel='):
        annotation.text = annotation.text.replace('ModeLabel=', '')

# Formatting
font_size = 15.5
family = 'Latin Modern Roman, Computer Modern Roman, serif'
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
    yaxis_title='Count'
)
fig2.update_traces(xbins=dict(start=0.0, end=1.0, size=0.02))

from notebooks.figures.helpers import save_figure
save_figure(fig2, height=320, name='left_right_consistency', lrtb_margin=(45, 10, 20, 0), standoff=0)

# %%
# Process visualization for Sequence 1 using Plotly
import plotly.subplots as sp
import plotly.graph_objects as go

def error_colormap(error_map):
    # error_map: float array in [0, 1]
    ERROR_GREEN = np.array([0, 200, 0])
    ERROR_YELLOW = np.array([255, 255, 0])
    ERROR_RED = np.array([255, 0, 0])
    
    color = np.zeros((*error_map.shape, 3), dtype=np.uint8)
    mask1 = error_map <= 0.5
    mask2 = error_map > 0.5

    # Green to Yellow (0.0 to 0.5)
    color[mask1] = (
        ERROR_GREEN * (1 - 2 * error_map[mask1, None]) +
        ERROR_YELLOW * (2 * error_map[mask1, None])
    ).astype(np.uint8)

    # Yellow to Red (0.5 to 1.0)
    color[mask2] = (
        ERROR_YELLOW * (2 - 2 * error_map[mask2, None]) +
        ERROR_RED * (2 * error_map[mask2, None] - 1)
    ).astype(np.uint8)

    return color

seq1_path = DATA_PATH / 'train' / 'instrument_dataset_1'
left_dir = seq1_path / 'target' / 'disparity'
right_dir = seq1_path / 'target' / 'disparity_right'
image_dir = seq1_path / 'input' / 'left_images'

img_name = sorted(list(left_dir.glob('*.png')))[0].name
left_disp_path = left_dir / img_name
right_disp_path = right_dir / img_name
left_image_path = image_dir / img_name

# Load and compute
left_disp_tensor = load_disparity(str(left_disp_path)).unsqueeze(0)
right_disp_tensor = load_disparity(str(right_disp_path)).unsqueeze(0)
left_image_pil = Image.open(left_image_path)
left_image = np.array(left_image_pil)
if left_image.max() <= 1.0:
    left_image = (left_image * 255).astype(np.uint8)

valid_mask = left_right_consistency_check(left_disp_tensor, right_disp_tensor, threshold=3.0)
mean_consistency = (valid_mask.nansum() / (~torch.isnan(valid_mask)).sum()).item()

# Calculate error for overlay
B, C, H, W = left_disp_tensor.shape
x_grid = torch.linspace(-1, 1, W)
y_grid = torch.linspace(-1, 1, H)
y, x = torch.meshgrid(y_grid, x_grid, indexing='ij')
grid = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
normalized_disp = (left_disp_tensor.squeeze(1) / (W / 2)).unsqueeze(-1)
shifted_grid = grid.clone()
shifted_grid[..., 0] -= normalized_disp[..., 0]
warped_right = F.grid_sample(right_disp_tensor, shifted_grid, align_corners=True, padding_mode='zeros')

# Consistency error approach from inference_figure.py
# 0 is perfect match, > 3.0 is mismatch. Threshold is 3px.
diff = torch.abs(left_disp_tensor - warped_right).squeeze().numpy()
error_norm = np.clip(diff / 3.0, 0, 1.0) # 0 to 1 scaling based on threshold

# Map to color
err_color = error_colormap(error_norm)

# Instead of blending with the original image, create a black background map
valid_region = (left_disp_tensor.squeeze() > 0).numpy() & (warped_right.squeeze() > 0).numpy()
overlay = np.zeros((*diff.shape, 3), dtype=np.uint8)  # Black background for invalid regions
overlay[valid_region] = err_color[valid_region]

# %%
# Plotting with Plotly
fig3 = sp.make_subplots(
    rows=1, cols=3,
    subplot_titles=("Left-Sided<br>Disparity", "Right-Sided<br>Disparity", f"3px Agreement<br>({mean_consistency:.1%})"),
    horizontal_spacing=0.01
)

left_disp_np = left_disp_tensor.squeeze().numpy().copy()
right_disp_np = right_disp_tensor.squeeze().numpy().copy()

# Set invalid to 0, which maps to black at the bottom of the Magma colorscale
left_disp_np[left_disp_np <= 0] = 0
right_disp_np[right_disp_np <= 0] = 0

# Flip the arrays vertically [::-1] because Plotly heatmaps draw from bottom up
# Disparity corresponds to 'Magma' to match 'Magma_r' for Depth
fig3.add_trace(go.Heatmap(z=left_disp_np[::-1], colorscale='Magma', showscale=False), row=1, col=1)

fig3.add_trace(
    go.Heatmap(
        z=right_disp_np[::-1], colorscale='Magma', 
        showscale=True, 
        colorbar=dict(
            title=dict(text="Disparity [px]", side="right", font=dict(size=14)), 
            thickness=12,
            x=1.01, len=1.0, y=0.5, yanchor='middle'
        )
    ), 
    row=1, col=2
)

# Image trace natively draws top-to-bottom, but axes are linked so we must flip the image too
# Removing the [::-1] flip because linked axes are already handling the orientation 
# for consistency between Heatmap and Image in this specific setup.
fig3.add_trace(go.Image(z=overlay), row=1, col=3)

# Add matching colorbar for Difference
ERROR_GREEN = np.array([0, 200, 0])
ERROR_YELLOW = np.array([255, 255, 0])
ERROR_RED = np.array([255, 0, 0])
green_red_scale = [
    [0.0, f'rgb({ERROR_GREEN[0]}, {ERROR_GREEN[1]}, {ERROR_GREEN[2]})'],
    [0.5, f'rgb({ERROR_YELLOW[0]}, {ERROR_YELLOW[1]}, {ERROR_YELLOW[2]})'],
    [1.0, f'rgb({ERROR_RED[0]}, {ERROR_RED[1]}, {ERROR_RED[2]})']
]
fig3.add_trace(
    go.Heatmap(
        z=[[0]], opacity=0, colorscale=green_red_scale, zmin=0, zmax=3.0, 
        showscale=True, 
        colorbar=dict(
            title=dict(text="Difference [px]", side="right", font=dict(size=14)), 
            thickness=12,
            x=1.17, len=1.0, y=0.5, yanchor='middle',
            tickvals=[0, 1, 2, 3],
            ticktext=['0', '1', '2', '>3']
        )
    ), 
    row=1, col=3
)

# Apply visual formatting strict to inference_figure.py
fig3.update_xaxes(showticklabels=False, visible=False)
fig3.update_yaxes(showticklabels=False, visible=False)
for i in range(1, 4): 
    axis_suffix = "" if i == 1 else str(i)
    fig3.layout[f"yaxis{axis_suffix}"].update(scaleanchor=f"x{axis_suffix}", scaleratio=1)

fig3.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
)

save_figure(fig3, height=170, name='consistency_process', lrtb_margin=(10, 140, 40, 0))

# %%

