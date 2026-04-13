# %%
import os
import sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

import mlflow
import plotly.graph_objects as go

from utils import helpers
from data.transforms import build_transforms
from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)

# %%
# Settings
arch = 'convnext'
run = 'wMT-KD/260406:2036/train'
task_mode = 'segmentation'
sample_indices = [226, 81] # Bad / Good
sample_metrics = {
    0: {'dice': 0.7095116376876831, 'absrel': 0.13883139193058014, 'score': 0.7780185064236758},
    1: {'dice': 0.9895262122154236, 'absrel': 0.03645792603492737, 'score': 0.9763612931583254}
}

# Final composite controls
# Set either width or height (or both) to resize every panel before stitching.
panel_width = 420
panel_height = None

# Optional overall figure scaling after stitching.
figure_scale = 1.0



# %%
# Load model
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model_run_id = helpers.get_model_run_id(run)
model = mlflow.pytorch.load_model(f'runs:/{model_run_id}/best_model_{task_mode}').to(device)
model.eval()

# %%
# Load data
config_name = run.split('/')[0]

with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
with open(os.path.join('configs', arch, config_name + '.yaml'), 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)

data_config = config['data']
dataset_class = helpers.load(data_config['dataset'])

test_transforms = build_transforms(config, mode='test')
dataset_test = dataset_class(
    mode='test',
    config=config,
    transforms=test_transforms,
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=data_config['num_workers'],
    pin_memory=data_config['pin_memory'],
    persistent_workers=False
)

# %%
if not sample_indices:
    from metrics.segmentation import Dice
    from metrics.disparity import AbsRel

    dice_metric = Dice(n_classes=config['data']['num_of_classes']['segmentation'], ignore_index=255, device=device)
    absrel_metric = AbsRel(max_disparity=512, device=device)

    sample_metrics = []

    with torch.no_grad():
        for idx, data in enumerate(dataloader_test):
            left_images = data['image'].to(device)
            right_images = data['right_image'].to(device) if 'right_image' in data else None
            
            targets_seg = data['segmentation'].to(device)
            targets_disp = data['disparity'].to(device)
            
            baseline = data['baseline'].to(device)
            focal_length = data['focal_length'].to(device)
            
            outputs = model(left_images, right_images)
            
            dice_metric.reset()
            dice_metric.update(outputs['segmentation'], targets_seg)
            dice_dict = dice_metric.compute() 
            
            val_dice = dice_dict[1]
            
            absrel_metric.reset()
            absrel_metric.update(outputs['disparity'], targets_disp, baseline, focal_length)
            absrel_val = absrel_metric.compute()['AbsRel_rate']
                
            absrel_clamped = max(0.0, min(1.0, 1.0 - absrel_val))
            denominator = val_dice + absrel_clamped
            if denominator == 0:
                val_heuristic = 0.0
            else:
                val_heuristic = (2 * val_dice * absrel_clamped) / denominator
            
            sample_metrics.append({
                'index': idx,
                'dice': val_dice,
                'absrel': absrel_val,
                'score': val_heuristic
            })

    sample_metrics.sort(key=lambda x: x['score'])
    bad_sample_idx = sample_metrics[0]['index']
    good_sample_idx = sample_metrics[-1]['index']

    print(f"Bad Sample Index: {bad_sample_idx} (DICE: {sample_metrics[0]['dice']:.4f}, AbsRel: {sample_metrics[0]['absrel']:.4f})")
    print(f"Good Sample Index: {good_sample_idx} (DICE: {sample_metrics[-1]['dice']:.4f}, AbsRel: {sample_metrics[-1]['absrel']:.4f})")
else:
    bad_sample_idx, good_sample_idx = sample_indices
    
data = {}
for i, idx in enumerate([bad_sample_idx, good_sample_idx]):
    dataset = dataset_test[idx]
    left_image = dataset['image'].unsqueeze(0).to(device)
    right_image = dataset['right_image'].unsqueeze(0).to(device) if 'right_image' in dataset else None
    
    target_segmentation = dataset['segmentation'].unsqueeze(0).to(device)
    target_disparity = dataset['disparity'].unsqueeze(0).to(device)
    
    baseline = dataset['baseline'].unsqueeze(0).to(device)
    focal_length = dataset['focal_length'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(left_image, right_image)
        
        
    data[i] = {
        'left_image': left_image.squeeze(0).cpu().numpy(),
        'target_segmentation': target_segmentation.squeeze(0).cpu().numpy(),
        'target_disparity': target_disparity.squeeze(0).cpu().numpy(),
        'output_segmentation': output['segmentation'].squeeze(0).cpu().numpy(),
        'output_disparity': output['disparity'].squeeze(0).cpu().numpy(),
        'baseline': baseline.item(),
        'focal_length': focal_length.item()
    }

# %%
import os
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Visualization Settings
max_disparity = 512.0
max_depth_viz = 150.0  # Cap for Depth visualization
seg_color = np.array([0, 255, 255])  # Cyan in RGB
seg_alpha = 0.4
error_alpha = 1.0
ERROR_GREEN = np.array([0, 200, 0])
ERROR_YELLOW = np.array([255, 255, 0])
ERROR_RED = np.array([255, 0, 0])

# Export Settings
export_dir = "poster_exports"

os.makedirs(export_dir, exist_ok=True)
print_width = 1500 # Adjusted for half an A0 poster width
save_files = False

def error_colormap(error_map):
    # error_map: float array in [0, 1]
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

def prepare_rgb(img_tensor):
    img = np.array(img_tensor)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.moveaxis(img, 0, -1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return (img * 255).astype(np.uint8)

def apply_seg_overlay(image, mask, color, alpha=0.5):
    overlay = image.copy()
    mask = np.squeeze(mask)
    valid = (mask > 0)
    overlay[valid] = overlay[valid] * (1 - alpha) + np.array(color) * alpha
    return overlay

def save_high_res(img_rgb, filename, width=print_width):
    if save_files:
        # Upscale to print size and convert RGB -> BGR for OpenCV
        h, w = img_rgb.shape[:2]
        scale = width / w
        img_resized = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img_bgr)
        print(f"Saved: {filename}")

def save_depth_high_res(depth_array, filename, max_viz, width=print_width):
    if save_files:
        # Convert arbitrary depth values to dense Magma colormap
        valid_mask = ~np.isnan(depth_array) & (depth_array > 0)
        
        # Normalize 0 -> max_viz to 0 -> 255 space
        depth_norm = np.clip(depth_array, 0, max_viz)
        depth_norm = (depth_norm / max_viz * 255).astype(np.uint8)
        
        # Apply Magma
        heatmap_bgr = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        
        # Force invalid/NaN pixels to pure white background
        heatmap_bgr[~valid_mask] = [255, 255, 255]
        
        # Upscale
        h, w = heatmap_bgr.shape[:2]
        scale = width / w
        heatmap_resized = cv2.resize(heatmap_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(filename, heatmap_resized)
        print(f"Saved: {filename}")

def save_colorbars():
    if save_files:
        # 1. Depth Colorbar (Magma_r: 255 down to 0)
        grad_depth = np.linspace(255, 0, 1000, dtype=np.uint8)
        grad_depth_2d = np.tile(grad_depth, (100, 1))
        cb_depth = cv2.applyColorMap(grad_depth_2d, cv2.COLORMAP_MAGMA)
        cv2.imwrite(os.path.join(export_dir, "Colorbar_Depth.png"), cb_depth)
        
        # 2. Error Colorbar (Green -> Yellow -> Red)
        grad_err = np.linspace(0, 1.0, 1000)
        cb_err = np.zeros((100, 1000, 3), dtype=np.uint8)
        for i, val in enumerate(grad_err):
            if val <= 0.5:
                r, g, b = int(255 * (2 * val)), int(128 + 127 * (2 * val)), 0
            else:
                r, g, b = 255, int(255 * (2.0 - 2 * val)), 0
            cb_err[:, i] = [b, g, r] # BGR for OpenCV
        cv2.imwrite(os.path.join(export_dir, "Colorbar_Error.png"), cb_err)
        print("Saved standalone colorbars.")

# Create Plotly subplots strictly for Notebook Viewing
fig = make_subplots(
    rows=4, cols=3, column_widths=[1, 1, 1],
    vertical_spacing=0.01, horizontal_spacing=0.0001,
    subplot_titles=("<b>Prediction</b>", "<b>Target</b>", "<b>Error</b>",
                    "", "", "",
                    "", "", "",
                    "", "", "")
)

for sample_idx, data_idx in enumerate([1, 0]): # Good first (1), Bad second (0)
    row_offset = 1 if sample_idx == 0 else 3
    sample = data[data_idx]
    prefix = "GoodSample" if sample_idx == 0 else "BadSample"
    
    rgb_img = prepare_rgb(sample['left_image'])
    f = sample['focal_length']
    B = sample['baseline']

    # --- Segmentation Row (Row 1 / 3) ---
    out_seg = sample['output_segmentation']
    if out_seg.ndim == 3 and out_seg.shape[0] < out_seg.shape[-1]: pred_seg_mask = out_seg.argmax(axis=0)
    elif out_seg.ndim == 3: pred_seg_mask = out_seg.argmax(axis=-1)
    else: pred_seg_mask = out_seg.squeeze()
        
    target_seg_mask = sample['target_segmentation'].squeeze()
    
    pred_seg_overlay = apply_seg_overlay(rgb_img, pred_seg_mask, seg_color, seg_alpha)
    target_seg_overlay = apply_seg_overlay(rgb_img, target_seg_mask, seg_color, seg_alpha)
    
    # Error Image for Segmentation
    # Green where agreeing, Red where disagreeing
    seg_match = (pred_seg_mask == target_seg_mask)
    # Background class is typically 255. 
    valid_mask = target_seg_mask != 255
    # 1. Create a float error map: 0 for correct, 1 for incorrect
    seg_error = np.zeros_like(target_seg_mask, dtype=np.float32)
    seg_error[valid_mask & ~seg_match] = 1.0  # Incorrect = 1, correct = 0

    # 2. Map error to color (green→yellow→red)
    seg_err_color = np.zeros((*seg_error.shape, 3), dtype=np.uint8)
    mask1 = seg_error <= 0.5
    mask2 = seg_error > 0.5

    seg_err_color = error_colormap(seg_error)

    # 3. Blend with the original image
    seg_err_img = rgb_img.copy()
    seg_err_img[valid_mask] = (
        rgb_img[valid_mask] * (1 - error_alpha) +
        seg_err_color[valid_mask] * error_alpha
    ).astype(np.uint8)
    
    # Save Segments directly to disk
    save_high_res(pred_seg_overlay, os.path.join(export_dir, f"{prefix}_Seg_Prediction.png"))
    save_high_res(target_seg_overlay, os.path.join(export_dir, f"{prefix}_Seg_GroundTruth.png"))
    save_high_res(seg_err_img, os.path.join(export_dir, f"{prefix}_Seg_Error.png"))
    
    fig.add_trace(go.Image(z=pred_seg_overlay), row=row_offset, col=1)
    fig.add_trace(go.Image(z=target_seg_overlay), row=row_offset, col=2)
    fig.add_trace(go.Image(z=seg_err_img), row=row_offset, col=3)
    
    # --- Depth Row (Row 2 / 4) ---
    pred_disp = sample['output_disparity'].squeeze() * max_disparity
    target_disp = sample['target_disparity'].squeeze() * max_disparity
    
    pred_depth = np.divide(f * B, pred_disp, out=np.zeros_like(pred_disp), where=(pred_disp > 0))
    target_depth = np.divide(f * B, target_disp, out=np.zeros_like(target_disp), where=(target_disp > 0))
    
    pred_depth[pred_disp <= 0] = np.nan
    target_depth[target_disp <= 0] = np.nan
    
    # Error Image for Depth (AbsRel)
    absrel_err = np.abs(pred_depth - target_depth) / np.where(target_depth > 0, target_depth, 1e-8)
    absrel_err[np.isnan(absrel_err)] = 0.0
    absrel_err = np.clip(absrel_err, 0, 1.0) # 0 to 100%
    
    # Save Depths directly to disk
    save_depth_high_res(pred_depth, os.path.join(export_dir, f"{prefix}_Depth_Prediction.png"), max_depth_viz)
    save_depth_high_res(target_depth, os.path.join(export_dir, f"{prefix}_Depth_GroundTruth.png"), max_depth_viz)
    
    # --- Option 1: Alpha-Blended Depth Error ---
    
    # 1. Create an RGB version of the error heatmap
    # 1. Create an RGB version of the error heatmap with PURE colors
    depth_err_color = np.zeros((*absrel_err.shape, 3), dtype=np.uint8)
    mask1 = absrel_err <= 0.5
    mask2 = absrel_err > 0.5
        
    # Green -> Yellow map (Error 0.0 to 0.5)
    # Starts at [0, 255, 0] (Pure Green) and goes to [255, 255, 0] (Pure Yellow)
    # Green -> Yellow map (Error 0.0 to 0.5)
    # Starts at [0, 128, 0] (Standard Green) and goes to [255, 255, 0] (Pure Yellow)
    depth_err_color = error_colormap(absrel_err)                                              # Blue stays 0
    
    # 2. Blend with the original tissue image using the same alpha
    valid_depth = target_depth > 0
    blended_depth_err = rgb_img.copy()
    blended_depth_err[valid_depth] = (
        rgb_img[valid_depth] * (1 - error_alpha) + 
        depth_err_color[valid_depth] * error_alpha
    ).astype(np.uint8)
    
    # Save blended map to disk (converting RGB back to BGR for OpenCV)
    save_high_res(cv2.cvtColor(blended_depth_err, cv2.COLOR_RGB2BGR), os.path.join(export_dir, f"{prefix}_Depth_Error.png"))
    
    # --- Plotly Traces ---
    fig.add_trace(go.Heatmap(z=pred_depth[::-1], colorscale='Magma_r', zmin=0, zmax=max_depth_viz, showscale=False), row=row_offset + 1, col=1)
    
    fig.add_trace(
        go.Heatmap(
            z=target_depth[::-1], colorscale='Magma_r', zmin=0, zmax=max_depth_viz, 
            showscale=(sample_idx == 0), 
            colorbar=dict(
                title=dict(text="Depth [mm]", side="right", font=dict(size=2 * 1)), 
                x=1.005, len=1.0, y=0.5, yanchor='middle'
            )
        ), 
        row=row_offset + 1, col=2
    )
    
    # Plot the Blended Image for the error column
    fig.add_trace(go.Image(z=blended_depth_err), row=row_offset + 1, col=3)
    
    # Dummy invisible heatmap purely to keep the Colorbar on the right side
    #green_red_scale = [[0.0, ERROR_GREEN], [0.5, ERROR_YELLOW], [1.0, ERROR_RED]]
    green_red_scale = [
        [0.0, f'rgb({ERROR_GREEN[0]}, {ERROR_GREEN[1]}, {ERROR_GREEN[2]})'],
        [0.5, f'rgb({ERROR_YELLOW[0]}, {ERROR_YELLOW[1]}, {ERROR_YELLOW[2]})'],
        [1.0, f'rgb({ERROR_RED[0]}, {ERROR_RED[1]}, {ERROR_RED[2]})']
    ]
    fig.add_trace(
        go.Heatmap(
            z=[[0]],                 # <-- Valid data so Plotly doesn't panic
            opacity=0,               # <-- Makes the heatmap invisible
            colorscale=green_red_scale, 
            zmin=0, zmax=1.0, 
            showscale=(sample_idx == 0), 
            colorbar=dict(
                title=dict(text="Error", side="right", font=dict(size=2 * 1)), 
                x=1.1,              # <-- Tucked right next to the Depth colorbar
                len=1.0, y=0.5, yanchor='middle'
            )
        ), 
        row=row_offset + 1, col=3
    )


# Small cleanup for the notebook visualization
fig.update_xaxes(showticklabels=False, visible=False)
fig.update_yaxes(showticklabels=False, visible=False)
for i in range(1, 13): 
    axis_suffix = "" if i == 1 else str(i)
    fig.layout[f"yaxis{axis_suffix}"].update(scaleanchor=f"x{axis_suffix}", scaleratio=1)

# %%
#fig.add_annotation(text="<b>Good Sample</b>", xref="paper", yref="paper", x=0.5, y=1.03, showarrow=False, font=dict(size=18), xanchor="center", yanchor="bottom")
#fig.add_annotation(text="<b>Bad Sample</b>", xref="paper", yref="paper", x=0.5, y=0.51, showarrow=False, font=dict(size=18), xanchor="center", yanchor="bottom")
# --- POSTER EXPORT SETTINGS ---
# 1. Calculate physical dimensions
a0_width_inches = 33.11
minipage_fraction = 0.45
target_width_inches = a0_width_inches * minipage_fraction

# Plotly uses 96 pixels per inch for layout calculations
scale_factor = 1
layout_dpi = 96
target_width_px = int(target_width_inches * layout_dpi * scale_factor)
target_height_px = int(target_width_px * 1.115) # Adjust this multiplier to tune the height

font_size = 36

fig.add_annotation(
    text=f"<b>Good Sample</b> (DICE: {sample_metrics[1]['dice']*100:.0f}%, AbsRel: {sample_metrics[1]['absrel']*100:.0f}%)",
    xref="paper", yref="paper",
    x=-0.034, y=0.96, 
    textangle=-90,
    showarrow=False,
    font=dict(size=font_size * scale_factor) # Make these prominent
)

# Bad Sample (Spans Rows 3 & 4 -> centered around y=0.25 in paper coordinates)
fig.add_annotation(
    text=f"<b>Bad Sample</b> (DICE: {sample_metrics[0]['dice']*100:.0f}%, AbsRel: {sample_metrics[0]['absrel']*100:.0f}%)",
    xref="paper", yref="paper",
    x=-0.034, y=0.04, 
    textangle=-90,
    showarrow=False,
    font=dict(size=font_size * scale_factor)
)

# 2. Update layout with exact dimensions and matching poster fonts
fig.update_layout(
    width=target_width_px,
    height=target_height_px,
    margin=dict(l=50 * scale_factor, r=160 * scale_factor, t=50 * scale_factor, b=10 * scale_factor), # Adjusted right margin to give colorbars breathing room
    plot_bgcolor='white', 
    paper_bgcolor='white',
    font=dict(
        family="Helvetica", # Change to match your LaTeX poster font (e.g., 'Computer Modern')
        size=font_size * scale_factor, # This size will accurately reflect an 18pt font on the printed A0 poster
        color="#19038B"
    )
)

# Optional: Tweak colorbar font sizes to be legible but not overpowering
fig.update_traces(
    colorbar_tickfont_size=28 * scale_factor, 
    colorbar_title_font_size=28 * scale_factor, 
    selector=dict(type='heatmap')
)

for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(size=font_size * scale_factor) # Slightly larger than base text for headers

# 4. Add Row Labels (Vertical labels spanning 2 rows each)
# Good Sample (Spans Rows 1 & 2 -> centered around y=0.75 in paper coordinates)


# 3. Export directly to a single high-res image
# You will need the 'kaleido' package installed to write images: pip install -U kaleido
#export_path = os.path.join(export_dir, "inference.png")

# Using scale=3 bumps the internal layout (96 DPI) up to print-quality resolution (288 DPI)
# This keeps the text crisp but prevents LaTeX from crashing since it's a single raster image.
#print(f"Successfully saved poster-ready figure to: {export_path}")

config = {
    'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'inference',
        'scale': 3 # Multiplies the resolution without breaking the layout!
    }
}
# Still show in the notebook for a quick sanity check
fig.show(config=config)

# %%
#fig.write_image(export_path, scale=1)

# %%
