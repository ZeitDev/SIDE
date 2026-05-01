# %% Imports
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# %% Settings
seg_classes = 8
base_dataset_path = '/data/Zeitler/SIDED/EndoVis17/processed'

# Font settings
font_size = 15
#plt.rcParams['font.family'] = 'Latin Modern Roman, serif'
plt.rcParams['font.size'] = font_size


# Load class name mappings
with open(f'{base_dataset_path}/mapping_8.json', 'r') as f:
    class_mapping = json.load(f)
# Reverse mapping: number -> name
class_names = {v: k for k, v in class_mapping.items()}

# Abbreviation map
abbr_map = {
    "bipolar_forceps": "BF",
    "prograsp_forceps": "PF",
    "large_needle_driver": "LND",
    "vessel_sealer": "VS",
    "grasping_retractor": "GR",
    "monopolar_curved_scissors": "MCS",
    "other": "Other"
}

# Use abbreviations for class names in legend
class_abbr = {}
for v, name in class_names.items():
    abbr = abbr_map.get(name, name.replace('_', ' ').title())
    class_abbr[v] = abbr

# Collect subsets from both train and test modes
subset_frames = {}
for mode in ['train', 'val', 'test']:
    dataset_path = f'{base_dataset_path}/{mode}'
    if os.path.exists(dataset_path):
        subsets = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        for sub in subsets:
            left_dir = os.path.join(dataset_path, sub, 'input', 'left_images')
            if os.path.exists(left_dir):
                frames = sorted(os.listdir(left_dir))
                if frames:
                    subset_key = f'{mode}/{sub}'  # Include mode in subset name
                    subset_frames[subset_key] = (mode, sub, frames)

if not subset_frames:
    print("No valid directories found.")
else:
    valid_subsets = sorted(list(subset_frames.keys()))

# %% Interactive Viewer
def show_images(subset_name, frame_idx):
    mode, subset, frames = subset_frames[subset_name]
    dataset_path = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}'
    
    if frame_idx >= len(frames):
        frame_idx = len(frames) - 1
        
    frame_name = frames[frame_idx]
    
    left_img_path = os.path.join(dataset_path, subset, 'input', 'left_images', frame_name)
    seg_img_path = os.path.join(dataset_path, subset, 'target', f'segmentation_{seg_classes}', frame_name)
    
    # Load and convert Left Image for matplotlib (BGR -> RGB)
    left_img = cv2.imread(left_img_path)
    if left_img is not None:
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    else:
        left_img = np.zeros((1024, 1280, 3), dtype=np.uint8)
        
    # Load Segmentation Mask
    if os.path.exists(seg_img_path):
        seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)
        unique_classes = np.unique(seg_img)
        
        # Colorize the mask for visibility
        seg_vis = (seg_img * 35).astype(np.uint8) 
        seg_color = cv2.applyColorMap(seg_vis, cv2.COLORMAP_TURBO)
        seg_color[seg_img == 0] = [0, 0, 0] # Keep background purely black
        seg_color = cv2.cvtColor(seg_color, cv2.COLOR_BGR2RGB)
        
        # Add class number labels with corresponding colors
        colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO)
        for class_val in unique_classes:
            if class_val == 0:  # Skip background
                continue
            # Find locations of this class
            mask = seg_img == class_val
            if np.any(mask):
                # Find center of mass for this class
                y_coords, x_coords = np.where(mask)
                center_y = int(np.mean(y_coords))
                center_x = int(np.mean(x_coords))
                
                # Get color for this class from colormap
                color_idx = min(int(class_val * 35), 255)
                color_bgr = colormap[color_idx][0]
                color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))  # BGR to RGB
                
                # Add text label with class name
                class_name = class_names.get(class_val, f"class_{class_val}")
                # cv2.putText(seg_color, class_name, (center_x, center_y),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rgb, 2)
    else:
        seg_color = np.zeros(left_img.shape, dtype=np.uint8)
        unique_classes = []
        
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1, 0.25]})
    
    axes[0].imshow(left_img)
    axes[0].set_title(f"Split: {subset_name.split('/')[-2].capitalize()}, Sequence: {subset_name.split('/')[-1].replace('instrument_dataset_', '').zfill(2)}, Frame: {frame_name.replace('image', '').replace('.png', '')}")
    axes[0].axis('off')
    
    axes[1].imshow(seg_color)
    axes[1].set_title(f"Segmentation - Unique Classes: {unique_classes}")
    axes[1].axis('off')
    
    # Create color legend for classes
    if len(unique_classes) > 0:
        colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO)
        
        # Create legend showing class colors
        legend_height = len(unique_classes)
        legend_img = np.zeros((legend_height * 60, 250, 3), dtype=np.uint8)
        
        for idx, class_val in enumerate(unique_classes):
            if class_val == 0:
                continue
            color_idx = min(int(class_val * 35), 255)
            color_bgr = colormap[color_idx][0]
            
            # Fill rectangle with color
            y_start = idx * 60
            legend_img[y_start:y_start+60, :] = [color_bgr[2], color_bgr[1], color_bgr[0]]
            
            # Add class name text
            abbr = class_abbr.get(class_val, f"class_{class_val}")
            class_name = str(class_val) + ": " + abbr
            cv2.putText(legend_img, class_name, (10, y_start + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        axes[2].imshow(legend_img)
        axes[2].set_title("Class Colors")
        axes[2].axis('off')
    else:
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if valid_subsets:
    # Setup interactive widgets
    subset_dropdown = widgets.Dropdown(options=valid_subsets, description='Subset:')
    first_subset_frames = subset_frames[valid_subsets[0]][2]
    frame_slider = widgets.IntSlider(min=0, max=len(first_subset_frames) - 1, step=1, value=0, description='Frame:', layout=widgets.Layout(width='80%'))

    # Automatically adjust slider max when folder changes
    def update_frame_range(*args):
        subset = subset_dropdown.value
        frames = subset_frames[subset][2]
        frame_slider.max = len(frames) - 1
        frame_slider.value = 0

    subset_dropdown.observe(update_frame_range, 'value')

    ui = widgets.VBox([subset_dropdown, frame_slider])
    out = widgets.interactive_output(show_images, {'subset_name': subset_dropdown, 'frame_idx': frame_slider})
    
    display(ui, out)
# %%