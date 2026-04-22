# %% Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# %% Settings
mode = 'val'  # 'train' or 'test'
dataset_path = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}'

subsets = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
if not subsets:
    print("No valid directories found.")
else:
    # Dictionary mapping subset to its frames
    subset_frames = {}
    for sub in subsets:
        left_dir = os.path.join(dataset_path, sub, 'input', 'left_images')
        if os.path.exists(left_dir):
            frames = sorted(os.listdir(left_dir))
            if frames:
                subset_frames[sub] = frames

    valid_subsets = list(subset_frames.keys())

# %% Interactive Viewer
def show_images(subset_name, frame_idx):
    frames = subset_frames[subset_name]
    if frame_idx >= len(frames):
        frame_idx = len(frames) - 1
        
    frame_name = frames[frame_idx]
    
    left_img_path = os.path.join(dataset_path, subset_name, 'input', 'left_images', frame_name)
    seg_img_path = os.path.join(dataset_path, subset_name, 'target', 'segmentation', frame_name)
    
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
    else:
        seg_color = np.zeros(left_img.shape, dtype=np.uint8)
        unique_classes = []
        
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(left_img)
    axes[0].set_title(f"Image: {frame_name}")
    axes[0].axis('off')
    
    axes[1].imshow(seg_color)
    axes[1].set_title(f"Segmentation - Unique Pixel Vals: {unique_classes}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if valid_subsets:
    # Setup interactive widgets
    subset_dropdown = widgets.Dropdown(options=valid_subsets, description='Subset:')
    frame_slider = widgets.IntSlider(min=0, max=len(subset_frames[valid_subsets[0]]) - 1, step=1, value=0, description='Frame:', layout=widgets.Layout(width='80%'))

    # Automatically adjust slider max when folder changes
    def update_frame_range(*args):
        subset = subset_dropdown.value
        frame_slider.max = len(subset_frames[subset]) - 1
        frame_slider.value = 0

    subset_dropdown.observe(update_frame_range, 'value')

    ui = widgets.VBox([subset_dropdown, frame_slider])
    out = widgets.interactive_output(show_images, {'subset_name': subset_dropdown, 'frame_idx': frame_slider})
    
    display(ui, out)
# %%