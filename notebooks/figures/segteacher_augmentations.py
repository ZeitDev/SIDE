# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))
from pathlib import Path
import numpy as np
from PIL import Image
import albumentations as A
import plotly.subplots as sp
import plotly.graph_objects as go
from notebooks.figures.helpers import save_figure

os.chdir('/data/Zeitler/code/SIDE')

# %%
# Load first image from instrument_dataset_1
DATA_PATH = Path('/data/Zeitler/SIDED/EndoVis17/processed')
seq1_path = DATA_PATH / 'train' / 'instrument_dataset_1'
image_dir = seq1_path / 'input' / 'left_images'

# Get first image
img_path = sorted(list(image_dir.glob('*.png')))[0]
img = np.array(Image.open(img_path))

# %%
# Define augmentations based on configs/misc/segmentation_teacher_binary.yaml
# We set p=1.0 for each to guarantee they are applied for the visualization

augs = [
    ("Original", A.NoOp(p=1.0)),
    ("CenterCrop<br>(1024x1024)", A.CenterCrop(height=1024, width=1024, p=1.0)),
    ("Resize<br>(1024x1024)", A.Resize(height=1024, width=1024, p=1.0)),
    ("Affine", A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.0625), rotate=(-45, 45), p=1.0)),
    ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
    ("VerticalFlip", A.VerticalFlip(p=1.0)),
    ("GridDistortion", A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)),
    ("RandomBrightness<br>Contrast", A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)),
    ("GaussianBlur", A.GaussianBlur(blur_limit=(3, 5), p=1.0)),
    ("MotionBlur", A.MotionBlur(blur_limit=(3, 5), p=1.0)),
    ("GaussNoise", A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)),  # Approx for std_range: [0.01, 0.03]
    ("CoarseDropout<br>(Occlusion - Tissue/Blood)", A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 20), hole_width_range=(1, 20), fill=0, p=1.0)),
    ("CoarseDropout<br>(Glare on Metal)", A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(1, 15), hole_width_range=(1, 15), fill=255, p=1.0)),
]

# %%
# Apply augmentations and track the images

# To mimic the pipeline appropriately and show the effects of subsequent transforms properly, 
# we create a base cropped/resized image (1024x1024) to apply the visual augmentations on.
base_transform = A.Compose([
    A.CenterCrop(height=1024, width=1024, p=1.0),
    A.Resize(height=1024, width=1024, p=1.0)
])
base_img = base_transform(image=img)['image']

aug_images = []

for name, transform in augs:
    # Original, pure crop and resize should be applied to the RAW image
    if "Original" in name or "CenterCrop" in name or "Resize" in name:
        res = transform(image=img)['image']
    else:
        # Subsequent transforms (flips, colour, noise, dropout) applied on base_img
        res = transform(image=base_img)['image']
        
    aug_images.append((name, res))

# %%
# Visualizing 
cols = 4
rows = (len(aug_images) + cols - 1) // cols

fig = sp.make_subplots(
    rows=rows, cols=cols,
    subplot_titles=[name for name, _ in aug_images],
    horizontal_spacing=0.02,
    vertical_spacing=0.08
)

for i, (name, aug_img) in enumerate(aug_images):
    r = (i // cols) + 1
    c = (i % cols) + 1
    fig.add_trace(go.Image(z=aug_img), row=r, col=c)

# Hide axes
fig.update_xaxes(showticklabels=False, visible=False)
fig.update_yaxes(showticklabels=False, visible=False)

fig.update_layout(
    height=300 * rows,
    width=1200,
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=10, r=10, t=60, b=10)
)

save_figure(fig, name='segteacher_augmentations')
fig.show()

# %%
