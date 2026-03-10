
# %%
import os
import torch
import torch.nn.functional as F
import plotly.express as px
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# data path /data/Zeitler/SIDED/EndoVis17/processed
# modes are 'train', 'val', 'test'
# disparity_logits are in datapath/mode/sequence/teacher/disparity_128_256_256/filename.pt

# %%
DATA_PATH = Path('/data/Zeitler/SIDED/EndoVis17/processed')
MODES = ['train', 'val', 'test']

def plot_confidence_histograms(sample_files_per_seq=10, sample_pixels_per_file=10000):
    pixel_records = []
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
            logits_path = seq_path / 'teacher' / 'disparity_128_256_256'
            
            if not logits_path.exists():
                print(f"Logits path not found: {logits_path}")
                continue
                
            pt_files = list(logits_path.glob('*.pt'))
            
            # Subsample files to avoid memory explosion
            for pt_file in tqdm(pt_files[:sample_files_per_seq], desc=f"{mode}/{seq}"):
                logits = torch.load(pt_file, map_location='cpu', weights_only=True)
                
                # Assuming shape is (1, D, H, W) or (D, H, W)
                # Softmax along disparity dimension. Often D is the first or second dim.
                if logits.dim() == 4:
                    probs = F.softmax(logits, dim=1)
                    conf = probs.max(dim=1)[0].squeeze(0) # (H, W)
                elif logits.dim() == 3:
                    probs = F.softmax(logits, dim=0)
                    conf = probs.max(dim=0)[0] # (H, W)
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
                
                # Image-level mean confidence
                img_mean_conf = conf.mean().item()
                image_records.append({
                    'Mean Confidence': img_mean_conf,
                    'Sequence': seq,
                    'Mode': mode
                })
                
                # Flatten the confidence map
                conf_flat = conf.flatten()
                
                # Randomly sample pixels
                if len(conf_flat) > sample_pixels_per_file:
                    indices = torch.randperm(len(conf_flat))[:sample_pixels_per_file]
                    conf_sampled = conf_flat[indices]
                else:
                    conf_sampled = conf_flat
                
                for c in conf_sampled.tolist():
                    pixel_records.append({
                        'Confidence': c,
                        'Sequence': seq,
                        'Mode': mode
                    })

    df_pixels = pd.DataFrame(pixel_records)
    df_images = pd.DataFrame(image_records)
    
    if not df_pixels.empty:
        fig1 = px.histogram(
            df_pixels, 
            x='Confidence', 
            color='Sequence', 
            facet_col='Mode',
            barmode='overlay',
            nbins=50,
            title='Pixel-level Confidence Distribution per Sequence'
        )
        fig1.show()
    else:
        print("No data collected for the pixel histogram.")
        
    if not df_images.empty:
        fig2 = px.histogram(
            df_images, 
            x='Mean Confidence', 
            color='Sequence', 
            facet_col='Mode',
            barmode='overlay',
            nbins=50,
            title='Image-level Mean Confidence Distribution per Sequence'
        )
        fig2.show()
    else:
        print("No data collected for the image histogram.")

# %%
plot_confidence_histograms()

