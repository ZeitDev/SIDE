
# %%
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

# Unwrapped execution with pre-aggregation to prevent OOM
num_bins = 50
confidence_bins = torch.linspace(0, 1.0, num_bins + 1)
bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2.0

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
        logits_path = seq_path / 'teacher' / 'disparity_128_256_256'
        
        if not logits_path.exists():
            print(f"Logits path not found: {logits_path}")
            continue
            
        pt_files = list(logits_path.glob('*.pt'))
        pixel_hist_accum[(mode, seq)] = torch.zeros(num_bins, dtype=torch.float64)
        
        for pt_file in tqdm(pt_files, desc=f"{mode}/{seq}"):
            logits = torch.load(pt_file, map_location='cpu', weights_only=True)
            
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
            
            # Pre-aggregate pixels using histogram
            conf_flat = conf.flatten()
            counts = torch.histogram(conf_flat.type(torch.float32), bins=confidence_bins).hist
            pixel_hist_accum[(mode, seq)] += counts

# Prepare DataFrame and Plotly Bar for aggregated pixels
pixel_agg_records = []
for (mode, seq), counts in pixel_hist_accum.items():
    for bin_ctr, count in zip(bin_centers.tolist(), counts.tolist()):
        if count > 0:
            pixel_agg_records.append({
                'Confidence': bin_ctr,
                'Count': count,
                'Sequence': seq,
                'Mode': mode
            })

df_pixels = pd.DataFrame(pixel_agg_records)
df_images = pd.DataFrame(image_records)

if not df_pixels.empty:
    fig1 = px.bar(
        df_pixels, 
        x='Confidence', 
        y='Count',
        color='Sequence', 
        facet_col='Mode',
        barmode='overlay',
        title='Pixel-level Confidence Distribution per Sequence (Aggregated histograms)'
    )
    fig1.update_layout(bargap=0)
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

