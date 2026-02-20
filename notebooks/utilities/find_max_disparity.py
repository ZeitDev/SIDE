# %%
# import
import os, sys
sys.path.append('/data/Zeitler/code/SIDE')

from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
from tqdm.notebook import tqdm

from collections import Counter

from utils.setup import setup_environment
setup_environment()

# %%
global_max = 0.0
value_counts = Counter()

for mode in ['train', 'test']:
    sequences = sorted([p for p in Path(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}').iterdir() if p.is_dir()])
    
    for sequence_name in tqdm(sequences, desc=f"Processing {mode}"):
        frames = sorted((sequence_name / 'ground_truth' / 'disparity').iterdir())
        
        for frame in tqdm(frames, desc="Frames", leave=False):
            disparity_map = np.array(Image.open(frame)).astype(float)
            
            valid_mask = disparity_map > 0
            disparity_map[valid_mask] = disparity_map[valid_mask] / 128
            disparity_map[~valid_mask] = 0

            current_max = disparity_map.max()
            if current_max > global_max:
                global_max = current_max
            
            unique_vals, counts = np.unique(disparity_map, return_counts=True)
            for val, count in zip(unique_vals, counts):
                value_counts[val] += count


# %%
print(f"Max Disparity: {global_max}")
print(f"Count: {sum(value_counts.values())}")

s = pd.Series(value_counts).sort_index()
print(s)

df = s.reset_index()
df.columns = ['Disparity', 'Count']

df['Binned_Disparity'] = (df['Disparity'] // 2) * 2  # Group every 2 values, adjust divisor as needed
df_binned = df.groupby('Binned_Disparity')['Count'].sum().reset_index()

fig = px.bar(df_binned, x='Binned_Disparity', y='Count', title="Disparity Value Distribution (Binned)", log_y=True)
fig.update_traces(marker_line_width=0) # Remove outlines that might clutter dense plots
fig.update_layout(xaxis_title="Disparity", yaxis_title="Count", bargap=0.0) # Reduce gap to make them solid
fig.show()



# %%
