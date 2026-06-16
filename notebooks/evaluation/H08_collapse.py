# %% Import
# Import
import os, sys
from pathlib import Path
root_path = Path.cwd()
while root_path.parent != root_path and not (root_path / 'pyproject.toml').exists(): root_path = root_path.parent
os.chdir(root_path)
if str(root_path) not in sys.path: sys.path.append(str(root_path))

import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from notebooks.figures.helpers import save_figure

with open('./notebooks/evaluation/storage/dataframes.pkl', 'rb') as f:
    data = pickle.load(f)
    
    df_final = data['final']
    df_params = data['params']
    df_historic = data['historic']

# %% Settings
# Settings
skip_sync = False

# %% Data preperation
# Data preperation

df_collapse = df_final.copy()
df_collapse = df_collapse[df_collapse['experiment'].isin(['exp01', 'exp08', 'exp09'])]
df_collapse = df_collapse[df_collapse['config'] == 'MT-KD']

metrics = ['DICE_score', 'AbsRel_rate']
METRIC_META = {
    'DICE_score': {'suffix': '/instrument_mean', 'task': 'segmentation'},
    'AbsRel_rate': {'suffix': '', 'task': 'disparity'}
}

for metric in metrics:
    meta = METRIC_META[metric]
    task = meta['task']
    col = f"metric.best_combined/performance/testing/{task}/{metric}{meta['suffix']}"
    fallback = f"metric.best_{task}/performance/testing/{task}/{metric}{meta['suffix']}"
    df_collapse[metric] = df_collapse[col].fillna(df_collapse[fallback])

exp_map = {'exp01': '01 (Base)', 'exp08': '08 (Inverse)', 'exp09': '09 (Temperature = 4)'}
df_collapse['regime'] = df_collapse['experiment'].map(exp_map)

# %% H08F01_Scatter_SharedTaskCollapse
# H08F01_Scatter_SharedTaskCollapse

fig_scatter = go.Figure()

colors = {
    'exp01': px.colors.qualitative.Plotly[3],    # Purple (from H06F02)
    'exp08': px.colors.qualitative.Plotly[5],    # Cyan (from H06F02)
    'exp09': px.colors.qualitative.Plotly[6]     # Pink (new for Temp=4)
}

for exp in ['exp01', 'exp08', 'exp09']:
    data = df_collapse[df_collapse['experiment'] == exp]
    fig_scatter.add_trace(go.Scatter(
        x=data['AbsRel_rate'],
        y=data['DICE_score'],
        mode='markers',
        marker=dict(color=colors[exp], size=10, opacity=0.8),
        name=exp_map[exp]
    ))

fig_scatter.update_layout(
    template='plotly_white',
    height=500,
    width=600,
    xaxis_title='AbsRel Rate [% ↓]',
    yaxis_title='DICE Score [% ↑]',
    legend=dict(
        orientation="h", 
        yanchor="top", 
        y=-0.2, 
        xanchor="center", 
        x=0.5, 
        title_text="MT-KD Experiment"
    )
)

# Reversing x-axis so that smaller AbsRel is on the right, making 'up/right' the better direction
fig_scatter.update_xaxes(autorange="reversed")

save_figure(fig_scatter, height=500, name='H08F01', lrtb_margin=(40, 20, 30, 0), folder='results', skip_sync=skip_sync)

# %%