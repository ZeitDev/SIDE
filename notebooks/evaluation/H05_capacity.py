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

from notebooks.figures.helpers import save_figure, apply_chart_config

with open('./notebooks/evaluation/storage/dataframes.pkl', 'rb') as f:
    data = pickle.load(f)
    
    df_final = data['final']
    df_params = data['params']
    df_historic = data['historic']

# %% Settings
# Settings
skip_sync = False

CHART_CONFIG = {
    'H05F01': {
        'x1': dict(range=[30, 60], dtick=5),
        'x2': dict(range=[40, 0], dtick=5),
    },
    'H05F02': {
        'x': dict(range=[95, 97], dtick=0.2),
    },
    'H05F03': {
        'x1': dict(range=[55, 0], dtick=10),
        'x2': dict(range=[105, 0], dtick=20),
    }
}

# %% Data preperation
# Data preperation

# Data preperation here

# %% H05F01_Barplot_CapacityStarvation (Effect of Model Size on Performance)
# H05F01_Barplot_CapacityStarvation (Effect of Model Size on Performance)

target_exps = ['exp01', 'exp04']
target_configs = ['MT', 'MT-KD', 'ST']

df_f01 = df_final[df_final['experiment'].isin(target_exps)].copy()

# Map SEG/DISP to ST
df_f01['config_mapped'] = df_f01['config'].replace({'SEG': 'ST', 'DISP': 'ST'})
df_f01['regime'] = df_f01['experiment'].map({'exp01': '01 (Tiny)', 'exp04': '04 (Nano)'})

# Metrics configuration
metrics = {
    'segmentation': {
        'name': 'DICE_score',
        'suffix': '/instrument_mean',
        'label': 'DICE Score [% ↑]',
        'autorange': None
    },
    'disparity': {
        'name': 'AbsRel_rate',
        'suffix': '',
        'label': 'AbsRel Rate [% ↓]',
        'autorange': 'reversed'
    }
}

# Process metrics with fallback logic
for task, m_info in metrics.items():
    col = f"metric.best_combined/performance/testing/{task}/{m_info['name']}{m_info['suffix']}"
    fallback = f"metric.best_{task}/performance/testing/{task}/{m_info['name']}{m_info['suffix']}"
    df_f01[task] = df_f01[col].fillna(df_f01[fallback])

# Styling
colors_dict = {
    'ST': px.colors.qualitative.Plotly[0],
    'MT': px.colors.qualitative.Plotly[1],
    'MT-KD': px.colors.qualitative.Plotly[2]
}
regimes = ['01 (Tiny)', '04 (Nano)']
display_configs = ['ST', 'MT', 'MT-KD']

fig = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"), horizontal_spacing=0.05)

for col, task in enumerate(['segmentation', 'disparity'], start=1):
    for config in display_configs:
        for regime in regimes:
            data = df_f01[(df_f01['regime'] == regime) & (df_f01['config_mapped'] == config)][task]
            
            showlegend = True if col == 1 and regime == regimes[0] else False
            
            fig.add_trace(go.Box(
                x=data,
                y=[regime] * len(data),
                orientation='h',
                name=config,
                marker_color=colors_dict[config],
                boxpoints='all',
                jitter=0.5,
                pointpos=-2.0,
                showlegend=showlegend,
                legendgroup=config,
                offsetgroup=config
            ), row=1, col=col)

fig.update_layout(
    # template='plotly_white',
    height=450,
    width=850,
    boxmode='group',
    boxgroupgap=0.6,
    boxgap=0.3,
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title_text="Config")
)

fig.update_xaxes(title_text=metrics['segmentation']['label'], row=1, col=1)
fig.update_xaxes(
    title_text=metrics['disparity']['label'], 
    autorange=metrics['disparity']['autorange'],
    row=1, col=2
)
fig.update_yaxes(title_text="Experiment (Model Size)", autorange="reversed", row=1, col=1)
fig.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2)

apply_chart_config(fig, 'H05F01', CHART_CONFIG)
save_figure(fig, height=400, name='H05F01', lrtb_margin=(100, 20, 30, 0), folder='results', skip_sync=skip_sync)

# %% H05F02_Barplot_BinaryDistillationPerformance (Effect of Binary Distillation on Performance)

target_exps = ['exp10']
display_configs = ['ST', 'MT', 'MT-KD']

df_f03 = df_final[df_final['experiment'].isin(target_exps)].copy()

# Map SEG to ST (single task tracking segmentation)
df_f03['config_mapped'] = df_f03['config'].replace({'SEG': 'ST'})

# Keep only the configs we want
df_f03 = df_f03[df_f03['config_mapped'].isin(display_configs)]

# Set up metric col
metric_desc = {
    'name': 'DICE_score',
    'suffix': '/instrument_mean',
    'label': 'DICE Score [% ↑]'
}

col = f"metric.best_combined/performance/testing/segmentation/{metric_desc['name']}{metric_desc['suffix']}"
fallback = f"metric.best_segmentation/performance/testing/segmentation/{metric_desc['name']}{metric_desc['suffix']}"
df_f03['dice'] = df_f03[col].fillna(df_f03[fallback])

colors_dict = {
    'ST': px.colors.qualitative.Plotly[0],
    'MT': px.colors.qualitative.Plotly[1],
    'MT-KD': px.colors.qualitative.Plotly[2]
}

fig = go.Figure()

for config in display_configs:
    data_vals = df_f03[df_f03['config_mapped'] == config]['dice']
    if len(data_vals) == 0: continue
    
    fig.add_trace(go.Box(
        x=data_vals,
        y=[config] * len(data_vals),
        orientation='h',
        name=config,
        marker_color=colors_dict[config],
        boxpoints='all',
        jitter=0.5,
        pointpos=-2.0,
        showlegend=False
    ))

fig.update_layout(
    # template='plotly_white',
    height=300,
    width=600
)

fig.update_xaxes(title_text=metric_desc['label'])
fig.update_yaxes(title_text="Experiment 10 Config", autorange="reversed")

apply_chart_config(fig, 'H05F02', CHART_CONFIG)
save_figure(fig, height=300, name='H05F02', standoff=15, lrtb_margin=(60, 20, 10, 0), folder='results', skip_sync=skip_sync)

# %% H05F03_Barplot_ComplexityShift (Effect of Binary Distillation on Performance)
target_exps = ['exp01', 'exp05', 'exp10']
display_configs = ['ST', 'MT', 'MT-KD']

df_f02 = df_final[df_final['experiment'].isin(target_exps)].copy()

# Map DISP to ST
df_f02['config_mapped'] = df_f02['config'].replace({'DISP': 'ST'})

# Filter out SEG since we only care about disparity metrics
df_f02 = df_f02[df_f02['config_mapped'].isin(display_configs)]

df_f02['regime'] = df_f02['experiment'].map({'exp01': '01', 'exp05': '05', 'exp10': '10'})
regimes = ['01', '05', '10']

# Metrics configuration
metrics = {
    'absrel': {
        'task': 'disparity',
        'name': 'AbsRel_rate',
        'suffix': '',
        'label': 'AbsRel Rate [% ↓]',
        'autorange': 'reversed'
    },
    'bad3': {
        'task': 'disparity',
        'name': 'Bad3_rate',
        'suffix': '',
        'label': 'Bad3 Rate [% ↓]',
        'autorange': 'reversed'
    }
}

# Process metrics
for m_key, m_info in metrics.items():
    task = m_info['task']
    col = f"metric.best_combined/performance/testing/{task}/{m_info['name']}{m_info['suffix']}"
    fallback = f"metric.best_{task}/performance/testing/{task}/{m_info['name']}{m_info['suffix']}"
    df_f02[m_key] = df_f02[col].fillna(df_f02[fallback])

# Handling inheritance from exp01
inherited_data = []
for config in display_configs:
    for regime in regimes:
        # Check if data exists
        subset = df_f02[(df_f02['regime'] == regime) & (df_f02['config_mapped'] == config)]
        if len(subset) == 0:
            # Inherit from exp01
            base_subset = df_f02[(df_f02['regime'] == '01') & (df_f02['config_mapped'] == config)].copy()
            if len(base_subset) > 0:
                base_subset['regime'] = regime
                base_subset['inherited'] = True
                inherited_data.append(base_subset)

df_f02['inherited'] = False
if inherited_data:
    df_f02 = pd.concat([df_f02] + inherited_data, ignore_index=True)

colors_dict = {
    'ST': px.colors.qualitative.Plotly[0],
    'MT': px.colors.qualitative.Plotly[1],
    'MT-KD': px.colors.qualitative.Plotly[2]
}

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05)

for col, m_key in enumerate(['absrel', 'bad3'], start=1):
    for config in display_configs:
        for regime in regimes:
            subset = df_f02[(df_f02['regime'] == regime) & (df_f02['config_mapped'] == config)]
            if len(subset) == 0: continue
            
            is_inherited = subset['inherited'].iloc[0]
            data_vals = subset[m_key]
            
            showlegend = True if col == 1 and regime == regimes[0] and not is_inherited else False
            
            opacity = 0.5 if is_inherited else 1.0
            
            fig.add_trace(go.Box(
                x=data_vals,
                y=[regime] * len(data_vals),
                orientation='h',
                name=config,
                marker_color=colors_dict[config],
                boxpoints='all',
                jitter=0.5,
                pointpos=-2.0,
                showlegend=showlegend,
                legendgroup=config,
                offsetgroup=config,
                opacity=opacity
            ), row=1, col=col)

fig.update_layout(
    # template='plotly_white',
    height=550,
    width=850,
    boxmode='group',
    boxgroupgap=0.6,
    boxgap=0.3,
    legend=dict(orientation="h", yanchor="top", y=-0.125, xanchor="center", x=0.5, title_text="Config")
)

fig.update_xaxes(title_text=metrics['absrel']['label'], autorange=metrics['absrel']['autorange'], row=1, col=1)
fig.update_xaxes(title_text=metrics['bad3']['label'], autorange=metrics['bad3']['autorange'], row=1, col=2)

fig.update_yaxes(title_text="Experiment", autorange="reversed", row=1, col=1)
fig.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2)

apply_chart_config(fig, 'H05F03', CHART_CONFIG)
save_figure(fig, height=500, name='H05F03', lrtb_margin=(40, 20, 10, 0), folder='results', skip_sync=skip_sync)



# %%