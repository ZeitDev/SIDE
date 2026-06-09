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

metrics = ['DICE_score', 'AbsRel_rate']

METRIC_META = {
    'DICE_score': {'label': 'DICE Score', 'short': 'DICE', 'arrow': '↑', 'suffix': '/instrument_mean', 'task': 'segmentation'},
    'AbsRel_rate': {'label': 'AbsRel Rate', 'short': 'AbsRel', 'arrow': '↓', 'suffix': '', 'task': 'disparity'}
}

df_bench = df_final.copy()
df_bench = df_bench[df_bench['experiment'].isin(['exp01', 'exp06'])]
df_bench = df_bench[df_bench['config'].isin(['MT', 'MT-KD'])]

for metric in metrics:
    meta = METRIC_META[metric]
    task = meta['task']
    col = f"metric.best_combined/performance/testing/{task}/{metric}{meta['suffix']}"
    fallback = f"metric.best_{task}/performance/testing/{task}/{metric}{meta['suffix']}"
    df_bench[metric] = df_bench[col].fillna(df_bench[fallback])
    
# Map experiment names for x-axis
exp_map = {'exp01': '01 (DTP)', 'exp06': '06 (1:1)'}
df_bench['regime'] = df_bench['experiment'].map(exp_map)

# Common styling
colors_dict = {'MT': px.colors.qualitative.Plotly[1], 'MT-KD': px.colors.qualitative.Plotly[2]}

# %% H03F01_Barplot_InterTaskPerformance (Inter-Task Performance Comparison for MT/MT-KD Configurations of Exp01 vs. 06)
# H03F01_Barplot_InterTaskPerformance (Inter-Task Performance Comparison for MT/MT-KD Configurations of Exp01 vs. 06)

seg_meta = METRIC_META['DICE_score']
disp_meta = METRIC_META['AbsRel_rate']

fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"), horizontal_spacing=0.05)

regimes = ['01 (DTP)', '06 (1:1)']
configs = ['MT', 'MT-KD']

for config in configs:
    seg_means = []
    seg_stds = []
    disp_means = []
    disp_stds = []
    for regime in regimes:
        data = df_bench[(df_bench['regime'] == regime) & (df_bench['config'] == config)]
        seg_means.append(data['DICE_score'].mean())
        seg_stds.append(data['DICE_score'].std())
        disp_means.append(data['AbsRel_rate'].mean())
        disp_stds.append(data['AbsRel_rate'].std())

    # Segmentation trace (Panel A)
    fig_bar.add_trace(go.Bar(
        name=config,
        y=regimes,
        x=seg_means,
        error_x=dict(type='data', array=seg_stds),
        orientation='h',
        marker_color=colors_dict[config],
        legendgroup=config,
        showlegend=True
    ), row=1, col=1)

    # Disparity trace (Panel B)
    fig_bar.add_trace(go.Bar(
        name=config,
        y=regimes,
        x=disp_means,
        error_x=dict(type='data', array=disp_stds),
        orientation='h',
        marker_color=colors_dict[config],
        legendgroup=config,
        showlegend=False
    ), row=1, col=2)

fig_bar.update_layout(
    template='plotly_white',
    height=400,
    width=850,
    barmode='group',
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title_text="Config")
)

fig_bar.update_xaxes(title_text=f"{seg_meta['label']} ({seg_meta['arrow']})", rangemode='tozero', row=1, col=1)
fig_bar.update_xaxes(
    title_text=f"{disp_meta['label']} ({disp_meta['arrow']})", 
    rangemode='tozero', 
    autorange="reversed" if disp_meta['arrow'] == '↓' else None,
    row=1, col=2
)
fig_bar.update_yaxes(title_text="Experiment (Inter-Task Weighting Method)", autorange="reversed", row=1, col=1)
fig_bar.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2)

save_figure(fig_bar, name='H03F01_Barplot_InterTaskPerformance', lrtb_margin=(100, 20, 30, 0), folder='results', skip_sync=skip_sync)


# %% H03F02_Lineplot_InterTaskWeighting (Inter-Task Weighting Evolution over Epochs for MT/MT-KD Configurations of Exp01 vs. 06)
# H03F02_Lineplot_InterTaskWeighting (Inter-Task Weighting Evolution over Epochs for MT/MT-KD Configurations of Exp01 vs. 06)

target_configs = ['MT', 'MT-KD']
target_exps = ['exp01', 'exp06']

# Filter data
df_hist_filtered = df_historic[
    (df_historic['experiment'].isin(target_exps)) & 
    (df_historic['config'].isin(target_configs))
].copy()

# Add epoch column
steps_per_epoch = 675
df_hist_filtered['epoch'] = df_hist_filtered['step'] / steps_per_epoch

metrics_dict = {
    'segmentation': {
        'val': 'performance/validation/segmentation/DICE_score/instrument_mean',
        'weight': 'optimization/training/loss/weights/inter_segmentation',
        'short': 'Val DICE',
        'arrow': '↑'
    },
    'disparity': {
        'val': 'performance/validation/disparity/AbsRel_rate',
        'weight': 'optimization/training/loss/weights/inter_disparity',
        'short': 'Val AbsRel',
        'arrow': '↓'
    }
}

fig_line = go.Figure()

def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color

colors = {
    'MT': {'base': px.colors.qualitative.Plotly[1]},
    'MT-KD': {'base': px.colors.qualitative.Plotly[2]}
}

for cfg in colors:
    base_col = colors[cfg]['base']
    colors[cfg]['val'] = hex_to_rgba(base_col, 1.0)
    colors[cfg]['fill'] = hex_to_rgba(base_col, 0.2)

task_dash = {'segmentation': 'solid', 'disparity': 'dot'}

# Plot Weights
for config in target_configs:
    for task in ['segmentation', 'disparity']:
        # Fetch dynamic weights from Exp01
        df_w = df_hist_filtered[(df_hist_filtered['experiment'] == 'exp01') & 
                                (df_hist_filtered['config'] == config) & 
                                (df_hist_filtered['metric_name'] == metrics_dict[task]['weight'])]
        df_w = df_w[df_w['epoch'] >= 0.5]
        
        grouped_w = df_w.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
        
        if not grouped_w.empty:
            # Ribbon
            fig_line.add_trace(go.Scatter(
                x=list(grouped_w['epoch']) + list(grouped_w['epoch'])[::-1],
                y=list(grouped_w['mean'] + grouped_w['std']) + list(grouped_w['mean'] - grouped_w['std'])[::-1],
                fill='toself',
                fillcolor=colors[config]['fill'],
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Line
            fig_line.add_trace(go.Scatter(
                x=grouped_w['epoch'], y=grouped_w['mean'],
                mode='lines',
                line=dict(color=colors[config]['val'], dash=task_dash[task], width=2),
                name=f"01 (DTP) - {task.capitalize()} - {config}",
                showlegend=True
            ))

# Add Exp06 flat weight baseline directly from data to be perfectly accurate
for config in target_configs:
    for task in ['segmentation', 'disparity']:
        df_w_exp06 = df_hist_filtered[(df_hist_filtered['experiment'] == 'exp06') & 
                                      (df_hist_filtered['config'] == config) & 
                                      (df_hist_filtered['metric_name'] == metrics_dict[task]['weight'])]
        df_w_exp06 = df_w_exp06[df_w_exp06['epoch'] >= 1]
        grouped_w_exp06 = df_w_exp06.groupby('epoch')['value'].mean().reset_index()
        
        if not grouped_w_exp06.empty:
            # We don't need multiple lines if they are identical, just show one standard for 1:1
            if task == 'segmentation' and config == 'MT':
                fig_line.add_trace(go.Scatter(
                    x=grouped_w_exp06['epoch'], y=grouped_w_exp06['value'],
                    mode='lines',
                    line=dict(color='gray', dash='dash', width=2),
                    name="06 (1:1) - Both Tasks",
                    showlegend=True
                ))

fig_line.update_layout(
    template='plotly_white',
    height=450,
    width=600,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title_text=""),
    xaxis_title="Epoch",
    yaxis_title="Inter-Task Weight"
)

fig_line.update_xaxes(tickvals=[10, 20, 30, 40, 50])

save_figure(fig_line, name='H03F02_Lineplot_InterTaskWeighting', lrtb_margin=(40, 20, 20, 60), folder='results', skip_sync=skip_sync)



# %%
