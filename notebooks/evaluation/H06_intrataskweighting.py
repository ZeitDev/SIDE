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
    'DICE_score': {'label': 'DICE Score', 'short': 'DICE', 'arrow': '% ↑', 'suffix': '/instrument_mean', 'task': 'segmentation'},
    'AbsRel_rate': {'label': 'AbsRel Rate', 'short': 'AbsRel', 'arrow': '% ↓', 'suffix': '', 'task': 'disparity'}
}

df_bench = df_final.copy()
df_bench = df_bench[df_bench['experiment'].isin(['exp01', 'exp07', 'exp08'])]

# Map SEG/DISP to ST
df_bench['config_mapped'] = df_bench['config'].replace({'SEG': 'ST', 'DISP': 'ST'})
df_bench = df_bench[df_bench['config_mapped'].isin(['ST', 'MT', 'MT-KD'])]

for metric in metrics:
    meta = METRIC_META[metric]
    task = meta['task']
    col = f"metric.best_combined/performance/testing/{task}/{metric}{meta['suffix']}"
    fallback = f"metric.best_{task}/performance/testing/{task}/{metric}{meta['suffix']}"
    df_bench[metric] = df_bench[col].fillna(df_bench[fallback])
    
# Map experiment names for x-axis
exp_map = {'exp01': '01 (Base)', 'exp07': '07 (3:1)', 'exp08': '08 (Inverse)'}
df_bench['regime'] = df_bench['experiment'].map(exp_map)

# Common styling
colors_dict = {
    'ST': px.colors.qualitative.Plotly[0],
    'MT': px.colors.qualitative.Plotly[1], 
    'MT-KD': px.colors.qualitative.Plotly[2]
}

# %% H06F01_BoxPlot_InterTaskPerformance (Exp01 vs. 07 vs. 08)
# H06F01_BoxPlot_InterTaskPerformance (Exp01 vs. 07 vs. 08)

seg_meta = METRIC_META['DICE_score']
disp_meta = METRIC_META['AbsRel_rate']

fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"), horizontal_spacing=0.05)

regimes = ['01 (Base)', '07 (3:1)', '08 (Inverse)']
configs = ['MT-KD'] # ['ST', 'MT', 'MT-KD']

for config in configs:
    for col, metric in enumerate(['DICE_score', 'AbsRel_rate'], start=1):
        for regime in regimes:
            config_filter = 'SEG' if metric == 'DICE_score' else 'DISP'
            
            if config == 'ST':
                data = df_bench[(df_bench['regime'] == regime) & (df_bench['config'] == config_filter)][metric]
            else:
                data = df_bench[(df_bench['regime'] == regime) & (df_bench['config_mapped'] == config)][metric]
            
            # Check for inherited values
            inherited = False
            if len(data.dropna()) == 0:
                if config == 'ST':
                    data = df_bench[(df_bench['regime'] == '01 (Base)') & (df_bench['config'] == config_filter)][metric]
                else:
                    data = df_bench[(df_bench['regime'] == '01 (Base)') & (df_bench['config_mapped'] == config)][metric]
                inherited = True

            # Show legend only once per config
            showlegend = True if col == 1 and regime == regimes[0] else False
            
            fig_bar.add_trace(go.Box(
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
                offsetgroup=config,
                opacity=0.5 if inherited else 1.0
            ), row=1, col=col)

fig_bar.update_layout(
    template='plotly_white',
    height=500,
    width=850,
    boxmode='group',
    boxgroupgap=0.6,
    boxgap=0.3,
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title_text="Config")
)

fig_bar.update_xaxes(title_text=f"{seg_meta['label']} [{seg_meta['arrow']}]", row=1, col=1)
fig_bar.update_xaxes(
    title_text=f"{disp_meta['label']} [{disp_meta['arrow']}]", 
    autorange="reversed" if disp_meta['arrow'] == '% ↓' else None,
    row=1, col=2
)
fig_bar.update_yaxes(title_text="Experiment (Intra-Task Weighting)", autorange="reversed", row=1, col=1)
fig_bar.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2)

save_figure(fig_bar, height=600, name='H06F01', lrtb_margin=(40, 20, 30, 0), folder='results', skip_sync=skip_sync)

# %% H06F02_Lineplot_IntraTaskWeighting (Intra-Task Weighting Progression for MT-KD Configurations of 01, 07, 08)

target_config = 'MT-KD'
target_exps = ['exp01', 'exp07', 'exp08']

# Filter data
df_hist_filtered = df_historic[
    (df_historic['experiment'].isin(target_exps)) & 
    (df_historic['config'] == target_config)
].copy()

# Add epoch column
steps_per_epoch = 675
df_hist_filtered['epoch'] = df_hist_filtered['step'] / steps_per_epoch

metrics_dict = {
    'segmentation': {
        'val': 'performance/validation/segmentation/DICE_score/instrument_mean',
        'weight': 'optimization/training/loss/weights/intra_segmentation_distillation',
        'short': 'Val DICE',
        'arrow': '↑'
    },
    'disparity': {
        'val': 'performance/validation/disparity/AbsRel_rate',
        'weight': 'optimization/training/loss/weights/intra_disparity_distillation',
        'short': 'Val AbsRel',
        'arrow': '↓'
    }
}

fig_line = make_subplots(
    rows=2, cols=2, 
    shared_xaxes=True,
    vertical_spacing=0.08, horizontal_spacing=0.12,
    row_heights=[0.7, 0.3], 
    subplot_titles=("Segmentation", "Disparity", "", "")
)

def hex_to_rgba(hex_color, alpha=1.0):
    if hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({r},{g},{b},{alpha})'
    return hex_color

# Use distinct colors that do not overlap with ST (Blue, 0), MT (Red, 1), and MT-KD (Green, 2)
colors = {
    'exp01': {'base': px.colors.qualitative.Plotly[3], 'name': '01 (Base)'},    # Purple
    'exp07': {'base': px.colors.qualitative.Plotly[4], 'name': '07 (3:1)'},     # Orange
    'exp08': {'base': px.colors.qualitative.Plotly[5], 'name': '08 (Inverse)'}  # Cyan
}

for exp in colors:
    base_col = colors[exp]['base']
    colors[exp]['val'] = hex_to_rgba(base_col, 1.0)
    colors[exp]['fill'] = hex_to_rgba(base_col, 0.2)

dash_map = {
    'exp01': 'solid',
    'exp07': 'solid',
    'exp08': 'solid'
}

for col, task in enumerate(['segmentation', 'disparity'], start=1):
    for experiment in target_exps:
        # 1. Performance
        df_v = df_hist_filtered[(df_hist_filtered['experiment'] == experiment) & 
                                (df_hist_filtered['metric_name'] == metrics_dict[task]['val'])]
        df_v = df_v[df_v['epoch'] >= 0.5]
        grouped_v = df_v.groupby('epoch')['value'].agg(['median', 'min', 'max']).reset_index()
        
        if not grouped_v.empty:
            showlegend = True if col == 1 else False
            
            # Ribbon (Min-Max)
            fig_line.add_trace(go.Scatter(
                x=list(grouped_v['epoch']) + list(grouped_v['epoch'])[::-1],
                y=list(grouped_v['max']) + list(grouped_v['min'])[::-1],
                fill='toself',
                fillcolor=colors[experiment]['fill'],
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=experiment
            ), row=1, col=col)
            
            # Performance Line (Solid)
            fig_line.add_trace(go.Scatter(
                x=grouped_v['epoch'], y=grouped_v['median'],
                mode='lines',
                line=dict(color=colors[experiment]['val'], dash=dash_map[experiment], width=2),
                name=colors[experiment]['name'],
                legendgroup=experiment,
                showlegend=showlegend
            ), row=1, col=col)

        # 2. Weights
        df_w = df_hist_filtered[(df_hist_filtered['experiment'] == experiment) & 
                                (df_hist_filtered['metric_name'] == metrics_dict[task]['weight'])]
        df_w = df_w[df_w['epoch'] >= 0.5]
        grouped_w = df_w.groupby('epoch')['value'].agg(['median', 'min', 'max']).reset_index()
        
        if not grouped_w.empty:
            # Ribbon (Min-Max)
            fig_line.add_trace(go.Scatter(
                x=list(grouped_w['epoch']) + list(grouped_w['epoch'])[::-1],
                y=list(grouped_w['max']) + list(grouped_w['min'])[::-1],
                fill='toself',
                fillcolor=colors[experiment]['fill'],
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=experiment
            ), row=2, col=col)
            
            # Weight Line
            fig_line.add_trace(go.Scatter(
                x=grouped_w['epoch'], y=grouped_w['median'],
                mode='lines',
                line=dict(color=colors[experiment]['val'], dash=dash_map[experiment], width=2),
                name=colors[experiment]['name'],
                legendgroup=experiment,
                showlegend=False
            ), row=2, col=col)

fig_line.update_layout(
    template='plotly_white',
    height=800,
    width=1100,
    legend=dict(
        orientation="h", 
        yanchor="top", 
        y=-0.11, 
        xanchor="center", 
        x=0.5, 
        title_text="Experiment"
    )
)

# Axis Configuration
fig_line.update_yaxes(title_text="Validation DICE [% ↑]", row=1, col=1)
fig_line.update_yaxes(title_text="Validation AbsRel [% ↓]", range=[40, 6], row=1, col=2)
fig_line.update_yaxes(title_text="Distillation Weight", title_standoff=5, row=2, col=1)
fig_line.update_yaxes(title_text="Distillation Weight", title_standoff=5, row=2, col=2)

fig_line.update_xaxes(tickvals=[10, 20, 30, 40, 50], title_text="Epoch", row=2, col=1)
fig_line.update_xaxes(tickvals=[10, 20, 30, 40, 50], title_text="Epoch", row=2, col=2)

save_figure(fig_line, height=600, name='H06F02', lrtb_margin=(40, 10, 60, 80), standoff=None, folder='results', skip_sync=skip_sync)

# %%
