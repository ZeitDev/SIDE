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
skip_sync = True

# %% Data preperation
# Data preperation

# Data preperation here

# %% H02F01_Lineplot_ConfigValidationPerformance (Train vs. Val Metrics over Epochs for MT vs. MT-KD)
# H02F01_Lineplot_ConfigValidationPerformance (Train vs. Val Metrics over Epochs for MT vs. MT-KD)

target_exp = 'exp01'
#target_configs = ['SEG', 'DISP', 'MT', 'MT-KD']
target_configs = ['MT', 'MT-KD']

# Filter data
df_hist_filtered = df_historic[
    (df_historic['experiment'] == target_exp) & 
    (df_historic['config'].isin(target_configs))
].copy()

# Add epoch column
steps_per_epoch = 675
df_hist_filtered['epoch'] = df_hist_filtered['step'] / steps_per_epoch

metrics_dict = {
    'segmentation': {
        'val': 'performance/validation/segmentation/DICE_score/instrument_mean',
        'short': 'DICE Score',
        'arrow': '% ↑',
        'st_config': 'SEG'
    },
    'disparity': {
        'val': 'performance/validation/disparity/AbsRel_rate',
        'short': 'AbsRel Rate',
        'arrow': '% ↓',
        'st_config': 'DISP'
    }
}

fig = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"), horizontal_spacing=0.05)

colors = {
    'ST': {'base': px.colors.qualitative.Plotly[0]},
    'MT': {'base': px.colors.qualitative.Plotly[1]},
    'MT-KD': {'base': px.colors.qualitative.Plotly[2]}
}

def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:  # Handle standard hex
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color

for cfg in colors:
    base_col = colors[cfg]['base']
    colors[cfg]['val'] = hex_to_rgba(base_col, 1.0)
    colors[cfg]['fill'] = hex_to_rgba(base_col, 0.2)
    
def add_metric_traces(fig, df, metric_meta, col):
    metric_val = metric_meta['val']
    st_cfg = metric_meta['st_config']
    
    current_configs = [st_cfg, 'MT', 'MT-KD']
    
    for config in current_configs:
        # Determine display name and color key
        display_name = 'ST' if config == st_cfg else config
        color_key = 'ST' if config == st_cfg else config
        
        # Val
        df_val = df[(df['config'] == config) & (df['metric_name'] == metric_val)]
        
        # Filter out early epochs (start at epoch 2)
        df_val = df_val[df_val['epoch'] >= 2]
        
        grouped_val = df_val.groupby('epoch')['value'].agg(['median', 'min', 'max']).reset_index()
        
        if grouped_val.empty:
            continue
            
        showlegend = True if col == 1 else False
        
        # Val Ribbon
        fig.add_trace(go.Scatter(
            y=list(grouped_val['epoch']) + list(grouped_val['epoch'])[::-1],
            x=list(grouped_val['max']) + list(grouped_val['min'])[::-1],
            fill='toself',
            fillcolor=colors[color_key]['fill'],
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=col)
        
        # Val Line
        fig.add_trace(go.Scatter(
            y=grouped_val['epoch'],
            x=grouped_val['median'],
            mode='lines',
            line=dict(color=colors[color_key]['val'], width=2),
            name=display_name,
            showlegend=showlegend,
            legendgroup=display_name
        ), row=1, col=col)

add_metric_traces(fig, df_hist_filtered, metrics_dict['segmentation'], col=1)
add_metric_traces(fig, df_hist_filtered, metrics_dict['disparity'], col=2)

fig.update_layout(
    template='plotly_white',
    height=450,
    width=850,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    legend_title_text="Experiment 01 Config"
)

fig.update_yaxes(title_text="Validation Epoch", tickvals=[2, 10, 20, 30, 40, 50], row=1, col=1)
fig.update_yaxes(tickvals=[2, 10, 20, 30, 40, 50], showticklabels=False, row=1, col=2)
fig.update_xaxes(title_text=f"{metrics_dict['segmentation']['short']} [{metrics_dict['segmentation']['arrow']}]", row=1, col=1)
fig.update_xaxes(
    title_text=f"{metrics_dict['disparity']['short']} [{metrics_dict['disparity']['arrow']}]", 
    autorange="reversed",
    row=1, col=2
)

save_figure(fig, name='H02F01_Lineplot_ConfigValidationPerformance', lrtb_margin=(40, 20, 20, 60), folder='results', skip_sync=skip_sync)

# %% H02F02_ConnectedScatter_ConfigValidationInterceptPerformance 
# H02F02_ConnectedScatter_ConfigValidationInterceptPerformance

target_exp = 'exp01'
target_configs = ['MT', 'MT-KD']

df_final_filtered = df_final[
    (df_final['experiment'] == target_exp) & 
    (df_final['config'].isin(target_configs))
].copy()

metrics_dict_f02 = {
    'segmentation': {
        'best': 'metric.best/combined/performance_validation_segmentation_DICE_score_instrument_mean',
        'intercept': 'metric.best/combined/performance_validation_misc_interceptDICE_score',
        'y_label': "DICE Score [% ↑]"
    },
    'disparity': {
        'best': 'metric.best/combined/performance_validation_disparity_AbsRel_rate',
        'intercept': 'metric.best/combined/performance_validation_misc_interceptAbsRel_rate',
        'y_label': "AbsRel Rate [% ↓]"
    }
}

fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"), horizontal_spacing=0.1)

stages = ['Projection', 'Prediction']

for col, task in enumerate(['segmentation', 'disparity'], start=1):
    best_col = metrics_dict_f02[task]['best']
    intercept_col = metrics_dict_f02[task]['intercept']
    
    for config in target_configs:
        config_data = df_final_filtered[df_final_filtered['config'] == config]
        
        intercept_median = config_data[intercept_col].median()
        intercept_min = config_data[intercept_col].min()
        intercept_max = config_data[intercept_col].max()
        
        best_median = config_data[best_col].median()
        best_min = config_data[best_col].min()
        best_max = config_data[best_col].max()
        
        showlegend = True if col == 1 else False
        
        fig2.add_trace(go.Scatter(
            y=stages,
            x=[intercept_median, best_median],
            mode='lines+markers',
            line=dict(color=colors[config]['val'], width=2),
            marker=dict(color=colors[config]['val'], size=8),
            error_x=dict(
                type='data',
                symmetric=False,
                array=[intercept_max - intercept_median, best_max - best_median],
                arrayminus=[intercept_median - intercept_min, best_median - best_min],
                visible=True,
                color=colors[config]['val'],
                thickness=1.5,
                width=5
            ),
            name=config,
            showlegend=showlegend,
            legendgroup=config
        ), row=1, col=col)
        
    fig2.update_xaxes(title_text=metrics_dict_f02[task]['y_label'], row=1, col=col)
    if col == 1:
        fig2.update_yaxes(title_text="Head", range=[-0.25, 1.25], row=1, col=col)
    else:
        fig2.update_yaxes(range=[-0.25, 1.25], showticklabels=False, row=1, col=col)
    
fig2.update_xaxes(autorange="reversed", row=1, col=2)

fig2.update_layout(
    template='plotly_white',
    height=450,
    width=850,
    legend=dict(orientation="h", yanchor="top", y=-0.275, xanchor="center", x=0.5),
    legend_title_text="Experiment 01 Config"
)

fig2.add_annotation(
    x=0.5, y=-0.275,
    showarrow=False,
    text="Best Validation Epoch Performance",
    xref="paper", yref="paper",
    font=dict(size=14)
)

save_figure(fig2, name='H02F02_ConnectedScatter_ConfigValidationInterceptPerformance', lrtb_margin=(40, 20, 20, 60), folder='results', skip_sync=skip_sync)

# %%