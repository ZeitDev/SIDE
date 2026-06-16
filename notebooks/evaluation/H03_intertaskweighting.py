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

# %% H03F01_Boxplot_InterTaskPerformance (Inter-Task Performance Comparison for MT/MT-KD Configurations of Exp01 vs. 06)
# H03F01_Boxplot_InterTaskPerformance (Inter-Task Performance Comparison for MT/MT-KD Configurations of Exp01 vs. 06)

seg_meta = METRIC_META['DICE_score']
disp_meta = METRIC_META['AbsRel_rate']

fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"), horizontal_spacing=0.05)

regimes = ['01 (DTP)', '06 (1:1)']
configs = ['MT', 'MT-KD']

for config in configs:
    for col, metric in enumerate(['DICE_score', 'AbsRel_rate'], start=1):
        for regime in regimes:
            data = df_bench[(df_bench['regime'] == regime) & (df_bench['config'] == config)][metric]
            
            # Show legend only once per config (first column, first regime)
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
                offsetgroup=config
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
fig_bar.update_yaxes(title_text="Experiment (Inter-Task Weighting Method)", autorange="reversed", row=1, col=1)
fig_bar.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2)

save_figure(fig_bar, height=450, name='H03F01', lrtb_margin=(100, 20, 30, 0), folder='results', skip_sync=skip_sync)

# %% H03F02_Boxplot_BestEpochs (Best Epochs Comparison for ST/MT/MT-KD Configurations)
# H03F02_Boxplot_BestEpochs (Best Epochs Comparison for ST/MT/MT-KD Configurations)

target_exps = ['exp01', 'exp06']
target_configs = ['MT', 'MT-KD', 'SEG', 'DISP']

df_epochs = df_final[
    (df_final['experiment'].isin(target_exps)) & 
    (df_final['config'].isin(target_configs))
].copy()

# Map SEG/DISP to ST
df_epochs['config_mapped'] = df_epochs['config'].replace({'SEG': 'ST', 'DISP': 'ST'})
df_epochs['regime'] = df_epochs['experiment'].map({'exp01': '01 (DTP)', 'exp06': '06 (1:1)'})

fig_epochs = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"), horizontal_spacing=0.05)

best_epoch_col = 'metric.best/combined/epoch'

colors_epochs = {
    'ST': px.colors.qualitative.Plotly[0],
    'MT': px.colors.qualitative.Plotly[1],
    'MT-KD': px.colors.qualitative.Plotly[2]
}

regimes = ['01 (DTP)', '06 (1:1)']
display_configs = ['ST', 'MT', 'MT-KD']

for col, task in enumerate(['segmentation', 'disparity'], start=1):
    for config in display_configs:
        for regime in regimes:
            # Select the correct column based on config
            if config == 'ST':
                st_metric = 'metric.best/segmentation/epoch' if task == 'segmentation' else 'metric.best/disparity/epoch'
                config_filter = 'SEG' if task == 'segmentation' else 'DISP'
                data = df_epochs[(df_epochs['regime'] == regime) & (df_epochs['config'] == config_filter)][st_metric]
            else:
                data = df_epochs[(df_epochs['regime'] == regime) & (df_epochs['config_mapped'] == config)][best_epoch_col]
            
            showlegend = True if col == 1 and regime == regimes[0] else False
            
            fig_epochs.add_trace(go.Box(
                x=data,
                y=[regime] * len(data),
                orientation='h',
                name=config,
                marker_color=colors_epochs[config],
                boxpoints='all',
                jitter=0.5,
                pointpos=-2.0,
                showlegend=showlegend,
                legendgroup=config,
                offsetgroup=config
            ), row=1, col=col)

fig_epochs.update_layout(
    template='plotly_white',
    height=500,
    width=850,
    boxmode='group',
    boxgroupgap=0.6,
    boxgap=0.3,
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title_text="Config")
)

fig_epochs.update_xaxes(title_text="Best Epoch", showgrid=True, gridcolor='rgba(0,0,0,0.1)', row=1, col=1)
fig_epochs.update_xaxes(title_text="Best Epoch", showgrid=True, gridcolor='rgba(0,0,0,0.1)', row=1, col=2)
fig_epochs.update_yaxes(title_text="Experiment (Inter-Task Weighting Method)", autorange="reversed", row=1, col=1)
fig_epochs.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2)

save_figure(fig_epochs, height=450, name='H03F02', lrtb_margin=(100, 20, 30, 0), folder='results', skip_sync=skip_sync)


# %% H03F03_Scatter_PerformanceVsEpoch (Seed Performance vs. Best Validation Epoch)
# H03F03_Scatter_PerformanceVsEpoch (Seed Performance vs. Best Validation Epoch)

target_exps = ['exp01', 'exp06']
target_configs = ['MT', 'MT-KD', 'SEG', 'DISP']

df_scatter = df_final[
    (df_final['experiment'].isin(target_exps)) & 
    (df_final['config'].isin(target_configs))
].copy()

# Add mapping for experiment labels
df_scatter['regime'] = df_scatter['experiment'].map({'exp01': '01 (DTP)', 'exp06': '06 (1:1)'})
# Map SEG/DISP to ST
df_scatter['config_mapped'] = df_scatter['config'].replace({'SEG': 'ST', 'DISP': 'ST'})

# Extract the base run name to match test and train runs
df_scatter['base_run_name'] = df_scatter['run_name'].str.replace('/test', '', regex=False).str.replace('/train', '', regex=False)

train_df = df_scatter[df_scatter['run_name'].str.endswith('/train')].copy()
train_df['epoch_seg_val'] = train_df['metric.best/combined/epoch'].fillna(train_df['metric.best/segmentation/epoch'])
train_df['epoch_disp_val'] = train_df['metric.best/combined/epoch'].fillna(train_df['metric.best/disparity/epoch'])

# Aggregate performance and epoch metrics using fallback logic
for metric in ['DICE_score', 'AbsRel_rate']:
    meta = METRIC_META[metric]
    task = meta['task']
    col = f"metric.best_combined/performance/testing/{task}/{metric}{meta['suffix']}"
    fallback = f"metric.best_{task}/performance/testing/{task}/{metric}{meta['suffix']}"
    df_scatter[metric] = df_scatter[col].fillna(df_scatter[fallback])

# Map the epochs from the train runs to all rows
df_scatter['best_epoch_seg'] = df_scatter['base_run_name'].map(train_df.set_index('base_run_name')['epoch_seg_val'])
df_scatter['best_epoch_disp'] = df_scatter['base_run_name'].map(train_df.set_index('base_run_name')['epoch_disp_val'])

# Keep only the test runs, which now have both performance metrics and mapped epochs
df_scatter = df_scatter[df_scatter['run_name'].str.endswith('/test')]

fig_scatter = make_subplots(
    rows=2, cols=2, 
    subplot_titles=("01 (DTP)", "06 (1:1)", "", ""),
    horizontal_spacing=0.02,
    vertical_spacing=0.08,
    shared_xaxes=True,
    shared_yaxes=True
)

for row, task in enumerate(['segmentation', 'disparity'], start=1):
    for col, regime in enumerate(['01 (DTP)', '06 (1:1)'], start=1):
        perf_metric = 'DICE_score' if task == 'segmentation' else 'AbsRel_rate'
        epoch_metric = 'best_epoch_seg' if task == 'segmentation' else 'best_epoch_disp'
        
        for config in ['MT', 'MT-KD']:
            mask = (df_scatter['regime'] == regime) & (df_scatter['config_mapped'] == config)
            
            x_val = df_scatter[mask][epoch_metric]
            y_val = df_scatter[mask][perf_metric]
            
            # Show legend only once per config
            showlegend = True if (row == 1 and col == 1) else False
            
            # Use marker edge color to make sure shapes are visible even if overlapping
            fig_scatter.add_trace(go.Scatter(
                x=x_val,
                y=y_val,
                mode='markers',
                name=config,
                marker=dict(
                    color=colors_epochs[config],
                    size=10,
                    opacity=0.7,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                legendgroup=config,
                showlegend=showlegend
            ), row=row, col=col)

fig_scatter.update_layout(
    template='plotly_white',
    height=700,
    width=950,
    legend=dict(orientation="h", yanchor="top", y=-0.07, xanchor="center", x=0.5, title_text="Config")
)

# Axis Configuration
# Row 1 (Top): Hide x-axis labels/titles
fig_scatter.update_xaxes(showticklabels=False, title_text="", showgrid=True, gridcolor='rgba(0,0,0,0.1)', row=1)
# Row 2 (Bottom): Show x-axis labels/titles
fig_scatter.update_xaxes(title_text="Best Epoch", showgrid=True, gridcolor='rgba(0,0,0,0.1)', row=2)

# Col 1 (Left): Show y-axis labels/titles
fig_scatter.update_yaxes(title_text="Test DICE Score [% ↑]", row=1, col=1)
fig_scatter.update_yaxes(title_text="Test AbsRel Rate [% ↓]", autorange="reversed", row=2, col=1)

# Col 2 (Right): Hide y-axis labels/titles
fig_scatter.update_yaxes(showticklabels=False, title_text="", row=1, col=2)
fig_scatter.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=2, col=2)

# ! omitted
# save_figure(fig_scatter, height=700, name='H03F03', lrtb_margin=(40, 20, 30, 80), folder='results', skip_sync=skip_sync)

# %% H03F03_Lineplot_InterTaskWeighting (Inter-Task Weighting Evolution over Epochs for MT/MT-KD Configurations of Exp01 vs. 06)
# H03F03_Lineplot_InterTaskWeighting (Inter-Task Weighting Evolution over Epochs for MT/MT-KD Configurations of Exp01 vs. 06)

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

fig_line = make_subplots(
    rows=2, cols=2, 
    shared_xaxes=True, shared_yaxes=True,
    vertical_spacing=0.05, horizontal_spacing=0.03,
    row_heights=[0.7, 0.3], 
    subplot_titles=("01 (DTP)", "06 (1:1)", "", "")
)

def hex_to_rgba(hex_color, alpha=1.0):
    if hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({r},{g},{b},{alpha})'
    elif hex_color == 'red':
        return f'rgba(255,0,0,{alpha})'
    elif hex_color == 'green':
        return f'rgba(0,128,0,{alpha})'
    return hex_color

colors = {
    'MT': {'base': 'red'},
    'MT-KD': {'base': 'green'}
}

for cfg in colors:
    base_col = colors[cfg]['base']
    colors[cfg]['val'] = hex_to_rgba(base_col, 1.0)
    colors[cfg]['fill'] = hex_to_rgba(base_col, 0.2)

# Plot Performance (Top Subplot) and Weights (Bottom Subplot)
weight_dash_map = {
    'exp01_MT': 'solid',
    'exp01_MT-KD': 'solid',
    'exp06_MT': '4px, 8px',          # 4px dash, 8px space (Cycle=12)
    'exp06_MT-KD': '0px, 6px, 4px, 2px'  # offset to land perfectly between the MT dashes
}

for col, experiment in enumerate(target_exps, start=1):
    for config in target_configs:
        # 1. Performance (Disparity)
        task = 'disparity'
        df_v = df_hist_filtered[(df_hist_filtered['experiment'] == experiment) & 
                                (df_hist_filtered['config'] == config) & 
                                (df_hist_filtered['metric_name'] == metrics_dict[task]['val'])]
        df_v = df_v[df_v['epoch'] >= 0.5]
        grouped_v = df_v.groupby('epoch')['value'].agg(['median', 'min', 'max']).reset_index()
        
        if not grouped_v.empty:
            # Show legend only once per config (first column)
            showlegend = True if col == 1 else False
            
            # Ribbon (Min-Max)
            fig_line.add_trace(go.Scatter(
                x=list(grouped_v['epoch']) + list(grouped_v['epoch'])[::-1],
                y=list(grouped_v['max']) + list(grouped_v['min'])[::-1],
                fill='toself',
                fillcolor=colors[config]['fill'],
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=config
            ), row=1, col=col)
            
            # Performance Line (Solid)
            fig_line.add_trace(go.Scatter(
                x=grouped_v['epoch'], y=grouped_v['median'],
                mode='lines',
                line=dict(color=colors[config]['val'], dash='solid', width=2),
                name=config,
                legendgroup=config,
                showlegend=showlegend
            ), row=1, col=col)

        # 2. Weights (Disparity only)
        df_w = df_hist_filtered[(df_hist_filtered['experiment'] == experiment) & 
                                (df_hist_filtered['config'] == config) & 
                                (df_hist_filtered['metric_name'] == metrics_dict[task]['weight'])]
        df_w = df_w[df_w['epoch'] >= 0.5]
        grouped_w = df_w.groupby('epoch')['value'].agg(['median', 'min', 'max']).reset_index()
        
        if not grouped_w.empty:
            # Ribbon (Min-Max)
            fig_line.add_trace(go.Scatter(
                x=list(grouped_w['epoch']) + list(grouped_w['epoch'])[::-1],
                y=list(grouped_w['max']) + list(grouped_w['min'])[::-1],
                fill='toself',
                fillcolor=colors[config]['fill'],
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=config
            ), row=2, col=col)
            
            # Weight Line (Custom dash map to restore alternating pattern for exp06)
            fig_line.add_trace(go.Scatter(
                x=grouped_w['epoch'], y=grouped_w['median'],
                mode='lines',
                line=dict(color=colors[config]['val'], dash=weight_dash_map[f"{experiment}_{config}"], width=2),
                name=config,
                legendgroup=config,
                showlegend=False
            ), row=2, col=col)

fig_line.update_layout(
    template='plotly_white',
    height=800,
    width=950,
    legend=dict(
        orientation="h", 
        yanchor="top", 
        y=-0.11, 
        xanchor="center", 
        x=0.5, 
        title_text="Config"
    )
)

# Axis Configuration
fig_line.update_yaxes(title_text="Validation AbsRel Rate [Log % ↓]", range=[40, 6], row=1, col=1)#, type="log", range=[np.log10(30), np.log10(6)], title_standoff=10, row=1, col=1)
#fig_line.update_yaxes(type="log", range=[np.log10(30), np.log10(6)], showticklabels=False, row=1, col=2)
fig_line.update_yaxes(range=[40, 6], showticklabels=False, row=1, col=2)
fig_line.update_yaxes(title_text="Disparity Weight", range=[0, 1.1], title_standoff=5, row=2, col=1)
fig_line.update_yaxes(range=[0, 1.1], showticklabels=False, row=2, col=2)

fig_line.update_xaxes(tickvals=[10, 20, 30, 40, 50], title_text="Epoch", row=2, col=1)
fig_line.update_xaxes(tickvals=[10, 20, 30, 40, 50], title_text="Epoch", row=2, col=2)

save_figure(fig_line, height=600, name='H03F03', lrtb_margin=(40, 40, 60, 80), standoff=None, folder='results', skip_sync=skip_sync)

# %%

