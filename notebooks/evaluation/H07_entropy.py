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
    
with open('./notebooks/evaluation/storage/entropy_metrics.pkl', 'rb') as f:
    df_entropy = pickle.load(f)

# %% Settings
# Settings
skip_sync = False

# %% Data preperation
# Data preperation

# Unified experiment labels and ordering
EXP_MAP = {
    'exp01': '01 (Base)', 
    'exp02': '02<br>(No Disparity Gate)',
    'exp03': '03 (No Spatial<br>Distillation Loss<br>Scaling)',
    'exp09': '09 (Temperature = 4)',
    #'exp04': 'debug'
}

metrics = ['DICE_score', 'AbsRel_rate']

METRIC_META = {
    'DICE_score': {'label': 'DICE Score', 'short': 'DICE', 'arrow': '% ↑', 'suffix': '/instrument_mean', 'task': 'segmentation'},
    'AbsRel_rate': {'label': 'AbsRel Rate', 'short': 'AbsRel', 'arrow': '% ↓', 'suffix': '', 'task': 'disparity'}
}

df_bench = df_final.copy()
df_bench = df_bench[df_bench['experiment'].isin(EXP_MAP.keys())]
df_bench = df_bench[df_bench['config'] == 'MT-KD']

for metric in metrics:
    meta = METRIC_META[metric]
    task = meta['task']
    col = f"metric.best_combined/performance/testing/{task}/{metric}{meta['suffix']}"
    fallback = f"metric.best_{task}/performance/testing/{task}/{metric}{meta['suffix']}"
    df_bench[metric] = df_bench[col].fillna(df_bench[fallback])
    
# Map experiment names for y-axis
df_bench['regime'] = df_bench['experiment'].map(EXP_MAP)

# Common styling
colors_dict = {'MT-KD': px.colors.qualitative.Plotly[2]}

# %% H07F01_Boxplot_Entropy (Exp01 vs. 09)
# H07F01_Boxplot_Entropy (Exp01 vs. 09)

seg_meta = METRIC_META['DICE_score']
disp_meta = METRIC_META['AbsRel_rate']

fig_bar = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

regimes = list(EXP_MAP.values())
configs = ['MT-KD']

for config in configs:
    for row, metric in enumerate(['DICE_score', 'AbsRel_rate'], start=1):
        for regime in regimes:
            data = df_bench[(df_bench['regime'] == regime) & (df_bench['config'] == config)][metric]
            
            # Show legend only once per config
            showlegend = True if row == 1 and regime == regimes[0] else False
            
            fig_bar.add_trace(go.Box(
                y=data,
                x=[regime] * len(data),
                name=config,
                marker_color=colors_dict[config],
                boxpoints='all',
                jitter=0.5,
                pointpos=-2.0,
                showlegend=showlegend,
                legendgroup=config,
                offsetgroup=config
            ), row=row, col=1)

fig_bar.update_layout(
    #template='plotly_white',
    height=600,
    width=600,
    boxmode='group',
    boxgroupgap=0.6,
    boxgap=0.3,
    legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5, title_text="Config")
)

fig_bar.update_yaxes(title_text=f"{seg_meta['label']} [{seg_meta['arrow']}]", row=1, col=1)
fig_bar.update_yaxes(
    title_text=f"{disp_meta['label']} [{disp_meta['arrow']}]", 
    autorange="reversed" if disp_meta['arrow'] == '% ↓' else None,
    row=2, col=1
)
fig_bar.update_xaxes(title_text="Experiment (Entropy/Confidence)", row=2, col=1)

save_figure(fig_bar, height=800, name='H07F01', folder='results', standoff=12, lrtb_margin=(40, 20, 30, 0), skip_sync=skip_sync)



# %% H07F02_Lineplot_NLEandSERR
# H07F02_Lineplot_NLEandSERR

target_exps = ['exp01']

# Filter data
df_entropy_filtered = df_entropy[
    (df_entropy['experiment'].isin(target_exps))
].copy()

# Map SEG/DISP to ST
df_entropy_filtered['config_mapped'] = df_entropy_filtered['config'].replace({'SEG': 'ST', 'DISP': 'ST'})
target_configs = ['ST', 'MT', 'MT-KD']

df_entropy_filtered = df_entropy_filtered[df_entropy_filtered['config_mapped'].isin(target_configs)]

fig_line = make_subplots(
    rows=4, cols=2, 
    specs=[
        [{"colspan": 2}, None],
        [{}, {}],
        [{"colspan": 2}, None],
        [{}, {}]
    ],
    vertical_spacing=0.08,
    horizontal_spacing=0.05,
    subplot_titles=(
        "NLE - Encoder", 
        "NLE - Segmentation Decoder", "NLE - Disparity Decoder",
        "SERR - Encoder", 
        "SERR - Segmentation Decoder", "SERR - Disparity Decoder"
    )
)

def hex_to_rgba(hex_color, alpha=1.0):
    if hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({r},{g},{b},{alpha})'
    return hex_color

colors = {
    'ST': {'base': px.colors.qualitative.Plotly[0], 'name': 'ST'},
    'MT': {'base': px.colors.qualitative.Plotly[1], 'name': 'MT'},
    'MT-KD': {'base': px.colors.qualitative.Plotly[2], 'name': 'MT-KD'},
}

for cfg in colors:
    base_col = colors[cfg]['base']
    colors[cfg]['val'] = hex_to_rgba(base_col, 1.0)
    colors[cfg]['fill'] = hex_to_rgba(base_col, 0.2)

def format_layer_name(col):
    col = col.replace('encoder.stages_', 'Stage ')
    col = col.replace('decoders.segmentation.decoder.blocks.', 'Block ')
    #col = col.replace('decoders.segmentation.intercept_head', 'Intercept Head')
    col = col.replace('decoders.segmentation.decoder.final_block', 'Final Block')
    #col = col.replace('decoders.segmentation.head', 'Head')
    col = col.replace('decoders.disparity.decoder.blocks.', 'Block ')
    #col = col.replace('decoders.disparity.intercept_head', 'Intercept Head')
    col = col.replace('decoders.disparity.decoder.final_block', 'Final Block')
    #col = col.replace('decoders.disparity.head', 'Head')
    return col

layer_names = [c for c in df_entropy.columns if c.endswith('_norm_entropy')]
layers = [c.replace('_norm_entropy', '') for c in layer_names]

encoder_layers = [f'encoder.stages_{i}' for i in range(4)]
seg_layers = [f'decoders.segmentation.decoder.blocks.{i}' for i in range(3)] + \
             ['decoders.segmentation.decoder.final_block']
disp_layers = [f'decoders.disparity.decoder.blocks.{i}' for i in range(3)] + \
              ['decoders.disparity.decoder.final_block']

groups = [
    ("Encoder", encoder_layers, 1, 1),
    ("Segmentation", seg_layers, 2, 1),
    ("Disparity", disp_layers, 2, 2)
]

for metric, base_row in [('_norm_entropy', 0), ('_erank_ratio', 2)]:
    for group_name, group_layers, r_offset, c in groups:
        row = base_row + r_offset
        for cfg in target_configs:
            cfg_df = df_entropy_filtered[df_entropy_filtered['config_mapped'] == cfg]
            
            medians = []
            mins = []
            maxs = []
            valid_formatted_layers = []
            
            for layer in group_layers:
                col_name = f"{layer}{metric}"
                if col_name in cfg_df.columns:
                    vals = cfg_df[col_name].dropna()
                    if not vals.empty:
                        medians.append(vals.median())
                        mins.append(vals.min())
                        maxs.append(vals.max())
                        valid_formatted_layers.append(format_layer_name(layer))
            
            if valid_formatted_layers:
                # Show legend only once per config (in the first encoder subplot)
                showlegend = True if (base_row == 0 and r_offset == 1) else False
                
                # Ribbon (Min-Max)
                fig_line.add_trace(go.Scatter(
                    x=valid_formatted_layers + valid_formatted_layers[::-1],
                    y=maxs + mins[::-1],
                    fill='toself',
                    fillcolor=colors[cfg]['fill'],
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip',
                    legendgroup=cfg
                ), row=row, col=c)
                
                # Median Line (Solid)
                fig_line.add_trace(go.Scatter(
                    x=valid_formatted_layers, y=medians,
                    mode='lines+markers',
                    line=dict(color=colors[cfg]['val'], width=2),
                    name=colors[cfg]['name'],
                    legendgroup=cfg,
                    showlegend=showlegend
                ), row=row, col=c)

fig_line.update_layout(
    template='plotly_white',
    height=1200,
    width=1100,
    legend=dict(
        orientation="h", 
        yanchor="top", 
        y=-0.08, 
        xanchor="center", 
        x=0.5, 
        title_text="Config"
    )
)

fig_line.update_yaxes(title_text="NLE", range=[0, 1], row=1, col=1)
fig_line.update_yaxes(title_text="NLE", range=[0, 1], row=2, col=1)
fig_line.update_yaxes(title_text="NLE", range=[0, 1], row=2, col=2)
fig_line.update_yaxes(title_text="SERR", range=[0, 1], row=3, col=1)
fig_line.update_yaxes(title_text="SERR", range=[0, 1], row=4, col=1)
fig_line.update_yaxes(title_text="SERR", range=[0, 1], row=4, col=2)

fig_line.update_xaxes(tickangle=45)

save_figure(fig_line, height=1200, name='H07F02', lrtb_margin=(40, 10, 60, 80), standoff=None, folder='results', skip_sync=skip_sync)

# %%