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
    
with open('./notebooks/evaluation/storage/entropy_metrics.pkl', 'rb') as f:
    df_entropy = pickle.load(f)

# %% Settings
# Settings
skip_sync = False

CHART_CONFIG = {
    'H07F01': {
        'y1': dict(range=[30, 60], dtick=5),
        'y2': dict(range=[55, 0], dtick=5),
    }
}

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

# %% H07F01_Boxplot_EntropyPerformance (01 vs. 02 vs. 03 vs. 09)
# H07F01_Boxplot_EntropyPerformance (01 vs. 02 vs. 03 vs. 09)

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
    ## template='plotly_white',
    height=600,
    width=600,
    boxmode='group',
    boxgroupgap=0.6,
    boxgap=0.3,
    legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5, title_text="Config")
)

fig_bar.update_yaxes(title_text=f"{seg_meta['label']} [{seg_meta['arrow']}]", row=1, col=1)
fig_bar.update_yaxes(
    title_text=f"{disp_meta['label']} [{disp_meta['arrow']}]", 
    autorange="reversed" if disp_meta['arrow'] == '% ↓' else None,
    row=2, col=1
)
fig_bar.update_xaxes(title_text="Experiment (Entropy/Confidence)", row=2, col=1)

apply_chart_config(fig_bar, 'H07F01', CHART_CONFIG)
save_figure(fig_bar, height=800, name='H07F01', folder='results', standoff=12, lrtb_margin=(40, 20, 30, 20), skip_sync=skip_sync)

# %% H07T01_SERRandNLE (SERR and NLE median +- min-max for MT-KD of exp 01, 02, 03, 09 on x-axis with stages on y-axis)
# H07T01_SERRandNLE (SERR and NLE median +- min-max for MT-KD of exp 01, 02, 03, 09 on x-axis with stages on y-axis)

def format_layer_name_t01(col):
    if 'encoder.stages_' in col:
        return str(int(col.split('_')[-1]) + 1)
    if 'decoder.blocks.' in col:
        return str(int(col.split('.')[-1]) + 1)
    if 'final_block' in col:
        return '4'
    return col

encoder_layers_t01 = [f'encoder.stages_{i}' for i in range(4)]
seg_layers_t01 = [f'decoders.segmentation.decoder.blocks.{i}' for i in range(3)] + ['decoders.segmentation.decoder.final_block']
disp_layers_t01 = [f'decoders.disparity.decoder.blocks.{i}' for i in range(3)] + ['decoders.disparity.decoder.final_block']

target_exps_t01 = ['exp01', 'exp02', 'exp03', 'exp09']
df_entropy_filtered_t01 = df_entropy[
    (df_entropy['experiment'].isin(target_exps_t01)) & 
    (df_entropy['config'] == 'MT-KD')
].copy()

metrics_t01 = [('_erank_ratio', 'SERR [\\%]'), ('_norm_entropy', 'NLE [\\%]')]
groups_t01 = [
    ("Encoder", encoder_layers_t01),
    ("SEG Decoder", seg_layers_t01),
    ("DISP Decoder", disp_layers_t01)
]

rows_t01 = []

for metric_suffix, metric_name in metrics_t01:
    for group_name, group_layers in groups_t01:
        for layer in group_layers:
            col_name = f"{layer}{metric_suffix}"
            layer_name = format_layer_name_t01(layer)
            
            row_data = {'Metric': metric_name, 'Module': group_name, 'Layer': layer_name}
            
            for exp in target_exps_t01:
                exp_df = df_entropy_filtered_t01[df_entropy_filtered_t01['experiment'] == exp]
                
                if col_name in exp_df.columns:
                    vals = exp_df[col_name].dropna() * 100
                    if not vals.empty:
                        med = vals.median()
                        vmin = vals.min()
                        vmax = vals.max()
                        val_str = f"${med:05.2f}_{{-{med - vmin:05.2f}}}^{{+{vmax - med:05.2f}}}$"
                    else:
                        val_str = "-"
                else:
                    val_str = "-"
                    
                row_data[exp] = val_str
            rows_t01.append(row_data)

df_t01 = pd.DataFrame(rows_t01)
df_t01.rename(columns={'exp01': '01', 'exp02': '02', 'exp03': '03', 'exp09': '09'}, inplace=True)
df_t01.set_index(['Metric', 'Module', 'Layer'], inplace=True)
df_t01.columns.name = None

latex_output_t01 = df_t01.to_latex(
    escape=False, 
    index=True, 
    multirow=True, 
    index_names=True,
    column_format='lll' + 'c' * 4
)
print(f"\\renewcommand{{\\arraystretch}}{{1.4}}\n{latex_output_t01}")

# %% H07F02_Lineplot_SERRandNLE (01 vs. 02 vs. 03 vs. 09)
# H07F02_Lineplot_SERRandNLE (01 vs. 02 vs. 03 vs. 09)

target_exps = ['exp01', 'exp09']#['exp01', 'exp02', 'exp03', 'exp09']

# Filter data
df_entropy_filtered = df_entropy[
    (df_entropy['experiment'].isin(target_exps)) &
    (df_entropy['config'] == 'MT-KD')
].copy()

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
    shared_yaxes='rows',
    subplot_titles=(
        "Encoder", 
        "Segmentation Decoder", "Disparity Decoder",
        "Encoder", 
        "Segmentation Decoder", "Disparity Decoder"
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
    'exp01': {'base': px.colors.qualitative.Plotly[3], 'name': '01 (Base)'},
    'exp02': {'base': px.colors.qualitative.Plotly[6], 'name': '02 (No Disparity Gate)'},
    'exp03': {'base': px.colors.qualitative.Plotly[7], 'name': '03 (No Spatial Distillation<br>Loss Scaling)'},
    'exp09': {'base': px.colors.qualitative.Plotly[9], 'name': '09 (Temperature = 4)'},
}

for exp in colors:
    base_col = colors[exp]['base']
    colors[exp]['val'] = hex_to_rgba(base_col, 1.0)
    colors[exp]['fill'] = hex_to_rgba(base_col, 0.2)

def format_layer_name(col):
    if 'encoder.stages_' in col:
        return str(int(col.split('_')[-1]) + 1)
    if 'decoder.blocks.' in col:
        return str(int(col.split('.')[-1]) + 1)
    if 'final_block' in col:
        return '4'
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

for metric, base_row in [('_erank_ratio', 0), ('_norm_entropy', 2)]:
    for group_name, group_layers, r_offset, c in groups:
        row = base_row + r_offset
        for exp in target_exps:
            exp_df = df_entropy_filtered[df_entropy_filtered['experiment'] == exp]
            
            medians = []
            mins = []
            maxs = []
            valid_formatted_layers = []
            
            for layer in group_layers:
                col_name = f"{layer}{metric}"
                if col_name in exp_df.columns:
                    vals = exp_df[col_name].dropna()
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
                    fillcolor=colors[exp]['fill'],
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip',
                    legendgroup=exp
                ), row=row, col=c)
                
                # Median Line
                fig_line.add_trace(go.Scatter(
                    x=valid_formatted_layers, y=medians,
                    mode='lines+markers',
                    line=dict(color=colors[exp]['val'], width=2),
                    name=colors[exp]['name'],
                    legendgroup=exp,
                    showlegend=showlegend
                ), row=row, col=c)

                # Add vertical line between Block 2 and Final Block for decoders
                if group_name in ["Segmentation", "Disparity"] and exp == target_exps[0]:
                    fig_line.add_vline(
                        x=2.5, 
                        line_width=1, 
                        line_dash="dash", 
                        line_color="grey",
                        row=row, col=c
                    )
                    # Add "KD" annotation slightly below the top of the plot
                    fig_line.add_annotation(
                        x=2.75, y=0.9,
                        text="KD",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="grey", size=10),
                        row=row, col=c
                    )

fig_line.update_layout(
    height=1200,
    width=1100,
    legend=dict(
        orientation="h", 
        yanchor="top", 
        y=0, 
        xanchor="center", 
        x=0.5, 
        title_text="Experiment (Entropy/Confidence)"
    )
)

fig_line.update_yaxes(
    range=[0, 1],
    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ticktext=["0", "20", "40", "60", "80", "100"]
)
fig_line.update_yaxes(title_text="SERR [%]", row=1, col=1)
fig_line.update_yaxes(title_text="SERR [%]", row=2, col=1)
fig_line.update_yaxes(title_text="NLE [%]", row=3, col=1)
fig_line.update_yaxes(title_text="NLE [%]", row=4, col=1)

fig_line.update_xaxes(title_text="Layer")

apply_chart_config(fig_line, 'H07F02', CHART_CONFIG)
save_figure(fig_line, height=1200, name='H07F02', lrtb_margin=(40, 10, 60, 80), standoff=None, folder='results', skip_sync=skip_sync)



# %%
