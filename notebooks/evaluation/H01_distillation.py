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
fallback_opacity = 0.5

CHART_CONFIG = {
    'H01F01': {
        'x1': dict(range=[30, 56], dtick=5),
        'x2': dict(range=[35, 0], dtick=5),
    },
    'H01F02': {
        'x1': dict(range=[0, 60], dtick=10),
        'x2': dict(range=[70, 0], dtick=10),
    },
    'H01F03': {
        'x': dict(range=[2, 50], dtick=5),
        'y1': dict(range=[10, 60], dtick=10),
        'y2': dict(range=[60, 5], dtick=10),
    },
    'H01F04': {
        'y': dict(range=[0, 1], dtick=0.2)
    },
    'H01F05': {
        'y': dict(range=[0, 100], dtick=10)
    }
}

metrics = ['DICE_score', 'AbsRel_rate', 'Bad3_rate']

METRIC_META = {
    'DICE_score': {'label': 'DICE Score', 'short': 'DICE', 'arrow': '% ↑', 'suffix': '/instrument_mean', 'task': 'segmentation'},
    'IoU_score': {'label': 'IoU Score', 'short': 'IoU', 'arrow': '% ↑', 'suffix': '/instrument_mean', 'task': 'segmentation'},
    'AbsRel_rate': {'label': 'AbsRel Rate', 'short': 'AbsRel', 'arrow': '% ↓', 'suffix': '', 'task': 'disparity'},
    'EPE_px': {'label': 'EPE', 'short': 'EPE', 'arrow': 'px ↓', 'suffix': '', 'task': 'disparity'},
    'MAE_mm': {'label': 'MAE', 'short': 'MAE', 'arrow': 'mm ↓', 'suffix': '', 'task': 'disparity'},
    'Bad3_rate': {'label': 'Bad3 Rate', 'short': 'Bad3', 'arrow': '% ↓', 'suffix': '', 'task': 'disparity'},
}

# %% Prepare Data
# Prepare Data

df_bench = df_final.copy()

# Filter experiments
excluded_experiments = ['exp10']
df_bench = df_bench[df_bench['experiment'].str.startswith('exp', na=False)]
df_bench = df_bench[~df_bench['experiment'].isin(excluded_experiments)]

# Prepare metrics with fallbacks
for metric in metrics:
    meta = METRIC_META[metric]
    task = meta['task']
    col = f"metric.best_combined/performance/testing/{task}/{metric}{meta['suffix']}"
    fallback = f"metric.best_{task}/performance/testing/{task}/{metric}{meta['suffix']}"
    df_bench[metric] = df_bench[col].fillna(df_bench[fallback])

# Group and calculate median/min/max
agg_dict = {}
for metric in metrics:
    agg_dict[f'{metric}_median'] = (metric, 'median')
    agg_dict[f'{metric}_min'] = (metric, 'min')
    agg_dict[f'{metric}_max'] = (metric, 'max')

grouped = df_bench.groupby(['experiment', 'config']).agg(**agg_dict).reset_index()

# Prepare matrices for easier plotting (Absolute Scores)
experiments = sorted([exp for exp in df_bench['experiment'].dropna().unique() if exp.startswith('exp') and exp not in excluded_experiments])
y_labels = [e.replace('exp', '') for e in experiments]

seg_matrix = pd.DataFrame(index=experiments, columns=['ST', 'MT', 'MT-KD'])
disp_matrix = pd.DataFrame(index=experiments, columns=['ST', 'MT', 'MT-KD'])
seg_is_fallback = pd.DataFrame(index=experiments, columns=['ST', 'MT', 'MT-KD'], dtype=bool)
disp_is_fallback = pd.DataFrame(index=experiments, columns=['ST', 'MT', 'MT-KD'], dtype=bool)

# Extract baseline performance from exp01 for filling NaNs
baseline_data = grouped[grouped['experiment'] == 'exp01'].set_index('config')
b_st_seg = baseline_data.loc['SEG', 'DICE_score_median'] if 'SEG' in baseline_data.index else np.nan
b_st_disp = baseline_data.loc['DISP', 'AbsRel_rate_median'] if 'DISP' in baseline_data.index else np.nan
b_mt_seg = baseline_data.loc['MT', 'DICE_score_median'] if 'MT' in baseline_data.index else np.nan
b_mt_disp = baseline_data.loc['MT', 'AbsRel_rate_median'] if 'MT' in baseline_data.index else np.nan
b_mtkd_seg = baseline_data.loc['MT-KD', 'DICE_score_median'] if 'MT-KD' in baseline_data.index else np.nan
b_mtkd_disp = baseline_data.loc['MT-KD', 'AbsRel_rate_median'] if 'MT-KD' in baseline_data.index else np.nan

for exp in experiments:
    exp_data = grouped[grouped['experiment'] == exp].set_index('config')
    
    # Populate with actual data or fallback to exp01 baseline
    seg_matrix.loc[exp, 'ST'] = exp_data.loc['SEG', 'DICE_score_median'] if 'SEG' in exp_data.index else b_st_seg
    seg_is_fallback.loc[exp, 'ST'] = 'SEG' not in exp_data.index and exp != 'exp01'
    seg_matrix.loc[exp, 'MT'] = exp_data.loc['MT', 'DICE_score_median'] if 'MT' in exp_data.index else b_mt_seg
    seg_is_fallback.loc[exp, 'MT'] = 'MT' not in exp_data.index and exp != 'exp01'
    seg_matrix.loc[exp, 'MT-KD'] = exp_data.loc['MT-KD', 'DICE_score_median'] if 'MT-KD' in exp_data.index else b_mtkd_seg
    seg_is_fallback.loc[exp, 'MT-KD'] = 'MT-KD' not in exp_data.index and exp != 'exp01'
    
    disp_matrix.loc[exp, 'ST'] = exp_data.loc['DISP', 'AbsRel_rate_median'] if 'DISP' in exp_data.index else b_st_disp
    disp_is_fallback.loc[exp, 'ST'] = 'DISP' not in exp_data.index and exp != 'exp01'
    disp_matrix.loc[exp, 'MT'] = exp_data.loc['MT', 'AbsRel_rate_median'] if 'MT' in exp_data.index else b_mt_disp
    disp_is_fallback.loc[exp, 'MT'] = 'MT' not in exp_data.index and exp != 'exp01'
    disp_matrix.loc[exp, 'MT-KD'] = exp_data.loc['MT-KD', 'AbsRel_rate_median'] if 'MT-KD' in exp_data.index else b_mtkd_disp
    disp_is_fallback.loc[exp, 'MT-KD'] = 'MT-KD' not in exp_data.index and exp != 'exp01'

seg_matrix = seg_matrix.astype(float)
disp_matrix = disp_matrix.astype(float)

# Common styling
colors_dict = {'ST': px.colors.qualitative.Plotly[0], 'MT': px.colors.qualitative.Plotly[1], 'MT-KD': px.colors.qualitative.Plotly[2]}
seg_meta = METRIC_META['DICE_score']
disp_meta = METRIC_META['AbsRel_rate']



# %% H01F01_Boxplot_ConfigStability (ST vs. MT vs. MT-KD of Exp01)
# H01F01_Boxplot_ConfigStability (ST vs. MT vs. MT-KD of Exp01)

target_exp = 'exp01'
df_target = df_bench[df_bench['experiment'] == target_exp].copy()
df_baseline = df_bench[df_bench['experiment'] == 'exp01'].copy()

seg_metric = 'DICE_score'
disp_metric = 'AbsRel_rate'

# SEG data for target_exp (falling back to exp01 if configs are missing)
df_seg_box = df_target[df_target['config'].isin(['SEG', 'MT', 'MT-KD'])].copy()
missing_seg = [c for c in ['SEG', 'MT', 'MT-KD'] if c not in df_seg_box['config'].unique()]
if missing_seg:
    df_seg_box = pd.concat([df_seg_box, df_baseline[df_baseline['config'].isin(missing_seg)]])

df_seg_box['config'] = df_seg_box['config'].replace({'SEG': 'ST'})
df_seg_box['config'] = pd.Categorical(df_seg_box['config'], categories=['ST', 'MT', 'MT-KD'], ordered=True)
df_seg_box = df_seg_box.sort_values('config')

# DISP data for target_exp (falling back to exp01 if configs are missing)
df_disp_box = df_target[df_target['config'].isin(['DISP', 'MT', 'MT-KD'])].copy()
missing_disp = [c for c in ['DISP', 'MT', 'MT-KD'] if c not in df_disp_box['config'].unique()]
if missing_disp:
    df_disp_box = pd.concat([df_disp_box, df_baseline[df_baseline['config'].isin(missing_disp)]])

df_disp_box['config'] = df_disp_box['config'].replace({'DISP': 'ST'})
df_disp_box['config'] = pd.Categorical(df_disp_box['config'], categories=['ST', 'MT', 'MT-KD'], ordered=True)
df_disp_box = df_disp_box.sort_values('config')

fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"), horizontal_spacing=0.05)

for config in ['ST', 'MT', 'MT-KD']:
    seg_data = df_seg_box[df_seg_box['config'] == config][seg_metric]
    disp_data = df_disp_box[df_disp_box['config'] == config][disp_metric]
    
    fig_bar.add_trace(go.Box(
        x=seg_data,
        y=[config] * len(seg_data),
        orientation='h',
        name=config,
        marker_color=colors_dict[config],
        boxpoints='all',
        jitter=0.25,
        pointpos=-1.8,
        showlegend=False,
        legendgroup=config
    ), row=1, col=1)
    
    fig_bar.add_trace(go.Box(
        x=disp_data,
        y=[config] * len(disp_data),
        orientation='h',
        name=config,
        marker_color=colors_dict[config],
        boxpoints='all',
        jitter=0.25,
        pointpos=-1.8,
        showlegend=False,
        legendgroup=config
    ), row=1, col=2)

fig_bar.update_yaxes(title_text="Experiment 01 Config", autorange="reversed", row=1, col=1)
fig_bar.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2) # Hide tick labels and title
fig_bar.update_xaxes(title_text=seg_meta['label'] + f" [{seg_meta['arrow']}]", row=1, col=1)
fig_bar.update_xaxes(title_text=disp_meta['label'] + f" [{disp_meta['arrow']}]", row=1, col=2)

apply_chart_config(fig_bar, 'H01F01', CHART_CONFIG)
save_figure(fig_bar, name='H01F01', lrtb_margin=(40, 40, 20, 40), folder='results', skip_sync=skip_sync)

# %% H01F02_Barplot_ConfigValidationInterceptPerformance (Inter-Head Performance Comparison for Prediction vs. Projection)
# H01F02_Barplot_ConfigValidationInterceptPerformance (Inter-Head Performance Comparison for Prediction vs. Projection)

# target_exp = 'exp01'
# target_configs = ['MT', 'MT-KD']

# df_f02 = df_final[
#     (df_final['experiment'] == target_exp) & 
#     (df_final['config'].isin(target_configs))
# ].copy()

# # Metrics configuration
# metrics_meta = {
#     'segmentation': {
#         'prediction': 'metric.best/combined/performance_validation_segmentation_DICE_score_instrument_mean',
#         'projection': 'metric.best/combined/performance_validation_misc_interceptDICE_score',
#         'label': 'DICE Score [% ↑]',
#         'autorange': None
#     },
#     'disparity': {
#         'prediction': 'metric.best/combined/performance_validation_disparity_AbsRel_rate',
#         'projection': 'metric.best/combined/performance_validation_misc_interceptAbsRel_rate',
#         'label': 'AbsRel Rate [% ↓]',
#         'autorange': 'reversed'
#     }
# }

# fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"), horizontal_spacing=0.05)

# stages = ['Prediction', 'Projection']
# colors_dict = {
#     'MT': px.colors.qualitative.Plotly[1],
#     'MT-KD': px.colors.qualitative.Plotly[2]
# }

# for col, task in enumerate(['segmentation', 'disparity'], start=1):
#     meta = metrics_meta[task]
#     for config in target_configs:
#         for stage in stages:
#             col_name = meta['projection'] if stage == 'Projection' else meta['prediction']
#             data = df_f02[df_f02['config'] == config][col_name].dropna()
            
#             showlegend = True if col == 1 and stage == stages[0] else False
            
#             fig2.add_trace(go.Box(
#                 x=data,
#                 y=[stage] * len(data),
#                 orientation='h',
#                 name=config,
#                 marker_color=colors_dict[config],
#                 boxpoints='all',
#                 jitter=0.5,
#                 pointpos=-2.0,
#                 showlegend=showlegend,
#                 legendgroup=config,
#                 offsetgroup=config
#             ), row=1, col=col)

# fig2.update_layout(
#     # template='plotly_white',
#     height=450,
#     width=850,
#     boxmode='group',
#     boxgroupgap=0.6,
#     boxgap=0.3,
#     legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title_text="Experiment 01 Config")
# )

# fig2.update_xaxes(title_text=metrics_meta['segmentation']['label'], row=1, col=1)
# fig2.update_xaxes(
#     title_text=metrics_meta['disparity']['label'], 
#     autorange=metrics_meta['disparity']['autorange'],
#     row=1, col=2
# )
# fig2.update_yaxes(title_text="Decoder Head", autorange="reversed", row=1, col=1)
# fig2.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2)

# apply_chart_config(fig2, 'H01F02', CHART_CONFIG)
# save_figure(fig2, height=400, name='H01F02', lrtb_margin=(100, 20, 30, 0), folder='results', skip_sync=skip_sync)
# ! omitted

# %% H01F03_Lineplot_ConfigValidationPerformance (Train vs. Val Metrics over Epochs for MT vs. MT-KD)
# H01F03_Lineplot_ConfigValidationPerformance (Train vs. Val Metrics over Epochs for MT vs. MT-KD)

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

fig = make_subplots(rows=2, cols=1, subplot_titles=("Segmentation", "Disparity"), vertical_spacing=0.1, shared_xaxes=True)

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
    
def add_metric_traces(fig, df, metric_meta, row):
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
            
        showlegend = True if row == 1 else False
        
        # Val Ribbon
        fig.add_trace(go.Scatter(
            x=list(grouped_val['epoch']) + list(grouped_val['epoch'])[::-1],
            y=list(grouped_val['max']) + list(grouped_val['min'])[::-1],
            fill='toself',
            fillcolor=colors[color_key]['fill'],
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=1)
        
        # Val Line
        fig.add_trace(go.Scatter(
            x=grouped_val['epoch'],
            y=grouped_val['median'],
            mode='lines',
            line=dict(color=colors[color_key]['val'], width=2),
            name=display_name,
            showlegend=showlegend,
            legendgroup=display_name
        ), row=row, col=1)

add_metric_traces(fig, df_hist_filtered, metrics_dict['segmentation'], row=1)
add_metric_traces(fig, df_hist_filtered, metrics_dict['disparity'], row=2)

fig.update_layout(
    # template='plotly_white',
    height=800,
    width=850,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    legend_title_text="Experiment 01 Config"
)

fig.update_xaxes(title_text="Validation Epoch", tickvals=[2, 10, 20, 30, 40, 50], row=2, col=1)
fig.update_yaxes(title_text=f"{metrics_dict['segmentation']['short']} [{metrics_dict['segmentation']['arrow']}]", row=1, col=1)
fig.update_yaxes(
    title_text=f"{metrics_dict['disparity']['short']} [{metrics_dict['disparity']['arrow']}]", 
    autorange="reversed",
    row=2, col=1
)

apply_chart_config(fig, 'H01F03', CHART_CONFIG)
save_figure(fig, name='H01F03', lrtb_margin=(40, 20, 20, 60), folder='results', skip_sync=skip_sync)


# %% H01T01_SERRandNLE (SERR and NLE median +- min-max for SEG vs. DISP vs. MT vs. MT-KD on x-axis with stages (Encoder, SEG Decoder, DISP Decoder) on y-axis)
# H01T01_SERRandNLE (SERR and NLE median +- min-max for SEG vs. DISP vs. MT vs. MT-KD on x-axis with stages (Encoder, SEG Decoder, DISP Decoder) on y-axis)

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

target_exps_t01 = ['exp01']
df_entropy_filtered_t01 = df_entropy[df_entropy['experiment'].isin(target_exps_t01)].copy()

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
            
            for cfg in ['ST', 'MT', 'MT-KD']:
                if cfg == 'ST':
                    if group_name == 'SEG Decoder':
                        cfg_df = df_entropy_filtered_t01[df_entropy_filtered_t01['config'] == 'SEG']
                    elif group_name == 'DISP Decoder':
                        cfg_df = df_entropy_filtered_t01[df_entropy_filtered_t01['config'] == 'DISP']
                    else:
                        cfg_df = df_entropy_filtered_t01[df_entropy_filtered_t01['config'].isin(['SEG', 'DISP'])]
                else:
                    cfg_df = df_entropy_filtered_t01[df_entropy_filtered_t01['config'] == cfg]
                    
                if col_name in cfg_df.columns:
                    vals = cfg_df[col_name].dropna() * 100
                    if not vals.empty:
                        med = vals.median()
                        vmin = vals.min()
                        vmax = vals.max()
                        val_str = f"${med:05.2f}_{{-{med - vmin:05.2f}}}^{{+{vmax - med:05.2f}}}$"
                    else:
                        val_str = "-"
                else:
                    val_str = "-"
                    
                row_data[cfg] = val_str
            rows_t01.append(row_data)

df_t01 = pd.DataFrame(rows_t01)
df_t01.set_index(['Metric', 'Module', 'Layer'], inplace=True)
df_t01.columns.name = None

latex_output_t01 = df_t01.to_latex(
    escape=False, 
    index=True, 
    multirow=True, 
    index_names=True,
    column_format='lll' + 'c' * 3
)
print(f"\\renewcommand{{\\arraystretch}}{{1.4}}\n{latex_output_t01}")

# %% H01F04_Lineplot_SERR
# H01F04_Lineplot_SERR (SERR only)

# * 0% SERR means representational collapse (underfitting)
# * 50% SERR means balancing compression with Expression (keeping enough dimensions to explain)
# * 100% SERR means no compression, just memorizing training data (overfitting)

# ! Dont mix up predictive entropy (framed confidence in thesis) with representational entropy of channels
# * Predictive entropy: Uncertainty or complement of Confidence into its predictions
# * Representational entropy: How much information is retained in the internal representations
# * High representational entropy means expressiveness, low covariance, features are less dependent/redundant

target_exps = ['exp01']

# Filter data
df_entropy_filtered = df_entropy[
    (df_entropy['experiment'].isin(target_exps))
].copy()

# Keep SEG and DISP separate
df_entropy_filtered['config_mapped'] = df_entropy_filtered['config']
target_configs = ['MT', 'MT-KD'] #['SEG', 'DISP', 'MT', 'MT-KD']

df_entropy_filtered = df_entropy_filtered[df_entropy_filtered['config_mapped'].isin(target_configs)]

def hex_to_rgba(hex_color, alpha=1.0):
    if hex_color.startswith('#'):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({r},{g},{b},{alpha})'
    return hex_color

colors = {
    'SEG': {'base': '#3E459C', 'name': 'ST (SEG)'}, # Darker shade of ST blue
    'DISP': {'base': '#8F97F9', 'name': 'ST (DISP)'}, # Lighter shade of ST blue
    'MT': {'base': px.colors.qualitative.Plotly[1], 'name': 'MT'},
    'MT-KD': {'base': px.colors.qualitative.Plotly[2], 'name': 'MT-KD'},
}

for cfg in colors:
    base_col = colors[cfg]['base']
    colors[cfg]['val'] = hex_to_rgba(base_col, 1.0)
    colors[cfg]['fill'] = hex_to_rgba(base_col, 0.2)

def format_layer_name(col):
    if 'encoder.stages_' in col:
        return str(int(col.split('_')[-1]) + 1)
    if 'decoder.blocks.' in col:
        return str(int(col.split('.')[-1]) + 1)
    if 'final_block' in col:
        return '4'
    return col

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

fig_line_serr = make_subplots(
    rows=2, cols=2, 
    specs=[
        [{"colspan": 2}, None],
        [{}, {}]
    ],
    vertical_spacing=0.115,
    horizontal_spacing=0.05,
    shared_yaxes='rows',
    # subplot_titles=(
    #     "Encoder", 
    #     "Segmentation Decoder", "Disparity Decoder"
    # )
)

for group_name, group_layers, r_offset, c in groups:
    row = r_offset
    for cfg in target_configs:
        cfg_df = df_entropy_filtered[df_entropy_filtered['config_mapped'] == cfg]
        
        medians = []
        mins = []
        maxs = []
        valid_formatted_layers = []
        
        for layer in group_layers:
            col_name = f"{layer}_erank_ratio"
            if col_name in cfg_df.columns:
                vals = cfg_df[col_name].dropna()
                if not vals.empty:
                    medians.append(vals.median())
                    mins.append(vals.min())
                    maxs.append(vals.max())
                    valid_formatted_layers.append(format_layer_name(layer))
        
        if valid_formatted_layers:
            # Show legend only once per config (in the first encoder subplot)
            showlegend = True if r_offset == 1 else False
            
            # Add vertical line between Block 2 and Final Block for decoders
            if group_name in ["Segmentation", "Disparity"]:
                fig_line_serr.add_vline(
                    x=2.5, 
                    line_width=1, 
                    line_dash="dash", 
                    line_color="grey",
                    row=row, col=c
                )
                # Add "KD" annotation slightly below the top of the plot
                fig_line_serr.add_annotation(
                    x=2.75, y=0.9,
                    text="KD",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="grey", size=10),
                    row=row, col=c
                )

            # Ribbon (Min-Max)
            fig_line_serr.add_trace(go.Scatter(
                x=valid_formatted_layers + valid_formatted_layers[::-1],
                y=maxs + mins[::-1],
                fill='toself',
                fillcolor=colors[cfg]['fill'],
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=cfg
            ), row=row, col=c)
            
            # Median Line
            fig_line_serr.add_trace(go.Scatter(
                x=valid_formatted_layers, y=medians,
                mode='lines+markers',
                line=dict(color=colors[cfg]['val'], width=2),
                name=colors[cfg]['name'],
                legendgroup=cfg,
                showlegend=showlegend
            ), row=row, col=c)

fig_line_serr.update_layout(
    height=600,
    width=1100,
    legend=dict(
        orientation="h", 
        yanchor="top", 
        y=-0.125, 
        xanchor="center", 
        x=0.5, 
        title_text="Experiment 01 Config"
    )
)

fig_line_serr.update_yaxes(
    range=[0, 1],
    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ticktext=["0", "20", "40", "60", "80", "100"]
)
fig_line_serr.update_yaxes(title_text="SERR [%]", row=1, col=1)
fig_line_serr.update_yaxes(title_text="SERR [%]", row=2, col=1)
fig_line_serr.update_xaxes(title_text="Shared Encoder Layer", row=1, col=1)
fig_line_serr.update_xaxes(title_text="Segmentation Decoder Layer", row=2, col=1)
fig_line_serr.update_xaxes(title_text="Disparity Decoder Layer", row=2, col=2)

apply_chart_config(fig_line_serr, 'H01F04', CHART_CONFIG)
save_figure(fig_line_serr, height=600, name='H01F04', lrtb_margin=(20, 10, 10, 10), standoff=5, folder='results', skip_sync=skip_sync)

# %% H01F04_NLE_Lineplot_NLE
# H01F04_NLE_Lineplot_NLE (NLE only)

fig_line_nle = make_subplots(
    rows=2, cols=2, 
    specs=[
        [{"colspan": 2}, None],
        [{}, {}]
    ],
    vertical_spacing=0.115,
    horizontal_spacing=0.05,
    shared_yaxes='rows',
    # subplot_titles=(
    #     "Encoder", 
    #     "Segmentation Decoder", "Disparity Decoder"
    # )
)

for group_name, group_layers, r_offset, c in groups:
    row = r_offset
    for cfg in target_configs:
        cfg_df = df_entropy_filtered[df_entropy_filtered['config_mapped'] == cfg]
        
        medians = []
        mins = []
        maxs = []
        valid_formatted_layers = []
        
        for layer in group_layers:
            col_name = f"{layer}_norm_entropy"
            if col_name in cfg_df.columns:
                vals = cfg_df[col_name].dropna()
                if not vals.empty:
                    medians.append(vals.median())
                    mins.append(vals.min())
                    maxs.append(vals.max())
                    valid_formatted_layers.append(format_layer_name(layer))
        
        if valid_formatted_layers:
            # Show legend only once per config (in the first encoder subplot)
            showlegend = True if r_offset == 1 else False
            
            # Add vertical line between Block 2 and Final Block for decoders
            if group_name in ["Segmentation", "Disparity"]:
                fig_line_nle.add_vline(
                    x=2.5, 
                    line_width=1, 
                    line_dash="dash", 
                    line_color="grey",
                    row=row, col=c
                )
                # Add "KD" annotation slightly below the top of the plot
                fig_line_nle.add_annotation(
                    x=2.75, y=0.9,
                    text="KD",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="grey", size=10),
                    row=row, col=c
                )

            # Ribbon (Min-Max)
            fig_line_nle.add_trace(go.Scatter(
                x=valid_formatted_layers + valid_formatted_layers[::-1],
                y=maxs + mins[::-1],
                fill='toself',
                fillcolor=colors[cfg]['fill'],
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=cfg
            ), row=row, col=c)
            
            # Median Line
            fig_line_nle.add_trace(go.Scatter(
                x=valid_formatted_layers, y=medians,
                mode='lines+markers',
                line=dict(color=colors[cfg]['val'], width=2),
                name=colors[cfg]['name'],
                legendgroup=cfg,
                showlegend=showlegend
            ), row=row, col=c)

fig_line_nle.update_layout(
    height=600,
    width=1100,
    legend=dict(
        orientation="h", 
        yanchor="top", 
        y=-0.125, 
        xanchor="center", 
        x=0.5, 
        title_text="Experiment 01 Config"
    )
)

fig_line_nle.update_yaxes(
    range=[0, 1],
    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ticktext=["0", "20", "40", "60", "80", "100"]
)
fig_line_nle.update_yaxes(title_text="NLE [%]", row=1, col=1)
fig_line_nle.update_yaxes(title_text="NLE [%]", row=2, col=1)
fig_line_nle.update_xaxes(title_text="Shared Encoder Layer", row=1, col=1)
fig_line_nle.update_xaxes(title_text="Segmentation Decoder Layer", row=2, col=1)
fig_line_nle.update_xaxes(title_text="Disparity Decoder Layer", row=2, col=2)

apply_chart_config(fig_line_nle, 'H01F04', CHART_CONFIG)
save_figure(fig_line_nle, height=600, name='H01F05', lrtb_margin=(20, 10, 10, 10), standoff=5, folder='results', skip_sync=skip_sync)

# %%
