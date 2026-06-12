# %%
import os, sys
sys.path.append(os.path.dirname('../../'))
os.chdir(os.path.dirname('../../'))

import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from notebooks.figures.helpers import save_figure

# %%
with open('./notebooks/evaluation/storage/dataframes.pkl', 'rb') as f:
    data = pickle.load(f)
    
    df_final = data['final']
    df_params = data['params']
    df_historic = data['historic']

# %% Settings
skip_sync = False
seg_metric = 'DICE_score'  # options: DICE_score, IoU_score
disp_metric = 'Bad3_rate' # options: AbsRel_rate, EPE_px, MAE_mm, Bad3_rate

METRIC_META = {
    'DICE_score': {'label': 'DICE Score', 'short': 'DICE', 'arrow': '↑', 'suffix': '/instrument_mean'},
    'IoU_score': {'label': 'IoU Score', 'short': 'IoU', 'arrow': '↑', 'suffix': '/instrument_mean'},
    'AbsRel_rate': {'label': 'AbsRel Rate', 'short': 'AbsRel', 'arrow': '↓', 'suffix': ''},
    'EPE_px': {'label': 'EPE', 'short': 'EPE', 'arrow': '↓', 'suffix': ''},
    'MAE_mm': {'label': 'MAE', 'short': 'MAE', 'arrow': '↓', 'suffix': ''},
    'Bad3_rate': {'label': 'Bad3 Rate', 'short': 'Bad3', 'arrow': '↓', 'suffix': ''},
}
seg_meta = METRIC_META[seg_metric]
disp_meta = METRIC_META[disp_metric]


# %% Figure 1 - Heatmap: Ablations Delta for ST, MT, MT-KD
# Prepare Data
seg_col = f"metric.best_combined/performance/testing/segmentation/{seg_metric}{seg_meta['suffix']}"
disp_col = f"metric.best_combined/performance/testing/disparity/{disp_metric}{disp_meta['suffix']}"
seg_fallback = f"metric.best_segmentation/performance/testing/segmentation/{seg_metric}{seg_meta['suffix']}"
disp_fallback = f"metric.best_disparity/performance/testing/disparity/{disp_metric}{disp_meta['suffix']}"

df_bench = df_final.copy()
df_bench[seg_metric] = df_bench[seg_col].fillna(df_bench[seg_fallback])
df_bench[disp_metric] = df_bench[disp_col].fillna(df_bench[disp_fallback])

grouped = df_bench.groupby(['experiment', 'config']).agg(
    seg_mean=(seg_metric, 'mean'),
    disp_mean=(disp_metric, 'mean')
).reset_index()

excluded_experiments = ['exp10']  # Specify experiments to exclude here
experiments = sorted([exp for exp in grouped['experiment'].dropna().unique() if exp.startswith('exp') and exp not in excluded_experiments])

seg_matrix = pd.DataFrame(index=experiments, columns=['ST', 'MT', 'MT-KD'])
disp_matrix = pd.DataFrame(index=experiments, columns=['ST', 'MT', 'MT-KD'])

for exp in experiments:
    exp_data = grouped[grouped['experiment'] == exp].set_index('config')
    seg_matrix.loc[exp] = [
        exp_data.loc['SEG', 'seg_mean'] if 'SEG' in exp_data.index else np.nan,
        exp_data.loc['MT', 'seg_mean'] if 'MT' in exp_data.index else np.nan,
        exp_data.loc['MT-KD', 'seg_mean'] if 'MT-KD' in exp_data.index else np.nan
    ]
    disp_matrix.loc[exp] = [
        exp_data.loc['DISP', 'disp_mean'] if 'DISP' in exp_data.index else np.nan,
        exp_data.loc['MT', 'disp_mean'] if 'MT' in exp_data.index else np.nan,
        exp_data.loc['MT-KD', 'disp_mean'] if 'MT-KD' in exp_data.index else np.nan
    ]

# Forward fill missing configurations using baseline (exp01)
baseline_seg = seg_matrix.loc['exp01'].copy()
baseline_disp = disp_matrix.loc['exp01'].copy()

for exp in experiments:
    for col in ['ST', 'MT', 'MT-KD']:
        if pd.isna(seg_matrix.loc[exp, col]):
            seg_matrix.loc[exp, col] = baseline_seg[col]
        if pd.isna(disp_matrix.loc[exp, col]):
            disp_matrix.loc[exp, col] = baseline_disp[col]

seg_matrix = seg_matrix.astype(float)
disp_matrix = disp_matrix.astype(float)

# Compute Deltas (Percentage points)
seg_delta = seg_matrix - seg_matrix.loc['exp01']
disp_delta = disp_matrix - disp_matrix.loc['exp01']

from plotly.subplots import make_subplots

fig_heat = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Segmentation", "Disparity"),
    horizontal_spacing=0.25
)

y_labels = [e.replace('exp', '') for e in experiments]

z_seg = seg_delta.values if seg_meta['arrow'] == '↑' else -seg_delta.values
fig_heat.add_trace(go.Heatmap(
    z=z_seg,
    x=seg_delta.columns,
    y=y_labels,
    colorscale='RdBu', 
    zmid=0,
    text=np.round(seg_delta.values, 2),
    texttemplate="%{text}",
    showscale=True,
    xgap=2, ygap=2,
    colorbar=dict(
        x=0.38, 
        title=f"Δ {seg_meta['short']}<br>(pp, {seg_meta['arrow']})",
        dtick=2
    )
), row=1, col=1)

# Dynamically set color and tick bounds symmetrically for Disp so the custom ticks don't go out of bounds
z_disp = disp_delta.values if disp_meta['arrow'] == '↑' else -disp_delta.values
disp_max_bound = int(np.ceil(np.nanmax(np.abs(disp_delta.values)) / 2.0) * 2)
disp_max_bound = max(disp_max_bound, 2)
tick_vals = list(range(-disp_max_bound, disp_max_bound + 1, 2))[1:-1]  # Exclude the outer bounds for ticks to avoid overlap with color limits

fig_heat.add_trace(go.Heatmap(
    z=z_disp,
    x=disp_delta.columns,
    y=y_labels,
    colorscale='RdBu', 
    zmin=-disp_max_bound,
    zmax=disp_max_bound,
    text=np.round(disp_delta.values, 2),
    texttemplate="%{text}",
    showscale=True,
    xgap=2, ygap=2,
    colorbar=dict(
        x=1.02, 
        title=f"Δ {disp_meta['short']}<br>(pp, {disp_meta['arrow']})",
        tickmode='array',
        tickvals=tick_vals,
        ticktext=[(v if disp_meta['arrow'] == '↑' else -v) for v in tick_vals]
    )
), row=1, col=2)

fig_heat.update_layout(
    template='plotly_white',
    height=450,
    width=850
)

# Axis titles
fig_heat.update_xaxes(title_text="Config")
fig_heat.update_yaxes(title_text="Experiment", row=1, col=1)

# Reverse y-axis to put Exp 01 at the top
fig_heat.update_yaxes(autorange="reversed")

save_figure(fig_heat, name='H04_F01_heatmap_ablation_deltas', lrtb_margin=(40, 20, 20, 20), folder='results', skip_sync=skip_sync)

# %% Figure 2 - Box Plots: 2x3 ST, MT, MT-KD plots showing seeds across ablations
from plotly.subplots import make_subplots

experiments_fig2 = experiments.copy()
df_fig2 = df_bench[df_bench['experiment'].isin(experiments_fig2)].copy()

fig_box = make_subplots(
    rows=2, cols=3, 
    subplot_titles=("Config: ST", "Config: MT", "Config: MT-KD"), 
    shared_yaxes=True,
    shared_xaxes=True,
    vertical_spacing=0.025,
    horizontal_spacing=0.01
)
colors_dict = {'ST': px.colors.qualitative.Plotly[0], 'MT': px.colors.qualitative.Plotly[1], 'MT-KD': px.colors.qualitative.Plotly[2]}
grey_color = '#c0c0c0'

for i, config_alias in enumerate(['ST', 'MT', 'MT-KD']):
    for exp in experiments_fig2:
        # Determine data and fallback for Segmentation (Row 1)
        target_seg_config = 'SEG' if config_alias == 'ST' else config_alias
        df_seg = df_fig2[(df_fig2['config'] == target_seg_config) & (df_fig2['experiment'] == exp)]
        is_seg_fb = df_seg.empty
        if is_seg_fb:
            df_seg = df_fig2[(df_fig2['config'] == target_seg_config) & (df_fig2['experiment'] == 'exp01')]
            
        # Determine data and fallback for Disparity (Row 2)
        target_disp_config = 'DISP' if config_alias == 'ST' else config_alias
        df_disp = df_fig2[(df_fig2['config'] == target_disp_config) & (df_fig2['experiment'] == exp)]
        is_disp_fb = df_disp.empty
        if is_disp_fb:
            df_disp = df_fig2[(df_fig2['config'] == target_disp_config) & (df_fig2['experiment'] == 'exp01')]

        fig_box.add_trace(go.Box(
            y=df_seg[seg_metric],
            name=exp.replace('exp', ''),
            marker_color=grey_color if is_seg_fb else colors_dict[config_alias],
            marker_opacity=0.6 if is_seg_fb else 1.0,
            showlegend=False
        ), row=1, col=i+1)
        
        fig_box.add_trace(go.Box(
            y=df_disp[disp_metric],
            name=exp.replace('exp', ''),
            marker_color=grey_color if is_disp_fb else colors_dict[config_alias],
            marker_opacity=0.6 if is_disp_fb else 1.0,
            showlegend=False
        ), row=2, col=i+1)

fig_box.update_layout(
    template='plotly_white',
    height=600,
    margin=dict(t=60, b=40, l=60, r=20)
)

# Axis titles and styling
fig_box.update_yaxes(title_text=f"Segmentation {seg_meta['short']} ({seg_meta['arrow']})", dtick=5, row=1, col=1)
fig_box.update_yaxes(title_text=f"Disparity {disp_meta['short']} ({disp_meta['arrow']})", autorange="reversed" if disp_meta['arrow'] == '↓' else None, dtick=5, row=2, col=1)

# Apply shared settings to all subplots
for col in range(1, 4):
    fig_box.update_yaxes(dtick=5, row=1, col=col)
    fig_box.update_yaxes(autorange="reversed" if disp_meta['arrow'] == '↓' else None, dtick=5, row=2, col=col)
    
    fig_box.update_xaxes(title_text="Experiment", row=2, col=col, tickmode='linear', dtick=1)
    fig_box.update_xaxes(tickmode='linear', dtick=1, row=1, col=col)

save_figure(fig_box, height=600, name='H04_F02_boxplot_ablation_stability', lrtb_margin=(40, 0, 40, 40), folder='results', skip_sync=skip_sync)

# %%
