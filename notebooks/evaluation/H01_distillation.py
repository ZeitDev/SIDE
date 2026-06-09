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
fallback_opacity = 0.5

metrics = ['DICE_score', 'AbsRel_rate', 'Bad3_rate']

METRIC_META = {
    'DICE_score': {'label': 'DICE Score', 'short': 'DICE', 'arrow': '↑', 'suffix': '/instrument_mean', 'task': 'segmentation'},
    'IoU_score': {'label': 'IoU Score', 'short': 'IoU', 'arrow': '↑', 'suffix': '/instrument_mean', 'task': 'segmentation'},
    'AbsRel_rate': {'label': 'AbsRel Rate', 'short': 'AbsRel', 'arrow': '↓', 'suffix': '', 'task': 'disparity'},
    'EPE_px': {'label': 'EPE', 'short': 'EPE', 'arrow': '↓', 'suffix': '', 'task': 'disparity'},
    'MAE_mm': {'label': 'MAE', 'short': 'MAE', 'arrow': '↓', 'suffix': '', 'task': 'disparity'},
    'Bad3_rate': {'label': 'Bad3 Rate', 'short': 'Bad3', 'arrow': '↓', 'suffix': '', 'task': 'disparity'},
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

# Group and calculate mean/std
agg_dict = {}
for metric in metrics:
    agg_dict[f'{metric}_mean'] = (metric, 'mean')
    agg_dict[f'{metric}_std'] = (metric, 'std')

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
b_st_seg = baseline_data.loc['SEG', 'DICE_score_mean'] if 'SEG' in baseline_data.index else np.nan
b_st_disp = baseline_data.loc['DISP', 'AbsRel_rate_mean'] if 'DISP' in baseline_data.index else np.nan
b_mt_seg = baseline_data.loc['MT', 'DICE_score_mean'] if 'MT' in baseline_data.index else np.nan
b_mt_disp = baseline_data.loc['MT', 'AbsRel_rate_mean'] if 'MT' in baseline_data.index else np.nan
b_mtkd_seg = baseline_data.loc['MT-KD', 'DICE_score_mean'] if 'MT-KD' in baseline_data.index else np.nan
b_mtkd_disp = baseline_data.loc['MT-KD', 'AbsRel_rate_mean'] if 'MT-KD' in baseline_data.index else np.nan

for exp in experiments:
    exp_data = grouped[grouped['experiment'] == exp].set_index('config')
    
    # Populate with actual data or fallback to exp01 baseline
    seg_matrix.loc[exp, 'ST'] = exp_data.loc['SEG', 'DICE_score_mean'] if 'SEG' in exp_data.index else b_st_seg
    seg_is_fallback.loc[exp, 'ST'] = 'SEG' not in exp_data.index and exp != 'exp01'
    seg_matrix.loc[exp, 'MT'] = exp_data.loc['MT', 'DICE_score_mean'] if 'MT' in exp_data.index else b_mt_seg
    seg_is_fallback.loc[exp, 'MT'] = 'MT' not in exp_data.index and exp != 'exp01'
    seg_matrix.loc[exp, 'MT-KD'] = exp_data.loc['MT-KD', 'DICE_score_mean'] if 'MT-KD' in exp_data.index else b_mtkd_seg
    seg_is_fallback.loc[exp, 'MT-KD'] = 'MT-KD' not in exp_data.index and exp != 'exp01'
    
    disp_matrix.loc[exp, 'ST'] = exp_data.loc['DISP', 'AbsRel_rate_mean'] if 'DISP' in exp_data.index else b_st_disp
    disp_is_fallback.loc[exp, 'ST'] = 'DISP' not in exp_data.index and exp != 'exp01'
    disp_matrix.loc[exp, 'MT'] = exp_data.loc['MT', 'AbsRel_rate_mean'] if 'MT' in exp_data.index else b_mt_disp
    disp_is_fallback.loc[exp, 'MT'] = 'MT' not in exp_data.index and exp != 'exp01'
    disp_matrix.loc[exp, 'MT-KD'] = exp_data.loc['MT-KD', 'AbsRel_rate_mean'] if 'MT-KD' in exp_data.index else b_mtkd_disp
    disp_is_fallback.loc[exp, 'MT-KD'] = 'MT-KD' not in exp_data.index and exp != 'exp01'

seg_matrix = seg_matrix.astype(float)
disp_matrix = disp_matrix.astype(float)

# Common styling
colors_dict = {'ST': px.colors.qualitative.Plotly[0], 'MT': px.colors.qualitative.Plotly[1], 'MT-KD': px.colors.qualitative.Plotly[2]}
seg_meta = METRIC_META['DICE_score']
disp_meta = METRIC_META['AbsRel_rate']

# %% H01T01_Table_GlobalResultsMatrix (DICE, AbsRel, Bad3)
# H01T01_Table_GlobalResultsMatrix (DICE, AbsRel, Bad3)

# Format strings "Mean ± Std"
for metric in metrics:
    grouped[f'{metric}_str'] = grouped.apply(
        lambda row: f"{row[f'{metric}_mean']:05.2f} ± {row[f'{metric}_std']:05.2f}" if pd.notna(row[f'{metric}_mean']) else "-", 
        axis=1
    )

# Melt and pivot
value_vars = [f'{metric}_str' for metric in metrics]
melted = grouped.melt(id_vars=['experiment', 'config'], value_vars=value_vars, var_name='Metric', value_name='Value')

# Map metric strings to clean LaTeX labels
metric_map = {f'{metric}_str': f"{METRIC_META[metric]['short']} ({METRIC_META[metric]['arrow']})" for metric in metrics}
melted['Metric'] = melted['Metric'].replace(metric_map)

# Exclusion logic for LaTeX table: No segmentation metrics for DISP config, No disparity metrics for SEG config
melted_table = melted.copy()
for metric in metrics:
    meta = METRIC_META[metric]
    label = metric_map[f'{metric}_str']
    if meta['task'] == 'segmentation':
        melted_table = melted_table[~((melted_table['Metric'] == label) & (melted_table['config'] == 'DISP'))]
    elif meta['task'] == 'disparity':
        melted_table = melted_table[~((melted_table['Metric'] == label) & (melted_table['config'] == 'SEG'))]

# Merge SEG and DISP into 'ST'
melted_table['config'] = melted_table['config'].replace({'SEG': 'ST', 'DISP': 'ST'})

# Pivot
pivot_combined = melted_table.pivot(index=['Metric', 'experiment'], columns='config', values='Value')

# Extract experiment number (e.g., 'exp01' -> '01')
pivot_combined.index = pivot_combined.index.set_levels(
    pivot_combined.index.levels[1].str.extract(r'(\d+)')[0].values,
    level='experiment'
)

# Fill NaN with -
pivot_combined = pivot_combined.fillna('-')

# Order rows: by metric sequence, then experiment ID
metrics_order = [metric_map[f'{metric}_str'] for metric in metrics]
pivot_combined = pivot_combined.reindex(metrics_order, level=0)

# Clean up index and column names
pivot_combined.index.names = ['Metric', 'ID']
pivot_combined.columns.name = None

# Order columns: ST, MT, MT-KD
desired_order = ['ST', 'MT', 'MT-KD']
ordered_cols = [c for c in desired_order if c in pivot_combined.columns]
pivot_combined = pivot_combined[ordered_cols]

# Print LaTeX with multirow for Metric and flattened header
latex_output = pivot_combined.to_latex(
    escape=False, 
    index=True, 
    multirow=True, 
    index_names=True,
    column_format='ll' + 'c' * len(pivot_combined.columns)
)

print(latex_output)

# %% H01F01_Barplot_E01ConfigStability (ST vs. MT vs. MT-KD of Exp01)
# H01F01_Barplot_ConfigStability (ST vs. MT vs. MT-KD of Exp01)

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
    
    fig_bar.add_trace(go.Bar(
        x=[seg_data.mean()],
        y=[config],
        error_x=dict(type='data', array=[seg_data.std()]),
        orientation='h',
        name=config,
        marker_color=colors_dict[config],
        showlegend=False
    ), row=1, col=1)
    
    fig_bar.add_trace(go.Bar(
        x=[disp_data.mean()],
        y=[config],
        error_x=dict(type='data', array=[disp_data.std()]),
        orientation='h',
        name=config,
        marker_color=colors_dict[config],
        showlegend=False
    ), row=1, col=2)

fig_bar.update_layout(
    template='plotly_white',
    height=400,
)
fig_bar.update_xaxes(title_text=f"{seg_meta['label']} ({seg_meta['arrow']})", rangemode='tozero', row=1, col=1)
fig_bar.update_xaxes(
    title_text=f"{disp_meta['label']} ({disp_meta['arrow']})", 
    rangemode='tozero', 
    autorange="reversed" if disp_meta['arrow'] == '↓' else None,
    row=1, col=2
)
fig_bar.update_yaxes(title_text="Experiment 01 Config", autorange="reversed", row=1, col=1)
fig_bar.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2) # Hide tick labels and title

save_figure(fig_bar, name='H01F01_Barplot_E01ConfigStability', lrtb_margin=(40, 20, 20, 0), folder='results', skip_sync=skip_sync)

# %% H01F02_Dumbbell_ConfigComparison (ST vs. MT vs. MT-KD)
# H01F02_Dumbbell_ConfigComparison (ST vs. MT vs. MT-KD)

fig_dumb = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Segmentation", "Disparity"),
    horizontal_spacing=0.05
)

# Legend-only dummy traces for full opacity
for config in ['ST', 'MT', 'MT-KD']:
    fig_dumb.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        name=config,
        marker=dict(color=colors_dict[config], size=10, opacity=1.0),
        legendgroup=config,
        showlegend=True
    ))

# For Segmentation (Col 1)
for i, exp in enumerate(experiments):
    is_fb = seg_is_fallback.loc[exp, ['ST', 'MT', 'MT-KD']]
    if not is_fb.all(): # Only plot line if at least one config is NOT a fallback
        row_vals = seg_matrix.loc[exp, ['ST', 'MT', 'MT-KD']].dropna()
        if not row_vals.empty:
            fig_dumb.add_trace(go.Scatter(
                x=[row_vals.min(), row_vals.max()],
                y=[y_labels[i], y_labels[i]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)

for config in ['ST', 'MT', 'MT-KD']:
    # Show markers only for experiments that have at least one actual (non-fallback) measurement
    # This avoids rows filled entirely with ghost/baseline data
    has_actual = ~seg_is_fallback.all(axis=1)
    fig_dumb.add_trace(go.Scatter(
        x=seg_matrix.loc[has_actual, config],
        y=[y_labels[i] for i, actual in enumerate(has_actual) if actual],
        mode='markers',
        name=config,
        marker=dict(
            color=colors_dict[config], 
            size=10,
            opacity=[fallback_opacity if is_fb else 1.0 for is_fb in seg_is_fallback.loc[has_actual, config]]
        ),
        legendgroup=config,
        showlegend=False
    ), row=1, col=1)

# For Disparity (Col 2)
for i, exp in enumerate(experiments):
    is_fb = disp_is_fallback.loc[exp, ['ST', 'MT', 'MT-KD']]
    if not is_fb.all():
        row_vals = disp_matrix.loc[exp, ['ST', 'MT', 'MT-KD']].dropna()
        if not row_vals.empty:
            fig_dumb.add_trace(go.Scatter(
                x=[row_vals.min(), row_vals.max()],
                y=[y_labels[i], y_labels[i]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=2)

for config in ['ST', 'MT', 'MT-KD']:
    has_actual = ~disp_is_fallback.all(axis=1)
    fig_dumb.add_trace(go.Scatter(
        x=disp_matrix.loc[has_actual, config],
        y=[y_labels[i] for i, actual in enumerate(has_actual) if actual],
        mode='markers',
        name=config,
        marker=dict(
            color=colors_dict[config], 
            size=10,
            opacity=[fallback_opacity if is_fb else 1.0 for is_fb in disp_is_fallback.loc[has_actual, config]]
        ),
        legendgroup=config,
        showlegend=False
    ), row=1, col=2)

fig_dumb.update_layout(
    template='plotly_white',
    height=450,
    width=850,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    legend_title_text="Config"
)

fig_dumb.update_xaxes(title_text=f"{seg_meta['label']} ({seg_meta['arrow']})", row=1, col=1)
fig_dumb.update_xaxes(title_text=f"{disp_meta['label']} ({disp_meta['arrow']})", autorange="reversed" if disp_meta['arrow'] == '↓' else None, row=1, col=2)
fig_dumb.update_yaxes(
    title_text="Experiment", 
    type='category',
    categoryorder='array',
    categoryarray=y_labels,
    autorange="reversed", 
    row=1, col=1
)
fig_dumb.update_yaxes(
    showticklabels=False, 
    type='category',
    categoryorder='array',
    categoryarray=y_labels,
    autorange="reversed", 
    row=1, col=2
)

save_figure(fig_dumb, name='H01F02_Dumbbell_ConfigComparison', lrtb_margin=(40, 20, 20, 60), folder='results', skip_sync=skip_sync)

# %% H01F03_Heatmap_ConfigComparison (ST vs. MT vs. MT-KD Delta)
# H01F03_Heatmap_ConfigComparison (ST vs. MT vs. MT-KD Delta)

# Compute Deltas (Percentage points) relative to ST of the SAME experiment
seg_delta = seg_matrix.sub(seg_matrix['ST'], axis=0)
disp_delta = disp_matrix.sub(disp_matrix['ST'], axis=0)

fig_heat = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Segmentation", "Disparity"),
    horizontal_spacing=0.2
)

# Overlay setup: Translucent white for fallback cells (so it looks translucent instead of grey)
seg_mask_z = seg_is_fallback.astype(float).replace(0, np.nan)
disp_mask_z = disp_is_fallback.astype(float).replace(0, np.nan)

# Use white with transparency to make it look "washed out" or translucent
mask_color = f'rgba(255,255,255,{1 - fallback_opacity})'

# --- Segmentation (Col 1) ---
z_seg = seg_delta.values if seg_meta['arrow'] == '↑' else -seg_delta.values
fig_heat.add_trace(go.Heatmap(
    z=z_seg, x=seg_delta.columns, y=y_labels,
    colorscale='RdBu', zmid=0,
    text=np.round(seg_delta.values, 2), texttemplate="%{text}",
    showscale=True, xgap=2, ygap=2,
    colorbar=dict(x=0.4, title=f"Δ {seg_meta['short']}<br>(pp, {seg_meta['arrow']})", dtick=2)
), row=1, col=1)

fig_heat.add_trace(go.Heatmap(
    z=seg_mask_z.values, x=seg_is_fallback.columns, y=y_labels,
    colorscale=[[0, mask_color], [1, mask_color]],
    showscale=False, hoverinfo='skip', xgap=2, ygap=2
), row=1, col=1)

# --- Disparity (Col 2) ---
z_disp = disp_delta.values if disp_meta['arrow'] == '↑' else -disp_delta.values
disp_max_bound = int(np.ceil(np.nanmax(np.abs(z_disp)) / 2.0) * 2)
disp_max_bound = max(disp_max_bound, 2)
tick_vals = list(range(-disp_max_bound, disp_max_bound + 1, 2))[1:-1]

fig_heat.add_trace(go.Heatmap(
    z=z_disp, x=disp_delta.columns, y=y_labels,
    colorscale='RdBu', zmin=-disp_max_bound, zmax=disp_max_bound,
    text=np.round(disp_delta.values, 2), texttemplate="%{text}",
    showscale=True, xgap=2, ygap=2,
    colorbar=dict(x=1.0, title=f"Δ {disp_meta['short']}<br>(pp, {disp_meta['arrow']})", tickmode='array', tickvals=tick_vals, ticktext=[(v if disp_meta['arrow'] == '↑' else -v) for v in tick_vals])
), row=1, col=2)

fig_heat.add_trace(go.Heatmap(
    z=disp_mask_z.values, x=disp_is_fallback.columns, y=y_labels,
    colorscale=[[0, mask_color], [1, mask_color]],
    showscale=False, hoverinfo='skip', xgap=2, ygap=2
), row=1, col=2)

fig_heat.update_layout(
    template='plotly_white',
    height=450,
    width=850
)

# Axis titles
fig_heat.update_xaxes(title_text="Config")
fig_heat.update_yaxes(title_text="Experiment", autorange="reversed", row=1, col=1)

# Reverse y-axis to put Exp 01 at the top and hide tick labels on right plot
fig_heat.update_yaxes(showticklabels=False, autorange="reversed", row=1, col=2)

save_figure(fig_heat, name='H01F03_Heatmap_ConfigComparison', lrtb_margin=(40, 20, 20, 20), folder='results', skip_sync=skip_sync)

# %%