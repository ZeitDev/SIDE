# %%
import os, sys
sys.path.append(os.path.dirname('../../'))
os.chdir(os.path.dirname('../../'))

import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
disp_metric = 'AbsRel_rate' # options: AbsRel_rate, EPE_px, MAE_mm, Bad3_rate

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

# %% Prepare Data for Figures and Tables
# From the dictionary, we identify the key performance metrics (Tier 2 Focus)
seg_col = f"metric.best_combined/performance/testing/segmentation/{seg_metric}{seg_meta['suffix']}"
disp_col = f"metric.best_combined/performance/testing/disparity/{disp_metric}{disp_meta['suffix']}"

# If combined metrics aren't available for standalone, we might need to fallback to their specific ones.
# We will coalesce the metrics if they are NaN.
seg_fallback = f"metric.best_segmentation/performance/testing/segmentation/{seg_metric}{seg_meta['suffix']}"
disp_fallback = f"metric.best_disparity/performance/testing/disparity/{disp_metric}{disp_meta['suffix']}"

df_bench = df_final.copy()
df_bench[seg_metric] = df_bench[seg_col].fillna(df_bench[seg_fallback])
df_bench[disp_metric] = df_bench[disp_col].fillna(df_bench[disp_fallback])

# Group by experiment and config to get mean and std across seeds of config per experiment
grouped = df_bench.groupby(['experiment', 'config']).agg(
    seg_mean=(seg_metric, 'mean'),
    seg_std=(seg_metric, 'std'),
    disp_mean=(disp_metric, 'mean'),
    disp_std=(disp_metric, 'std')
).reset_index()

# Define common experiment list and labels
excluded_experiments = ['exp10']
experiments = sorted([exp for exp in df_bench['experiment'].dropna().unique() if exp.startswith('exp') and exp not in excluded_experiments])
y_labels = [e.replace('exp', '') for e in experiments]

# Prepare matrices for easier plotting (Absolute Scores)
seg_matrix = pd.DataFrame(index=experiments, columns=['ST', 'MT', 'MT-KD'])
disp_matrix = pd.DataFrame(index=experiments, columns=['ST', 'MT', 'MT-KD'])
seg_fb = pd.DataFrame(index=experiments, columns=['ST', 'MT', 'MT-KD'])
disp_fb = pd.DataFrame(index=experiments, columns=['ST', 'MT', 'MT-KD'])

# Extract baseline performance from exp01 for filling NaNs
baseline_data = grouped[grouped['experiment'] == 'exp01'].set_index('config')
b_st_seg = baseline_data.loc['SEG', 'seg_mean'] if 'SEG' in baseline_data.index else np.nan
b_st_disp = baseline_data.loc['DISP', 'disp_mean'] if 'DISP' in baseline_data.index else np.nan
b_mt_seg = baseline_data.loc['MT', 'seg_mean'] if 'MT' in baseline_data.index else np.nan
b_mt_disp = baseline_data.loc['MT', 'disp_mean'] if 'MT' in baseline_data.index else np.nan
b_mtkd_seg = baseline_data.loc['MT-KD', 'seg_mean'] if 'MT-KD' in baseline_data.index else np.nan
b_mtkd_disp = baseline_data.loc['MT-KD', 'disp_mean'] if 'MT-KD' in baseline_data.index else np.nan

for exp in experiments:
    exp_data = grouped[grouped['experiment'] == exp].set_index('config')
    
    # Track fallbacks
    seg_fb.loc[exp, 'ST'] = 'SEG' not in exp_data.index
    seg_fb.loc[exp, 'MT'] = 'MT' not in exp_data.index
    seg_fb.loc[exp, 'MT-KD'] = 'MT-KD' not in exp_data.index
    
    disp_fb.loc[exp, 'ST'] = 'DISP' not in exp_data.index
    disp_fb.loc[exp, 'MT'] = 'MT' not in exp_data.index
    disp_fb.loc[exp, 'MT-KD'] = 'MT-KD' not in exp_data.index

    # Populate with actual data or fallback to exp01 baseline
    seg_matrix.loc[exp, 'ST'] = exp_data.loc['SEG', 'seg_mean'] if 'SEG' in exp_data.index else b_st_seg
    seg_matrix.loc[exp, 'MT'] = exp_data.loc['MT', 'seg_mean'] if 'MT' in exp_data.index else b_mt_seg
    seg_matrix.loc[exp, 'MT-KD'] = exp_data.loc['MT-KD', 'seg_mean'] if 'MT-KD' in exp_data.index else b_mtkd_seg
    
    disp_matrix.loc[exp, 'ST'] = exp_data.loc['DISP', 'disp_mean'] if 'DISP' in exp_data.index else b_st_disp
    disp_matrix.loc[exp, 'MT'] = exp_data.loc['MT', 'disp_mean'] if 'MT' in exp_data.index else b_mt_disp
    disp_matrix.loc[exp, 'MT-KD'] = exp_data.loc['MT-KD', 'disp_mean'] if 'MT-KD' in exp_data.index else b_mtkd_disp

seg_matrix = seg_matrix.astype(float)
disp_matrix = disp_matrix.astype(float)

# Common styling
colors_dict = {'ST': px.colors.qualitative.Plotly[0], 'MT': px.colors.qualitative.Plotly[1], 'MT-KD': px.colors.qualitative.Plotly[2]}

# %% Table 1 - Performance Summary
# To build a nice pivot, we will format strings "Mean \pm Std"
grouped['seg_str'] = grouped.apply(lambda row: f"{row['seg_mean']:05.2f} ± {row['seg_std']:05.2f}" if pd.notna(row['seg_mean']) else "-", axis=1)
grouped['disp_str'] = grouped.apply(lambda row: f"{row['disp_mean']:05.2f} ± {row['disp_std']:05.2f}" if pd.notna(row['disp_mean']) else "-", axis=1)

# Melt and pivot into a single table with MultiIndex columns (Metric, config)
melted = grouped.melt(id_vars=['experiment', 'config'], value_vars=['seg_str', 'disp_str'], var_name='Metric', value_name='Value')
seg_label_tex = f"{seg_meta['short']} ({seg_meta['arrow']})"
disp_label_tex = f"{disp_meta['short']} ({disp_meta['arrow']})"
melted['Metric'] = melted['Metric'].replace({'seg_str': seg_label_tex, 'disp_str': disp_label_tex})

# Exclusion logic for LaTeX table: No SEG for DISP config, No DISP for SEG config
melted_table = melted.copy()
melted_table = melted_table[~((melted_table['Metric'] == seg_label_tex) & (melted_table['config'] == 'DISP'))]
melted_table = melted_table[~((melted_table['Metric'] == disp_label_tex) & (melted_table['config'] == 'SEG'))]

# Merge SEG and DISP into 'ST'
melted_table['config'] = melted_table['config'].replace({'SEG': 'ST', 'DISP': 'ST'})

# Vertical stacking: Pivot with MultiIndex index (Metric, experiment)
pivot_combined = melted_table.pivot(index=['Metric', 'experiment'], columns='config', values='Value')

# Extract experiment number (e.g., 'exp01' -> '1')
pivot_combined.index = pivot_combined.index.set_levels(
    pivot_combined.index.levels[1].str.extract(r'(\d+)')[0].values,
    level='experiment'
)

# Fill NaN with N/A
pivot_combined = pivot_combined.fillna('-')

# Order rows: SEG first, then DISP
metrics_order = [seg_label_tex, disp_label_tex]
pivot_combined = pivot_combined.reindex(metrics_order, level=0)

# Replace 'experiment' with 'ID' and remove 'config' column title
pivot_combined.index.names = ['Metric', 'ID']
pivot_combined.columns.name = None

# Order columns: ST, MT, MT-KD
desired_order = ['ST', 'MT', 'MT-KD']
ordered_cols = [c for c in desired_order if c in pivot_combined.columns]
remaining_cols = [c for c in pivot_combined.columns if c not in ordered_cols]
pivot_combined = pivot_combined[ordered_cols + remaining_cols]


# Print LaTeX with multirow for Metric and flattened header
# We use multicolumn_format='c' and index_names=True to get the header right
latex_output = pivot_combined.to_latex(
    escape=False, 
    index=True, 
    multirow=True, 
    index_names=True,
    column_format='ll' + 'c' * len(pivot_combined.columns)
)

print(latex_output)

# %% Figure 1 - Box Plots: Configuration Stability


target_exp = 'exp01'  # Change this to test other experiments
df_target = df_bench[df_bench['experiment'] == target_exp].copy()
df_baseline = df_bench[df_bench['experiment'] == 'exp01'].copy()

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

fig_box = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"))

for config in ['ST', 'MT', 'MT-KD']:
    fig_box.add_trace(go.Box(
        y=df_seg_box[df_seg_box['config'] == config][seg_metric],
        name=config,
        marker_color=colors_dict[config],
        boxpoints='all',   # Show all raw data points
        showlegend=False
    ), row=1, col=1)
    
    fig_box.add_trace(go.Box(
        y=df_disp_box[df_disp_box['config'] == config][disp_metric],
        name=config,
        marker_color=colors_dict[config],
        boxpoints='all',
        showlegend=False
    ), row=1, col=2)

fig_box.update_layout(
    template='plotly_white',
    height=400,
)
fig_box.update_yaxes(title_text=f"{seg_meta['label']} ({seg_meta['arrow']})", row=1, col=1)
fig_box.update_yaxes(title_text=f"{disp_meta['label']} ({disp_meta['arrow']})", row=1, col=2)
fig_box.update_xaxes(title_text="Experiment 01 Config", row=1, col=1)
fig_box.update_xaxes(title_text="Experiment 01 Config", row=1, col=2)

# Calculate equivalent spans to make metrics have the exact same visual scale (% per pixel)
seg_min, seg_max = df_seg_box[seg_metric].min(), df_seg_box[seg_metric].max()
disp_min, disp_max = df_disp_box[disp_metric].min(), df_disp_box[disp_metric].max()

# Find the maximum span between the two metrics and pad it slightly (e.g. 5%)
max_span = max(seg_max - seg_min, disp_max - disp_min) * 1.05

seg_center = (seg_max + seg_min) / 2
disp_center = (disp_max + disp_min) / 2

# Apply the strict ranges and reverse if arrow is down so "up" is always better
fig_box.update_yaxes(range=[seg_center - max_span/2, seg_center + max_span/2], row=1, col=1)
if disp_meta['arrow'] == '↓':
    fig_box.update_yaxes(range=[disp_center + max_span/2, disp_center - max_span/2], row=1, col=2)
else:
    fig_box.update_yaxes(range=[disp_center - max_span/2, disp_center + max_span/2], row=1, col=2)

save_figure(fig_box, name='H01_F01_boxplot_configuration_stability', lrtb_margin=(40, 0, 20, 0), folder='results', skip_sync=skip_sync)


# %% Figure 2 - Dumbbell Plot: Absolute Scores Overview
fig_dumb = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Segmentation", "Disparity"),
    horizontal_spacing=0.1
)

# For Segmentation (Col 1)
for i, exp in enumerate(experiments):
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
    fig_dumb.add_trace(go.Scatter(
        x=seg_matrix[config],
        y=y_labels,
        mode='markers',
        name=config,
        marker=dict(color=colors_dict[config], size=10),
        legendgroup=config
    ), row=1, col=1)

# For Disparity (Col 2)
for i, exp in enumerate(experiments):
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
    fig_dumb.add_trace(go.Scatter(
        x=disp_matrix[config],
        y=y_labels,
        mode='markers',
        name=config,
        marker=dict(color=colors_dict[config], size=10),
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
fig_dumb.update_yaxes(title_text="Experiment", row=1, col=1)
fig_dumb.update_yaxes(autorange="reversed", row=1, col=1)
fig_dumb.update_yaxes(autorange="reversed", row=1, col=2)

save_figure(fig_dumb, name='H01_F02_dumbbell_overview', lrtb_margin=(40, 20, 20, 60), folder='results', skip_sync=skip_sync)

# %% Figure 3 - Heatmap: Configuration Deltas relative to Single-Task (ST)

# Compute Deltas (Percentage points) relative to ST of the SAME experiment
seg_delta = seg_matrix.sub(seg_matrix['ST'], axis=0)
disp_delta = disp_matrix.sub(disp_matrix['ST'], axis=0)

fig_heat = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Segmentation", "Disparity"),
    horizontal_spacing=0.25
)

# Overlay setup: Translucent grey for fallback cells
seg_mask_z = seg_fb.astype(float).replace(0, np.nan)
disp_mask_z = disp_fb.astype(float).replace(0, np.nan)

# --- Segmentation (Col 1) ---
z_seg = seg_delta.values if seg_meta['arrow'] == '↑' else -seg_delta.values
fig_heat.add_trace(go.Heatmap(
    z=z_seg, x=seg_delta.columns, y=y_labels,
    colorscale='RdBu', zmid=0,
    text=np.round(seg_delta.values, 2), texttemplate="%{text}",
    showscale=True, xgap=2, ygap=2,
    colorbar=dict(x=0.38, title=f"Δ {seg_meta['short']}<br>(pp, {seg_meta['arrow']})", dtick=2)
), row=1, col=1)

fig_heat.add_trace(go.Heatmap(
    z=seg_mask_z.values, x=seg_fb.columns, y=y_labels,
    colorscale=[[0, 'rgba(200,200,200,0.6)'], [1, 'rgba(200,200,200,0.6)']],
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
    colorbar=dict(x=1.02, title=f"Δ {disp_meta['short']}<br>(pp, {disp_meta['arrow']})", tickmode='array', tickvals=tick_vals, ticktext=[(v if disp_meta['arrow'] == '↑' else -v) for v in tick_vals])
), row=1, col=2)

fig_heat.add_trace(go.Heatmap(
    z=disp_mask_z.values, x=disp_fb.columns, y=y_labels,
    colorscale=[[0, 'rgba(200,200,200,0.6)'], [1, 'rgba(200,200,200,0.6)']],
    showscale=False, hoverinfo='skip', xgap=2, ygap=2
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

save_figure(fig_heat, name='H01_F03_heatmap_configuration_deltas', lrtb_margin=(40, 20, 20, 20), folder='results', skip_sync=skip_sync)

# %%