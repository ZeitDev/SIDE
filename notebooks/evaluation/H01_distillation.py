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
fallback_opacity = 0.5

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

# %% H01T01_Table_GlobalResultsMatrix (DICE, AbsRel, Bad3)
# H01T01_Table_GlobalResultsMatrix (DICE, AbsRel, Bad3)

# Format strings as stacked deltas: Median_{-low}^{+high}
for metric in metrics:
    grouped[f'{metric}_str'] = grouped.apply(
        lambda row: (
            f"${row[f'{metric}_median']:05.2f}_{{-{row[f'{metric}_median'] - row[f'{metric}_min']:05.2f}}}^{{+{row[f'{metric}_max'] - row[f'{metric}_median']:05.2f}}}$"
            if pd.notna(row[f'{metric}_median']) else "-"
        ), 
        axis=1
    )

# Melt and pivot
value_vars = [f'{metric}_str' for metric in metrics]
melted = grouped.melt(id_vars=['experiment', 'config'], value_vars=value_vars, var_name='Metric', value_name='Value')

# Map metric strings to clean LaTeX labels
metric_map = {f'{metric}_str': f"{METRIC_META[metric]['short']} [{METRIC_META[metric]['arrow'].replace('%', '\\%')}]" for metric in metrics}
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

print(f"\\renewcommand{{\\arraystretch}}{{1.4}}\n{latex_output}")

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
        showlegend=True,
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

# Manual ranges for the stability boxplots (H01F01)
# Use [None, None] for automatic scaling
F01_SEG_RANGE = [25, 65]
F01_DISP_RANGE = [0, 35]

fig_bar.update_layout(
    template='plotly_white',
    height=400,
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title_text="Config")
)
fig_bar.update_xaxes(
    title_text=f"{seg_meta['label']} [{seg_meta['arrow']}]", 
    range=F01_SEG_RANGE if all(v is not None for v in F01_SEG_RANGE) else None,
    row=1, col=1
)
fig_bar.update_xaxes(
    title_text=f"{disp_meta['label']} [{disp_meta['arrow']}]", 
    range=F01_DISP_RANGE[::-1] if all(v is not None for v in F01_DISP_RANGE) and '↓' in disp_meta['arrow'] else (F01_DISP_RANGE if all(v is not None for v in F01_DISP_RANGE) else None),
    autorange="reversed" if '↓' in disp_meta['arrow'] and all(v is None for v in F01_DISP_RANGE) else None,
    row=1, col=2
)
fig_bar.update_yaxes(title_text="Experiment 01 Config", autorange="reversed", row=1, col=1)
fig_bar.update_yaxes(showticklabels=False, title_text="", autorange="reversed", rangemode='tozero', row=1, col=2) # Hide tick labels and title

save_figure(fig_bar, name='H01F01_Boxplot_ConfigStability', lrtb_margin=(40, 40, 20, 40), folder='results', skip_sync=skip_sync)

# %% H01F02_Boxplot_StabilityOverview (ST vs. MT vs. MT-KD across all Experiments)
# H01F02_Boxplot_StabilityOverview (ST vs. MT vs. MT-KD across all Experiments)

df_fig2 = df_bench[df_bench['experiment'].isin(experiments)].copy()

fig_box = make_subplots(
    rows=2, cols=3, 
    subplot_titles=("", "", ""), 
    shared_yaxes=True,
    shared_xaxes=True,
    vertical_spacing=0.025,
    horizontal_spacing=0.01
)
grey_color = '#c0c0c0'

for i, config_alias in enumerate(['ST', 'MT', 'MT-KD']):
    for exp_idx, exp in enumerate(experiments):
        # Determine data and fallback for Segmentation (Row 1)
        target_seg_config = 'SEG' if config_alias == 'ST' else config_alias
        df_seg = df_fig2[(df_fig2['config'] == target_seg_config) & (df_fig2['experiment'] == exp)]
        is_seg_fb = df_seg.empty
        if is_seg_fb:
            df_seg = df_baseline[df_baseline['config'] == target_seg_config]
            
        # Determine data and fallback for Disparity (Row 2)
        target_disp_config = 'DISP' if config_alias == 'ST' else config_alias
        df_disp = df_fig2[(df_fig2['config'] == target_disp_config) & (df_fig2['experiment'] == exp)]
        is_disp_fb = df_disp.empty
        if is_disp_fb:
            df_disp = df_baseline[df_baseline['config'] == target_disp_config]

        fig_box.add_trace(go.Box(
            y=df_seg[seg_metric],
            x=[exp.replace('exp', '')] * len(df_seg),
            name=config_alias,
            marker_color=grey_color if is_seg_fb else colors_dict[config_alias],
            marker_opacity=0.6 if is_seg_fb else 1.0,
            showlegend=True if exp_idx == 0 else False,
            legendgroup=config_alias
        ), row=1, col=i+1)
        
        fig_box.add_trace(go.Box(
            y=df_disp[disp_metric],
            x=[exp.replace('exp', '')] * len(df_disp),
            name=config_alias,
            marker_color=grey_color if is_disp_fb else colors_dict[config_alias],
            marker_opacity=0.6 if is_disp_fb else 1.0,
            showlegend=False,
            legendgroup=config_alias
        ), row=2, col=i+1)

fig_box.update_layout(
    template='plotly_white',
    height=600,
    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5, title_text="Config"),
    margin=dict(t=20, b=40, l=60, r=20)
)

# Axis titles and styling
fig_box.update_yaxes(title_text=f"{seg_meta['label']} [{seg_meta['arrow']}]", dtick=5, row=1, col=1)
fig_box.update_yaxes(title_text=f"{disp_meta['label']} [{disp_meta['arrow']}]", autorange="reversed" if '↓' in disp_meta['arrow'] else None, dtick=5, row=2, col=1)

# Apply shared settings to all subplots
for col in range(1, 4):
    fig_box.update_yaxes(dtick=5, row=1, col=col)
    fig_box.update_yaxes(autorange="reversed" if '↓' in disp_meta['arrow'] else None, dtick=5, row=2, col=col)
    
    fig_box.update_xaxes(title_text="Experiment", row=2, col=col, tickmode='linear', dtick=1)
    fig_box.update_xaxes(tickmode='linear', dtick=1, row=1, col=col)

save_figure(fig_box, height=600, name='H01F02_Boxplot_StabilityOverview', lrtb_margin=(40, 0, 0, 40), folder='results', skip_sync=skip_sync)

# %% H01F03_Dumbbell_ConfigComparisonOverview (ST vs. MT vs. MT-KD)
# H01F03_Dumbbell_ConfigComparisonOverview (ST vs. MT vs. MT-KD)

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

fig_dumb.update_xaxes(title_text=f"{seg_meta['label']} [{seg_meta['arrow']}]", row=1, col=1)
fig_dumb.update_xaxes(title_text=f"{disp_meta['label']} [{disp_meta['arrow']}]", autorange="reversed" if '↓' in disp_meta['arrow'] else None, row=1, col=2)
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

save_figure(fig_dumb, name='H01F03_Dumbbell_ConfigComparisonOverview', lrtb_margin=(40, 20, 20, 60), folder='results', skip_sync=skip_sync)

# %% H01F04_Heatmap_ConfigComparisonOverview (ST vs. MT vs. MT-KD Delta)
# H01F04_Heatmap_ConfigComparisonOverview (ST vs. MT vs. MT-KD Delta)

# Compute Deltas (Percentage points) relative to ST of the SAME experiment
seg_delta = seg_matrix.sub(seg_matrix['ST'], axis=0)
disp_delta = disp_matrix.sub(disp_matrix['ST'], axis=0)

fig_heat = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Segmentation", "Disparity"),
    horizontal_spacing=0.2
)

# Helper to format text with * for fallbacks
def format_heatmap_text(delta_df, fallback_df):
    return [
        [f"{val:.2f}*" if is_fb else f"{val:.2f}<span style='color:rgba(0,0,0,0)'>*</span>" for val, is_fb in zip(d_row, f_row)]
        for d_row, f_row in zip(delta_df.values, fallback_df.values)
    ]

# Text arrays
seg_text = format_heatmap_text(seg_delta, seg_is_fallback)
disp_text = format_heatmap_text(disp_delta, disp_is_fallback)

# --- Segmentation (Col 1) ---
z_seg = seg_delta.values if '↑' in seg_meta['arrow'] else -seg_delta.values
fig_heat.add_trace(go.Heatmap(
    z=z_seg, x=seg_delta.columns, y=y_labels,
    colorscale='RdBu', zmid=0,
    text=seg_text, texttemplate="%{text}", textfont=dict(size=13),
    showscale=True, xgap=2, ygap=2,
    colorbar=dict(x=0.4, title=f"Δ {seg_meta['short']}<br>[pp {seg_meta['arrow']}]", dtick=2)
), row=1, col=1)

# --- Disparity (Col 2) ---
z_disp = disp_delta.values if '↑' in disp_meta['arrow'] else -disp_delta.values
disp_max_bound = int(np.ceil(np.nanmax(np.abs(z_disp)) / 2.0) * 2)
disp_max_bound = max(disp_max_bound, 2)
tick_vals = list(range(-disp_max_bound, disp_max_bound + 1, 2))

fig_heat.add_trace(go.Heatmap(
    z=z_disp, x=disp_delta.columns, y=y_labels,
    colorscale='RdBu', zmin=-disp_max_bound, zmax=disp_max_bound,
    text=disp_text, texttemplate="%{text}", textfont=dict(size=13),
    showscale=True, xgap=2, ygap=2,
    colorbar=dict(x=1.0, title=f"Δ {disp_meta['short']}<br>[pp {disp_meta['arrow']}]", tickmode='array', tickvals=tick_vals, ticktext=[(v if '↑' in disp_meta['arrow'] else -v) for v in tick_vals])
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

save_figure(fig_heat, name='H01F04_Heatmap_ConfigComparisonOverview', lrtb_margin=(40, 20, 20, 20), folder='results', skip_sync=skip_sync)

# %% H01F05_Heatmap_AblationComparisonOverview (Exp01 vs. ...)
# H01F05_Heatmap_AblationComparisonOverview (Exp01 vs. ...)

# Compute Deltas (Percentage points) relative to Baseline (exp01)
seg_delta_abl = seg_matrix - seg_matrix.loc['exp01']
disp_delta_abl = disp_matrix - disp_matrix.loc['exp01']

fig_heat_abl = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Segmentation", "Disparity"),
    horizontal_spacing=0.2
)

# Text arrays
seg_text_abl = format_heatmap_text(seg_delta_abl, seg_is_fallback)
disp_text_abl = format_heatmap_text(disp_delta_abl, disp_is_fallback)

# --- Segmentation (Col 1) ---
z_seg_abl = seg_delta_abl.values if '↑' in seg_meta['arrow'] else -seg_delta_abl.values
fig_heat_abl.add_trace(go.Heatmap(
    z=z_seg_abl, x=seg_delta_abl.columns, y=y_labels,
    colorscale='RdBu', zmid=0,
    text=seg_text_abl, texttemplate="%{text}", textfont=dict(size=13),
    showscale=True, xgap=2, ygap=2,
    colorbar=dict(x=0.4, title=f"Δ {seg_meta['short']}<br>[pp {seg_meta['arrow']}]", dtick=2)
), row=1, col=1)

# --- Disparity (Col 2) ---
z_disp_abl = disp_delta_abl.values if '↑' in disp_meta['arrow'] else -disp_delta_abl.values
disp_max_bound_abl = int(np.ceil(np.nanmax(np.abs(z_disp_abl)) / 2.0) * 2)
disp_max_bound_abl = max(disp_max_bound_abl, 2)
tick_vals_abl = list(range(-disp_max_bound_abl, disp_max_bound_abl + 1, 2))

fig_heat_abl.add_trace(go.Heatmap(
    z=z_disp_abl, x=disp_delta_abl.columns, y=y_labels,
    colorscale='RdBu', zmin=-disp_max_bound_abl, zmax=disp_max_bound_abl,
    text=disp_text_abl, texttemplate="%{text}", textfont=dict(size=13),
    showscale=True, xgap=2, ygap=2,
    colorbar=dict(x=1.0, title=f"Δ {disp_meta['short']}<br>[pp {disp_meta['arrow']}]", tickmode='array', tickvals=tick_vals_abl, ticktext=[(v if '↑' in disp_meta['arrow'] else -v) for v in tick_vals_abl])
), row=1, col=2)

fig_heat_abl.update_layout(
    template='plotly_white',
    height=450,
    width=850
)

# Axis titles
fig_heat_abl.update_xaxes(title_text="Config")
fig_heat_abl.update_yaxes(title_text="Experiment", autorange="reversed", row=1, col=1)

# Reverse y-axis to put Exp 01 at the top and hide tick labels on right plot
fig_heat_abl.update_yaxes(showticklabels=False, autorange="reversed", row=1, col=2)

save_figure(fig_heat_abl, name='H01F05_Heatmap_AblationComparisonOverview', lrtb_margin=(40, 20, 20, 20), folder='results', skip_sync=skip_sync)

# %%