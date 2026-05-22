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

# %% Prepare Data for Figures and Tables
# From the dictionary, we identify the key performance metrics (Tier 2 Focus)
dice_col = 'metric.best_combined/performance/testing/segmentation/DICE_score/instrument_mean'
absrel_col = 'metric.best_combined/performance/testing/disparity/AbsRel_rate'

# If combined metrics aren't available for standalone, we might need to fallback to their specific ones.
# We will coalesce the metrics if they are NaN.
dice_fallback = 'metric.best_segmentation/performance/testing/segmentation/DICE_score/instrument_mean'
absrel_fallback = 'metric.best_disparity/performance/testing/disparity/AbsRel_rate'

df_bench = df_final.copy()
df_bench['DICE_score'] = df_bench[dice_col].fillna(df_bench[dice_fallback])
df_bench['AbsRel_rate'] = df_bench[absrel_col].fillna(df_bench[absrel_fallback])

# Group by experiment and config to get mean and std across seeds of config per experiment
grouped = df_bench.groupby(['experiment', 'config']).agg(
    DICE_mean=('DICE_score', 'mean'),
    DICE_std=('DICE_score', 'std'),
    AbsRel_mean=('AbsRel_rate', 'mean'),
    AbsRel_std=('AbsRel_rate', 'std')
).reset_index()

# %% Figure 1 - Box Plots: Configuration Stability
from plotly.subplots import make_subplots

target_exp = 'exp01'  # Change this to test other experiments
df_target = df_bench[df_bench['experiment'] == target_exp].copy()
df_baseline = df_bench[df_bench['experiment'] == 'exp01'].copy()

# DICE data for target_exp (falling back to exp01 if configs are missing)
df_dice = df_target[df_target['config'].isin(['SEG', 'MT', 'MT-KD'])].copy()
missing_dice = [c for c in ['SEG', 'MT', 'MT-KD'] if c not in df_dice['config'].unique()]
if missing_dice:
    df_dice = pd.concat([df_dice, df_baseline[df_baseline['config'].isin(missing_dice)]])

df_dice['config'] = df_dice['config'].replace({'SEG': 'ST'})
df_dice['config'] = pd.Categorical(df_dice['config'], categories=['ST', 'MT', 'MT-KD'], ordered=True)
df_dice = df_dice.sort_values('config')

# AbsRel data for target_exp (falling back to exp01 if configs are missing)
df_absrel = df_target[df_target['config'].isin(['DISP', 'MT', 'MT-KD'])].copy()
missing_absrel = [c for c in ['DISP', 'MT', 'MT-KD'] if c not in df_absrel['config'].unique()]
if missing_absrel:
    df_absrel = pd.concat([df_absrel, df_baseline[df_baseline['config'].isin(missing_absrel)]])

df_absrel['config'] = df_absrel['config'].replace({'DISP': 'ST'})
df_absrel['config'] = pd.Categorical(df_absrel['config'], categories=['ST', 'MT', 'MT-KD'], ordered=True)
df_absrel = df_absrel.sort_values('config')

fig_box = make_subplots(rows=1, cols=2, subplot_titles=("DICE Score", "AbsRel Rate"))
colors_dict = {'ST': px.colors.qualitative.Plotly[0], 'MT': px.colors.qualitative.Plotly[1], 'MT-KD': px.colors.qualitative.Plotly[2]}

for config in ['ST', 'MT', 'MT-KD']:
    fig_box.add_trace(go.Box(
        y=df_dice[df_dice['config'] == config]['DICE_score'],
        name=config,
        marker_color=colors_dict[config],
        boxpoints='all',   # Show all raw data points
        #jitter=0.3,        # Spread them horizontally to prevent overlap
        #pointpos=-1.8,     # Offset them slightly to the left of the box
        showlegend=False
    ), row=1, col=1)
    
    fig_box.add_trace(go.Box(
        y=df_absrel[df_absrel['config'] == config]['AbsRel_rate'],
        name=config,
        marker_color=colors_dict[config],
        boxpoints='all',
        #jitter=0.3,
        #pointpos=-1.8,
        showlegend=False
    ), row=1, col=2)

fig_box.update_layout(
    template='plotly_white',
    height=400,
)

# Calculate equivalent spans to make DICE and AbsRel have the exact same visual scale (% per pixel)
dice_min, dice_max = df_dice['DICE_score'].min(), df_dice['DICE_score'].max()
absrel_min, absrel_max = df_absrel['AbsRel_rate'].min(), df_absrel['AbsRel_rate'].max()

# Find the maximum span between the two metrics and pad it slightly (e.g. 10%)
max_span = max(dice_max - dice_min, absrel_max - absrel_min) * 1.05

dice_center = (dice_max + dice_min) / 2
absrel_center = (absrel_max + absrel_min) / 2

# Apply the strict ranges and reverse AbsRel so "up" is always better
fig_box.update_yaxes(range=[dice_center - max_span/2, dice_center + max_span/2], row=1, col=1)
fig_box.update_yaxes(range=[absrel_center + max_span/2, absrel_center - max_span/2], row=1, col=2)

save_figure(fig_box, name='H01_configuration_stability', lrtb_margin=(0, 0, 20, 0), folder='results')

# %% Figure 2 - Slopegraph: Configuration Performance over Experiments
from plotly.subplots import make_subplots

# Get baseline values from 'exp01' for fallback if missing
baseline_exp_data = grouped[grouped['experiment'] == 'exp01'].set_index('config') if 'exp01' in grouped['experiment'].values else None
baseline_seg_dice = baseline_exp_data.get('DICE_mean', {}).get('SEG', np.nan) if baseline_exp_data is not None else np.nan
baseline_disp_absrel = baseline_exp_data.get('AbsRel_mean', {}).get('DISP', np.nan) if baseline_exp_data is not None else np.nan
baseline_mt_dice = baseline_exp_data.get('DICE_mean', {}).get('MT', np.nan) if baseline_exp_data is not None else np.nan
baseline_mt_absrel = baseline_exp_data.get('AbsRel_mean', {}).get('MT', np.nan) if baseline_exp_data is not None else np.nan

fig1_data = []
for exp in grouped['experiment'].unique():
    exp_data = grouped[grouped['experiment'] == exp].set_index('config')
    
    # DICE
    seg_dice = exp_data.get('DICE_mean', {}).get('SEG', np.nan)
    mt_dice = exp_data.get('DICE_mean', {}).get('MT', np.nan)
    mtkd_dice = exp_data.get('DICE_mean', {}).get('MT-KD', np.nan)
    
    # AbsRel
    disp_absrel = exp_data.get('AbsRel_mean', {}).get('DISP', np.nan)
    mt_absrel = exp_data.get('AbsRel_mean', {}).get('MT', np.nan)
    mtkd_absrel = exp_data.get('AbsRel_mean', {}).get('MT-KD', np.nan)

    # Dynamic fallback to exp01 if ST data is missing
    if pd.isna(seg_dice):
        seg_dice = baseline_seg_dice
    if pd.isna(disp_absrel):
        disp_absrel = baseline_disp_absrel

    # Dynamic fallback to exp01 if MT data is missing
    if pd.isna(mt_dice):
        mt_dice = baseline_mt_dice
    if pd.isna(mt_absrel):
        mt_absrel = baseline_mt_absrel

    fig1_data.append({
        'experiment': exp,
        'ST_DICE': seg_dice, 'MT_DICE': mt_dice, 'MT-KD_DICE': mtkd_dice,
        'ST_AbsRel': disp_absrel, 'MT_AbsRel': mt_absrel, 'MT-KD_AbsRel': mtkd_absrel
    })

df_fig1 = pd.DataFrame(fig1_data)

# To achieve a "broken axis" look for the DICE Score, we split the first column into two rows.
# The second column (AbsRel Rate) spans both rows.
fig_config = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"type": "xy"}, {"type": "xy", "rowspan": 2}],
        [{"type": "xy"}, None]
    ],
    subplot_titles=("DICE Score", "AbsRel Rate", "--- ✂ ---"),
    vertical_spacing=0.03,
    shared_xaxes=True,
    row_heights=[2/18, 16/18]
)

x_cats = ['ST', 'MT', 'MT-KD']
colors = px.colors.qualitative.Plotly

for i, exp in enumerate(df_fig1['experiment']):
    row = df_fig1[df_fig1['experiment'] == exp].iloc[0]
    color = colors[i % len(colors)]
    
    display_name = exp.replace('exp', '')
    
    # DICE - Add the same trace to both the top and bottom subplots
    y_dice = [row['ST_DICE'], row['MT_DICE'], row['MT-KD_DICE']]
    fig_config.add_trace(go.Scatter(x=x_cats, y=y_dice, mode='lines+markers',
                                    name=display_name, legendgroup=exp, marker_color=color),
                         row=1, col=1)
    fig_config.add_trace(go.Scatter(x=x_cats, y=y_dice, mode='lines+markers',
                                    name=display_name, legendgroup=exp, showlegend=False, marker_color=color),
                         row=2, col=1)
    
    # AbsRel
    y_absrel = [row['ST_AbsRel'], row['MT_AbsRel'], row['MT-KD_AbsRel']]
    fig_config.add_trace(go.Scatter(x=x_cats, y=y_absrel, mode='lines+markers',
                                    name=display_name, legendgroup=exp, showlegend=False, marker_color=color),
                         row=1, col=2)

fig_config.update_layout(
    template='plotly_white',
    height=600,
    legend_title_text='Experiment'
)

# Remove horizontal grids to keep slopes clear, reverse AbsRel axis
fig_config.update_yaxes(showgrid=False, row=1, col=1)
fig_config.update_yaxes(showgrid=False, row=2, col=1)
fig_config.update_yaxes(showgrid=False, autorange="reversed", row=1, col=2)

# Set DICE top range to [95, 97] (size 2) and bottom to [39, 55] (size 16) to keep the original values intact.
# The row_heights=[2/18, 16/18] mathematically locks the spacing to perfectly fair scaling between both plots.
fig_config.update_yaxes(range=[95, 97], tickvals=[95, 97], row=1, col=1) # Length = 2
fig_config.update_yaxes(range=[39, 55], row=2, col=1) # Length = 16

# Hide the x-axis tick labels for the top part of the broken axis
fig_config.update_xaxes(showticklabels=False, row=1, col=1)


save_figure(fig_config, name='H01_configuration_trajectory', lrtb_margin=(0, 0, 20, 0), folder='results')
    
# %% Table 1 - Performance Summary
# %% TABLE 1: Global Results Matrix (LaTeX Table)
# To build a nice pivot, we will format strings "Mean \pm Std"
grouped['DICE_str'] = grouped.apply(lambda row: f"{row['DICE_mean']:05.2f} ± {row['DICE_std']:05.2f}" if pd.notna(row['DICE_mean']) else "-", axis=1)
grouped['AbsRel_str'] = grouped.apply(lambda row: f"{row['AbsRel_mean']:05.2f} ± {row['AbsRel_std']:05.2f}" if pd.notna(row['AbsRel_mean']) else "-", axis=1)

# Melt and pivot into a single table with MultiIndex columns (Metric, config)
melted = grouped.melt(id_vars=['experiment', 'config'], value_vars=['DICE_str', 'AbsRel_str'], var_name='Metric', value_name='Value')
melted['Metric'] = melted['Metric'].replace({'DICE_str': r'DICE ($\uparrow$)', 'AbsRel_str': r'AbsRel ($\downarrow$)'})

# Exclusion logic for LaTeX table: No DICE for DISP, No AbsRel for SEG
melted_table = melted.copy()
melted_table = melted_table[~((melted_table['Metric'] == r'DICE ($\uparrow$)') & (melted_table['config'] == 'DISP'))]
melted_table = melted_table[~((melted_table['Metric'] == r'AbsRel ($\downarrow$)') & (melted_table['config'] == 'SEG'))]

# Merge SEG and DISP into 'Single-Task'
melted_table['config'] = melted_table['config'].replace({'SEG': 'Single-Task', 'DISP': 'Single-Task'})

# Vertical stacking: Pivot with MultiIndex index (Metric, experiment)
pivot_combined = melted_table.pivot(index=['Metric', 'experiment'], columns='config', values='Value')

# Extract experiment number (e.g., 'exp01' -> '1')
pivot_combined.index = pivot_combined.index.set_levels(
    pivot_combined.index.levels[1].str.extract(r'(\d+)')[0].values,
    level='experiment'
)

# Fill NaN with N/A
pivot_combined = pivot_combined.fillna('-')

# Order rows: DICE first, then AbsRel
metrics_order = [r'DICE ($\uparrow$)', r'AbsRel ($\downarrow$)']
pivot_combined = pivot_combined.reindex(metrics_order, level=0)

# Replace 'experiment' with 'ID' and remove 'config' column title
pivot_combined.index.names = ['Metric', 'ID']
pivot_combined.columns.name = None

# Order columns: Single-Task, MT, MT-KD
desired_order = ['Single-Task', 'MT', 'MT-KD']
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

# %%