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

# %%
skip_sync = False  # Set to False to enable Weights & Biases sync for this notebook's figures


# %% Prepare Data for Scatter Plots (Exp01 only)
# Metrics definitions
dice_col = 'metric.best_combined/performance/testing/segmentation/DICE_score/instrument_mean'
dice_fallback = 'metric.best_segmentation/performance/testing/segmentation/DICE_score/instrument_mean'

iou_col = 'metric.best_combined/performance/testing/segmentation/IoU_score/instrument_mean'
iou_fallback = 'metric.best_segmentation/performance/testing/segmentation/IoU_score/instrument_mean'

EPE_col = 'metric.best_combined/performance/testing/disparity/EPE_px'
EPE_fallback = 'metric.best_disparity/performance/testing/disparity/EPE_px'

bad3_col = 'metric.best_combined/performance/testing/disparity/Bad3_rate'
bad3_fallback = 'metric.best_disparity/performance/testing/disparity/Bad3_rate'

# Create a clean dataframe for Exp01
df_exp01 = df_final[df_final['experiment'] == 'exp01'].copy()
df_exp01['DICE_score'] = df_exp01[dice_col].fillna(df_exp01[dice_fallback])
df_exp01['IoU_score'] = df_exp01[iou_col].fillna(df_exp01[iou_fallback])
df_exp01['EPE_px'] = df_exp01[EPE_col].fillna(df_exp01[EPE_fallback])
df_exp01['Bad3_rate'] = df_exp01[bad3_col].fillna(df_exp01[bad3_fallback])

# Normalize config naming to ST, MT, MT-KD
df_exp01['config_seg'] = df_exp01['config'].replace({'SEG': 'ST'})
df_exp01['config_disp'] = df_exp01['config'].replace({'DISP': 'ST'})

# Filter datasets for Segmentation (ST vs MT vs MT-KD) and Disparity (ST vs MT vs MT-KD)
df_seg = df_exp01[df_exp01['config_seg'].isin(['ST', 'MT', 'MT-KD'])].dropna(subset=['DICE_score', 'IoU_score'])
df_disp = df_exp01[df_exp01['config_disp'].isin(['ST', 'MT', 'MT-KD'])].dropna(subset=['EPE_px', 'Bad3_rate'])

# Define colors
colors_dict = {'ST': px.colors.qualitative.Plotly[0], 'MT': px.colors.qualitative.Plotly[1], 'MT-KD': px.colors.qualitative.Plotly[2]}

# %% Figure 1: EPE vs DICE Scatter Plot
# Merge params to get the seed
df_exp01_with_seeds = pd.merge(df_exp01, df_params[['identifier', 'param.general.seed']], on='identifier', how='left')

# For ST, we pair SEG and DISP models by seed to get both metrics
st_seg = df_exp01_with_seeds[df_exp01_with_seeds['config'] == 'SEG'][['param.general.seed', 'DICE_score']].set_index('param.general.seed')
st_disp = df_exp01_with_seeds[df_exp01_with_seeds['config'] == 'DISP'][['param.general.seed', 'EPE_px']].set_index('param.general.seed')
st_combined = st_seg.join(st_disp, how='inner').reset_index()
st_combined['config_label'] = 'ST'

# For MT and MT-KD, metrics belong to the same run
mt_data = df_exp01_with_seeds[df_exp01_with_seeds['config'] == 'MT'][['param.general.seed', 'DICE_score', 'EPE_px']].copy()
mt_data['config_label'] = 'MT'

mtkd_data = df_exp01_with_seeds[df_exp01_with_seeds['config'] == 'MT-KD'][['param.general.seed', 'DICE_score', 'EPE_px']].copy()
mtkd_data['config_label'] = 'MT-KD'

df_combined_cross = pd.concat([st_combined, mt_data, mtkd_data]).dropna(subset=['DICE_score', 'EPE_px'])

fig_scatter2 = go.Figure()
for config in ['ST', 'MT', 'MT-KD']:
    subset = df_combined_cross[df_combined_cross['config_label'] == config]
    fig_scatter2.add_trace(go.Scatter(
        x=subset['EPE_px'],
        y=subset['DICE_score'],
        mode='markers',
        name=config,
        marker=dict(color=colors_dict[config], size=10, opacity=0.8)
    ))

fig_scatter2.update_layout(
    # template='plotly_white',
    height=450,
    width=550,
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title="Experiment 01 Config")
)

fig_scatter2.update_xaxes(title_text="EPE (↓)", autorange="reversed")
fig_scatter2.update_yaxes(title_text="DICE Score (↑)")

save_figure(fig_scatter2, name='H02_F01_scatter_task_stability', lrtb_margin=(40, 20, 20, 60), folder='results', skip_sync=skip_sync)

# %% Figure 2: The Trade-off Scatter Plots
fig_scatter = go.Figure()

# Plot 1: EPE vs Bad3 (Disparity)
for config in ['ST', 'MT', 'MT-KD']:
    subset = df_disp[df_disp['config_disp'] == config]
    fig_scatter.add_trace(go.Scatter(
        x=subset['EPE_px'],
        y=subset['Bad3_rate'],
        mode='markers',
        name=config,
        marker=dict(color=colors_dict[config], size=10, opacity=0.8)
    ))

fig_scatter.update_layout(
    # template='plotly_white',
    height=450,
    width=550,
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title="Experiment 01 Config")
)

# Set axes titles and ranges
fig_scatter.update_xaxes(title_text="EPE (↓)", autorange="reversed") # Smaller EPE is better
fig_scatter.update_yaxes(title_text="Bad3 Rate (↓)", autorange="reversed")  # Smaller Bad3 is better

save_figure(fig_scatter, name='H02_F02_scatter_globalVSlocal_stability', lrtb_margin=(40, 20, 20, 60), folder='results', skip_sync=skip_sync)




# %%