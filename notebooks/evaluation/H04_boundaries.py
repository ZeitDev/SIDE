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

# %% Settings
# Settings
skip_sync = False

CHART_CONFIG = {
    'H04F01': {
        'x': dict(range=[20, 0], dtick=5),
        'y': dict(range=[90, 20], dtick=10)
    },
    'H04F02': {
        'x1': dict(range=[40, 0], dtick=10),
        'x2': dict(range=[105, 20], dtick=20),
    }
}

# %% Data preperation
# Data preperation

# Data preperation here

# %% H04F01_Scatter_BoundaryAttackTrajectories (Boundary Attack Trajectories for MT vs. MT-KD)
# H04F01_Scatter_BoundaryAttackTrajectories (Boundary Attack Trajectories for MT vs. MT-KD)

target_exps = ['exp01', 'exp05']#, 'exp02', 'exp03', 'exp04', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10']

df_bench = df_final[df_final['experiment'].isin(target_exps)].copy()

# Extract task metrics with fallbacks
metrics_info = {
    'AbsRel': {
        'combined': 'metric.best_combined/performance/testing/disparity/AbsRel_rate',
        'task': 'metric.best_disparity/performance/testing/disparity/AbsRel_rate'
    },
    'Bad3': {
        'combined': 'metric.best_combined/performance/testing/disparity/Bad3_rate',
        'task': 'metric.best_disparity/performance/testing/disparity/Bad3_rate'
    }
}

for m_name, cols in metrics_info.items():
    df_bench[m_name] = df_bench[cols['combined']].fillna(df_bench[cols['task']])

# Harmonize config naming
df_bench['config'] = df_bench['config'].replace({'DISP': 'ST'})
target_configs = ['ST', 'MT', 'MT-KD']
df_bench = df_bench[df_bench['config'].isin(target_configs)]

# Calculate medians for all experiments and configs
df_medians = df_bench.groupby(['experiment', 'config'])[['AbsRel', 'Bad3']].median().reset_index()

colors = {
    'ST': px.colors.qualitative.Plotly[0],
    'MT': px.colors.qualitative.Plotly[1],
    'MT-KD': px.colors.qualitative.Plotly[2]
}

exp_colors = {
    'exp01': 'rgba(100, 100, 100, 0.5)',
    'exp04': 'rgba(100, 100, 100, 0.5)',
    'exp05': 'rgba(100, 100, 100, 0.5)',
    'exp06': 'rgba(100, 100, 100, 0.5)'
}

fig = go.Figure()

for exp in target_exps:
    df_exp = df_medians[df_medians['experiment'] == exp].copy()
    
    # Sort configs to ST -> MT -> MT-KD
    df_exp['config'] = pd.Categorical(df_exp['config'], categories=target_configs, ordered=True)
    df_exp = df_exp.sort_values('config')
    
    # Add trajectory line
    fig.add_trace(go.Scatter(
        x=df_exp['AbsRel'],
        y=df_exp['Bad3'],
        mode='lines',
        line=dict(color=exp_colors.get(exp, 'rgba(100, 100, 100, 0.5)'), width=2, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

# Add markers for each configuration to group by config
for config in target_configs:
    df_config = df_medians[df_medians['config'] == config]
    fig.add_trace(go.Scatter(
        x=df_config['AbsRel'],
        y=df_config['Bad3'],
        mode='markers+text',
        marker=dict(
            color=colors[config],
            size=12,
            symbol='circle',
            line=dict(color='white', width=1)
        ),
        name=config,
        text=[e[-2:] for e in df_config['experiment']],
        textposition="top center",
        hoverinfo='text',
        hovertext=df_config['experiment'] + f" ({config})<br>AbsRel Rate: " + df_config['AbsRel'].round(2).astype(str) + "<br>Bad3 Rate: " + df_config['Bad3'].round(2).astype(str)
    ))

fig.update_layout(
    # template='plotly_white',
    height=500,
    width=600,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    legend_title_text="Config"
)

fig.update_xaxes(title_text="AbsRel Rate [% ↓]", autorange="reversed")
fig.update_yaxes(title_text="Bad3 Rate [% ↓]", autorange="reversed")

apply_chart_config(fig, 'H04F01', CHART_CONFIG)
save_figure(fig, height=400, name='H04F01', lrtb_margin=(40, 20, 0, 60), folder='results', skip_sync=skip_sync)

# %% H04F02_Barplot_GatingOnBoundary (Effect of Confidence-Based Gating on Performance)
# H04F02_Barplot_GatingOnBoundary (Effect of Confidence-Based Gating on Performance)

target_exps = ['exp01', 'exp02']
target_configs = ['MT', 'MT-KD', 'ST']

df_f02 = df_final[df_final['experiment'].isin(target_exps)].copy()

# Map SEG/DISP to ST
df_f02['config_mapped'] = df_f02['config'].replace({'SEG': 'ST', 'DISP': 'ST'})
df_f02['regime'] = df_f02['experiment'].map({'exp01': '01 (ON)', 'exp02': '02 (OFF)'})

# Metrics configuration
metrics = {
    'AbsRel': 'AbsRel_rate',
    'Bad3': 'Bad3_rate'
}

# Process metrics with fallback logic
for m_name, m_col in metrics.items():
    col = f"metric.best_combined/performance/testing/disparity/{m_col}"
    fallback = f"metric.best_disparity/performance/testing/disparity/{m_col}"
    df_f02[m_name] = df_f02[col].fillna(df_f02[fallback])

# Styling
colors_dict = {
    'ST': px.colors.qualitative.Plotly[0],
    'MT': px.colors.qualitative.Plotly[1],
    'MT-KD': px.colors.qualitative.Plotly[2]
}
regimes = ['01 (ON)', '02 (OFF)']
display_configs = ['ST', 'MT', 'MT-KD']

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05)

for col, task in enumerate(['AbsRel', 'Bad3'], start=1):
    for config in display_configs:
        for regime in regimes:
            data = df_f02[(df_f02['regime'] == regime) & (df_f02['config_mapped'] == config)][task]
            
            showlegend = True if col == 1 and regime == regimes[0] else False
            
            fig.add_trace(go.Box(
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

fig.update_layout(
    # template='plotly_white',
    height=450,
    width=850,
    boxmode='group',
    boxgroupgap=0.6,
    boxgap=0.3,
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title_text="Config")
)

fig.update_xaxes(title_text="AbsRel Rate [% ↓]", autorange="reversed", row=1, col=1)
fig.update_xaxes(title_text="Bad3 Rate [% ↓]", autorange="reversed", row=1, col=2)
fig.update_yaxes(title_text="Experiment (Disparity Gating)", autorange="reversed", row=1, col=1)
fig.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2)
apply_chart_config(fig, 'H04F02', CHART_CONFIG)
save_figure(fig, height=450, name='H04F02', lrtb_margin=(100, 20, 0, 0), standoff=10, folder='results', skip_sync=skip_sync)

# %%
