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

# %% Data preperation
# Data preperation

# Data preperation here

# %% F01_Orthogonality_Scatter
# F01 — The Boundary Attack Trajectories

target_exps = ['exp01', 'exp05']

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
    template='plotly_white',
    height=500,
    width=600,
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    legend_title_text="Config"
)

fig.update_xaxes(title_text="AbsRel Rate [% ↓]")
fig.update_yaxes(title_text="Bad3 Rate [% ↓]")

save_figure(fig, height=400, name='F01_Orthogonality_Scatter', lrtb_margin=(40, 20, 0, 60), folder='results', skip_sync=skip_sync)

# %% H04F02