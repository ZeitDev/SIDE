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

# %% Data preperation
# Data preperation

metrics = ['DICE_score', 'AbsRel_rate']

METRIC_META = {
    'DICE_score': {'label': 'DICE Score', 'short': 'DICE', 'arrow': '% ↑', 'suffix': '/instrument_mean', 'task': 'segmentation'},
    'AbsRel_rate': {'label': 'AbsRel Rate', 'short': 'AbsRel', 'arrow': '% ↓', 'suffix': '', 'task': 'disparity'}
}

df_bench = df_final.copy()
df_bench = df_bench[df_bench['experiment'].isin(['exp01', 'exp02', 'exp03', 'exp09'])]
df_bench = df_bench[df_bench['config'] == 'MT-KD']

for metric in metrics:
    meta = METRIC_META[metric]
    task = meta['task']
    col = f"metric.best_combined/performance/testing/{task}/{metric}{meta['suffix']}"
    fallback = f"metric.best_{task}/performance/testing/{task}/{metric}{meta['suffix']}"
    df_bench[metric] = df_bench[col].fillna(df_bench[fallback])
    
# Map experiment names for y-axis
exp_map = {
    'exp01': '01 (70%<br>Mean Confidence)', 
    'exp02': '02 (No Conf. Gate)',
    'exp03': '03 (No Conf. Scaling)',
    'exp09': '09 (Temperature = 4)'
}
df_bench['regime'] = df_bench['experiment'].map(exp_map)

# Common styling
colors_dict = {'MT-KD': px.colors.qualitative.Plotly[2]}

# %% H07F01_Boxplot_Entropy (Exp01 vs. 09)
# H07F01_Boxplot_Entropy (Exp01 vs. 09)

seg_meta = METRIC_META['DICE_score']
disp_meta = METRIC_META['AbsRel_rate']

fig_bar = make_subplots(rows=1, cols=2, subplot_titles=("Segmentation", "Disparity"), horizontal_spacing=0.05)

#regimes = ['01 (Base)', '02 (No Conf. Gate)', '03 (No Conf. Scaling)', '09 (Temperature = 4)']
regimes = ['01 (70%<br>Mean Confidence)', '09 (Temperature = 4)']
configs = ['MT-KD']

for config in configs:
    for col, metric in enumerate(['DICE_score', 'AbsRel_rate'], start=1):
        for regime in regimes:
            data = df_bench[(df_bench['regime'] == regime) & (df_bench['config'] == config)][metric]
            
            # Show legend only once per config
            showlegend = True if col == 1 and regime == regimes[0] else False
            
            fig_bar.add_trace(go.Box(
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

fig_bar.update_layout(
    template='plotly_white',
    height=400,
    width=850,
    boxmode='group',
    boxgroupgap=0.6,
    boxgap=0.3,
    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, title_text="Config")
)

fig_bar.update_xaxes(title_text=f"{seg_meta['label']} [{seg_meta['arrow']}]", row=1, col=1)
fig_bar.update_xaxes(
    title_text=f"{disp_meta['label']} [{disp_meta['arrow']}]", 
    autorange="reversed" if disp_meta['arrow'] == '% ↓' else None,
    row=1, col=2
)
fig_bar.update_yaxes(title_text="Experiment (Distillation Temperature)", autorange="reversed", row=1, col=1)
fig_bar.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=2)

save_figure(fig_bar, height=400, name='H07F01', lrtb_margin=(170, 10, 30, 0), folder='results', skip_sync=skip_sync)



# %%