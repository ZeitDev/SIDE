# %%
# * Show FPS, VRAM, PARAMs for relevant experiments
# * Exp01 - Baseline
# * Exp04 - ConvNeXt Nano
# * Exp05 - Less Input Resolution
# * Teachers

# 01/04 Teacher
# NVIDIA SF (SEG) = 7.14 FPS // 140.06 ms // 2114.68 MB // 56.89 DICE (01, 04) // 53.01 DICE (05 with 33.3 FPS / 0.76 GB)
# NVIDIA FS (DISP) = 0.49 FPS // 2040.82 ms // 7126.30 MB in half precision // 2.14 AbsRel (01, 04) // 2.09 AbsRel (05 with 2 FPS / 2.85 GB)
# Sequence Teacher = 0.46 FPS // 2180.88 ms

# 01 Student
# SEG = 20.08 FPS // 49.80 ms // 1389.72 MB // 43.93 DICE
# DISP = 13.28 FPS // 75.30 ms // 1435.92 MB // 5.69 AbsRel
# Sequence Single Task = 7.99 FPS // 125.10 ms
# MT (convnext/wMT/260406:2037/train) = 10.52 FPS // 95.02 ms // 1.44 GB // 40.32 DICE // 12.26 AbsRel
# MT-KD (convnext/wMT-KD/260406:2036/train) = 10.50 FPS // 95.23 ms // 1.44 GB // 51.18 DICE // 7.91 AbsRel

# 04 Student
# SEG = 32.25 FPS // 30.91 ms // 119.28 MB // 1.09 GB // 42.46 DICE
# DISP = 22.88 FPS // 43.72 ms // 1157.52 MB // 1.13 GB // 6.06 AbsRel
# Sequence Single Task = 13.38 FPS // 74.74 ms 
# MT = 16.47 FPS // 60.73 ms // 1210.77 MB // 1.18 GB // 42.16 DICE // 8.25 AbsRel
# MT-KD = 16.47 FPS // 60.70 ms // 1210.77 MB // 1.18 GB // 55.06 DICE // 5.84 AbsRel

# 05 Student
# SEG = 75.46 FPS // 13.25 ms // 456.5 MB // 0.45 GB // 39.07 DICE 
# DISP = 49.24 FPS // 20.31 ms // 486.96 MB // 0.46 GB // 10.97 AbsRel
# Sequence Single Task = 29.8 FPS // 33.56 ms
# MT = 38.06 FPS // 26.27 ms // 506.41 MB // 0.49 GB // 44.36 DICE // 16.68 AbsRel
# MT-KD = 38.56 FPS // 25.94 ms // 505.88 // 0.49 GB // 49.00 DICE // 11.57 AbsRel

# %% H09F01 - Deployment Metrics
# H09F01_Barplot_DeploymentMetrics (DICE, AbsRel, FPS, VRAM)

import os, sys
from pathlib import Path
root_path = Path.cwd()
while root_path.parent != root_path and not (root_path / 'pyproject.toml').exists(): root_path = root_path.parent
os.chdir(root_path)
if str(root_path) not in sys.path: sys.path.append(str(root_path))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from notebooks.figures.helpers import save_figure, apply_chart_config

skip_sync = False

CHART_CONFIG = {
    'H09F01': {
        'x1': dict(range=[35, 60], dtick=5),
        'x2': dict(range=[18, 0], dtick=6),
        'x3': dict(range=[0, 80], dtick=20),
        'x4': dict(range=[8, 0], dtick=2),
    }
}

data = [
    # Model, Exp, DICE, AbsRel, FPS, VRAM (GB)
    ('Teacher SEG', '01 (Base)', 56.89, np.nan, 7.14, 2.07),
    ('Teacher SEG', '04 (Nano)', 56.89, np.nan, 7.14, 2.07),
    ('Teacher SEG', '05 (Low-Res)', 53.01, np.nan, 33.3, 0.76),

    ('Teacher DISP', '01 (Base)', np.nan, 2.14, 0.49, 6.96),
    ('Teacher DISP', '04 (Nano)', np.nan, 2.14, 0.49, 6.96),
    ('Teacher DISP', '05 (Low-Res)', np.nan, 2.09, 2.0, 2.85),

    ('ST SEG', '01 (Base)', 43.93, np.nan, 20.08, 1.36),
    ('ST SEG', '04 (Nano)', 42.46, np.nan, 32.25, 1.09),
    ('ST SEG', '05 (Low-Res)', 39.07, np.nan, 75.46, 0.45),

    ('ST DISP', '01 (Base)', np.nan, 5.69, 13.28, 1.40),
    ('ST DISP', '04 (Nano)', np.nan, 6.06, 22.88, 1.13),
    ('ST DISP', '05 (Low-Res)', np.nan, 10.97, 49.24, 0.46),

    ('MT', '01 (Base)', 40.32, 12.26, 10.52, 1.44),
    ('MT', '04 (Nano)', 42.16, 8.25, 16.47, 1.18),
    ('MT', '05 (Low-Res)', 44.36, 16.68, 38.06, 0.49),

    ('MT-KD', '01 (Base)', 51.18, 7.91, 10.50, 1.44),
    ('MT-KD', '04 (Nano)', 55.06, 5.84, 16.47, 1.18),
    ('MT-KD', '05 (Low-Res)', 49.00, 11.57, 38.56, 0.49),
]

df = pd.DataFrame(data, columns=['Model', 'Experiment', 'DICE', 'AbsRel', 'FPS', 'VRAM'])

models = ['Teacher SEG', 'Teacher DISP', 'ST SEG', 'ST DISP', 'MT', 'MT-KD']
experiments = ['01 (Base)', '04 (Nano)', '05 (Low-Res)']
metrics = ['DICE', 'AbsRel', 'FPS', 'VRAM']

metrics_meta = {
    'DICE': {'title': 'DICE [% ↑]'},
    'AbsRel': {'title': 'AbsRel [% ↓]'},
    'FPS': {'title': 'FPS [↑]'},
    'VRAM': {'title': 'VRAM [GB ↓]'}
}

fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.05, shared_yaxes=True)

colors = [px.colors.qualitative.Plotly[3], px.colors.qualitative.Plotly[8], px.colors.qualitative.Plotly[9]]
color_map = dict(zip(experiments, colors))

def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color

seq_fps_data = {
    '01 (Base)': 7.99,
    '04 (Nano)': 13.38,
    '05 (Low-Res)': 29.80,
}

seq_fps_teacher_data = {
    '01 (Base)': 0.46,
    '04 (Nano)': 0.46,
    '05 (Low-Res)': 1 / (1/33.3 + 1/2.0),
}

for col, metric in enumerate(metrics, start=1):
    for exp in experiments:
        df_exp = df[df['Experiment'] == exp].set_index('Model').reindex(models).reset_index()
        
        showlegend = True if col == 1 else False
        
        marker_colors = []
        for model in df_exp['Model']:
            if metric == 'FPS' and model in ['ST SEG', 'ST DISP', 'Teacher SEG', 'Teacher DISP']:
                marker_colors.append(hex_to_rgba(color_map[exp], 0.5))
            else:
                marker_colors.append(hex_to_rgba(color_map[exp], 1.0))
        
        text_labels = []
        for i, val in enumerate(df_exp[metric]):
            model = df_exp['Model'].iloc[i]
            if pd.notna(val) and metric in ['AbsRel', 'FPS']:
                if metric == 'FPS' and model in ['ST SEG', 'ST DISP', 'Teacher SEG', 'Teacher DISP']:
                    seq_val = seq_fps_data[exp] if model in ['ST SEG', 'ST DISP'] else seq_fps_teacher_data[exp]
                    if val < 3 or (pd.notna(seq_val) and seq_val < 3):
                        text_labels.append(f"{seq_val:.2f} / {val:.2f}")
                    else:
                        text_labels.append("")
                elif val < 3:
                    text_labels.append(f"{val:.2f}")
                else:
                    text_labels.append("")
            else:
                text_labels.append("")
        
        fig.add_trace(go.Bar(
            y=df_exp['Model'],
            x=df_exp[metric],
            orientation='h',
            name=exp,
            marker_color=marker_colors,
            offsetgroup=exp,
            showlegend=showlegend,
            legendgroup=exp,
            text=text_labels,
            textposition='outside',
            textfont=dict(size=10, color='black')
        ), row=1, col=col)

        # Add sequence FPS as solid overlay
        if metric == 'FPS':
            seq_x = []
            for m in df_exp['Model']:
                if m in ['ST SEG', 'ST DISP']:
                    seq_x.append(seq_fps_data[exp])
                elif m in ['Teacher SEG', 'Teacher DISP']:
                    seq_x.append(seq_fps_teacher_data[exp])
                else:
                    seq_x.append(np.nan)
                    
            fig.add_trace(go.Bar(
                y=df_exp['Model'],
                x=seq_x,
                orientation='h',
                name=f"{exp} (Seq)",
                marker_color=hex_to_rgba(color_map[exp], 1.0),
                offsetgroup=exp,
                showlegend=False,
                legendgroup=exp,
                hoverinfo='x'
            ), row=1, col=col)

fig.update_layout(
    height=600,
    width=1200,
    barmode='group',
    bargap=0.15,
    bargroupgap=0.05,
    legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5, title_text="Experiment")
)

fig.update_yaxes(title_text="Configuration", autorange="reversed", row=1, col=1)
for col in range(2, 5):
    fig.update_yaxes(showticklabels=False, title_text="", autorange="reversed", row=1, col=col)

fig.update_xaxes(title_text=metrics_meta['DICE']['title'], title_font=dict(size=14), row=1, col=1)
fig.update_xaxes(title_text=metrics_meta['AbsRel']['title'], title_font=dict(size=14), autorange="reversed", row=1, col=2)  # AbsRel (lower is better)
fig.update_xaxes(title_text=metrics_meta['FPS']['title'], title_font=dict(size=14), row=1, col=3)
fig.update_xaxes(title_text=metrics_meta['VRAM']['title'], title_font=dict(size=14), autorange="reversed", row=1, col=4)  # VRAM (lower is better)

for col in range(1, 5):
    fig.add_hline(y=1.5, line_width=1, line_dash="dash", line_color="grey", row=1, col=col)

apply_chart_config(fig, 'H09F01', CHART_CONFIG)
save_figure(fig, height=600, name='H09F01', lrtb_margin=(40, 20, 20, 0), folder='results', skip_sync=skip_sync)

# %%
