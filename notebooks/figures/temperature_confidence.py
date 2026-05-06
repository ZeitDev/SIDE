# %%
import os, sys
sys.path.append(os.path.dirname('../../'))
os.chdir('/data/Zeitler/code/SIDE')

'''
Train only
Temperature                  1.0       1.5       2.0       3.0       4.0
Task                                                                    
Disparity                   0.770954  0.715840  0.666788  0.564569  0.447254
Binary Segmentation         0.972196  0.945113  0.899848  0.763570  0.614543
Multi-Class Segmentation    0.991238  0.972873  0.915289  0.695270  0.471549
'''

import pandas as pd
import plotly.graph_objects as go
from notebooks.figures.helpers import save_figure

# Data Preparation
data = {
    'Temperature': [1.0, 1.5, 2.0, 3.0, 4.0],
    'Disparity': [x * 100 for x in [0.770954, 0.715840, 0.666788, 0.564569, 0.447254]],
    'Binary Segmentation': [x * 100 for x in [0.972196, 0.945113, 0.899848, 0.763570, 0.614543]],
    'Multi-Class Segmentation': [x * 100 for x in [0.991238, 0.972873, 0.915289, 0.695270, 0.471549]]
}

df = pd.DataFrame(data)

# Create Plot
fig = go.Figure()

colors = ['#636EFA', '#EF553B', '#00CC96']
tasks = ['Disparity', 'Binary Segmentation', 'Multi-Class Segmentation']

for i, task in enumerate(tasks):
    fig.add_trace(go.Scatter(
        x=df['Temperature'],
        y=df[task],
        mode='lines+markers',
        name=task,
        line=dict(color=colors[i], width=3),
        marker=dict(size=10)
    ))

# Add 70% confidence threshold line
fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="", annotation_position="bottom left")

# Layout configuration
fig.update_layout(
    xaxis_title='Distillation Temperature',
    yaxis_title='Average Confidence of Train Set [%]',
    legend=dict(
        title='Teacher',
        orientation="h",
        yanchor="top",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
)

save_figure(fig, height=450, name='temperature_confidence', lrtb_margin=(60, 20, 20, 60))

# %%