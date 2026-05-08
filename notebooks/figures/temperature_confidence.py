# %%
import os, sys
sys.path.append(os.path.dirname('../../'))
os.chdir('/data/Zeitler/code/SIDE')

'''
Train only
Temperature                  1.0       1.5       2.0       2.5       3.0       3.5       4.0
Task                                                                       
disparity_128_256_256   0.770954  0.715840  0.666788  0.617503  0.564569  0.507206  0.447254 
segmentation_2_256_256  0.978072  0.956652  0.911692  0.844661  0.765904  0.685089  0.608310 
segmentation_8_256_256  0.993996  0.981560  0.937401  0.853292  0.743663  0.628890  0.523107 

Disparity -> 1.66
Binary Segmentation -> 3.41
Multi-Class Segmentation -> 3.19
'''

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from notebooks.figures.helpers import save_figure

# Data Preparation
data = {
    'Temperature': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    'Disparity': [x * 100 for x in [0.770954, 0.715840, 0.666788, 0.617503, 0.564569, 0.507206, 0.447254]],
    'Binary Segmentation': [x * 100 for x in [0.978072, 0.956652, 0.911692, 0.844661, 0.765904, 0.685089, 0.608310]],
    'Multi-Class Segmentation': [x * 100 for x in [0.993996, 0.981560, 0.937401, 0.853292, 0.743663, 0.628890, 0.523107]]
}

df = pd.DataFrame(data)

# Create Plot
fig = go.Figure()

# Add 70% confidence threshold line
fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 0, 0, 0.5)", annotation_text="", annotation_position="bottom left")

colors = ['#636EFA', '#EF553B', '#00CC96']
tasks = ['Disparity', 'Binary Segmentation', 'Multi-Class Segmentation']
chosen_temps = {'Disparity': 1.66, 'Binary Segmentation': 3.41, 'Multi-Class Segmentation': 3.19}

for i, task in enumerate(tasks):
    fig.add_trace(go.Scatter(
        x=df['Temperature'],
        y=df[task],
        mode='lines+markers',
        name=task,
        line=dict(color=colors[i], width=3),
        marker=dict(size=10)
    ))
    
    # Highlight chosen temperature
    chosen_temp = chosen_temps[task]
    chosen_val = np.interp(chosen_temp, df['Temperature'], df[task])
    fig.add_trace(go.Scatter(
        x=[chosen_temp],
        y=[chosen_val],
        mode='markers',
        marker=dict(symbol='star', color=colors[i], size=15, line=dict(color='black', width=1)),
        showlegend=False,
        hoverinfo='skip'
    ))

# Layout configuration
fig.update_layout(
    xaxis_title='Distillation Temperature',
    yaxis_title='Average Raw Confidence of Train Set [%]',
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