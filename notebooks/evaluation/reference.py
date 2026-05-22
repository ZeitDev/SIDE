# %% Imports and Setup
import os
import glob
import pickle
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# %% Load all runs from multiple MLflow tracking URIs

if False:
    mlflow_base_path = 'data/mlruns_experiments'  # Base directory where your MLflow experiments are stored
    exp_folders = sorted(glob.glob(os.path.join(mlflow_base_path, 'exp*')))

    final_runs_data = []
    params_runs_data = []
    historic_runs_data = []

    for exp_folder in exp_folders:
        tracking_uri = f'file://{os.path.abspath(exp_folder)}'
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri)
        
        experiments = client.search_experiments()
        for experiment in experiments:
            print(f'Loading from tracking URL: {exp_folder} -> experiment: {experiment.name} (ID: {experiment.experiment_id})')
            
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            for run in runs:
                run_id = run.info.run_id
                run_name = run.data.tags.get('mlflow.runName', 'Unnamed')
                if '/' not in run_name: continue
                
                # general
                identifier = f'{os.path.basename(exp_folder)}/{experiment.name}/{run_name}'
                run_info = {
                    'identifier': identifier,
                    'experiment': os.path.basename(exp_folder),
                    'config': experiment.name,
                    'run_name': run_name,
                    'mode': run_name.split('/')[-1],
                }
                
                # final metrics
                final_info = run_info.copy()
                for m_name, m_val in run.data.metrics.items():
                    if any(x in m_name for x in ['DICE', 'IoU', 'AbsRel', 'Bad3']):
                        m_val *= 100
                    final_info[f'metric.{m_name}'] = m_val
                final_runs_data.append(final_info)
                
                # params 
                params_info = run_info.copy()
                for p_name, p_val in run.data.params.items(): params_info[f'param.{p_name}'] = p_val
                params_info['mlflow_experiment_id'] = experiment.experiment_id
                params_info['mlflow_run_id'] = run_id
                params_runs_data.append(params_info)
                
                # historic metrics
                for metric_name in run.data.metrics.keys():
                    history = client.get_metric_history(run_id, metric_name)
                    for m in history:
                        val = m.value
                        if any(x in metric_name for x in ['DICE', 'IoU', 'AbsRel', 'Bad3']):
                            val *= 100
                        historic_runs_data.append({
                            'identifier': identifier,
                            'experiment': os.path.basename(exp_folder),
                            'config': experiment.name,
                            'run_name': run_name,
                            
                            'metric_name': metric_name,
                            'value': val,
                            'step': m.step,
                            'timestamp': m.timestamp,
                            
                            'mlflow_experiment_id': experiment.experiment_id,
                            'mlflow_run_id': run_id,
                        })
                        

    df_final = pd.DataFrame(final_runs_data)
    print(f'Loaded {len(df_final)} runs.')

    df_params = pd.DataFrame(params_runs_data)
    print(f'Loaded {len(df_params)} parameter data points.')

    df_historic = pd.DataFrame(historic_runs_data)
    print(f'Loaded {len(df_historic)} metric data points.')
        
    data_df = {
        'final': df_final,
        'params': df_params,
        'historic': df_historic
    }

    with open('output/loaded_mlflow_data.pkl', 'wb') as f:
        pickle.dump(data_df, f)
    
# %% Inspect the Data
with open('output/loaded_mlflow_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
    df_final = data['final']
    df_params = data['params']
    df_historic = data['historic']

print('--- Runs Data Summary ---')
df_final.info()

print('\n--- First 5 Runs ---')
display(df_final.head())

print('\n--- Available Metrics ---')
print(df_historic['metric_name'].unique())

# %%
metric = 'metric.best_combined/performance/testing/segmentation/DICE_score/instrument_mean'
data_mtkd = df_final.query('config == "wMT-KD"').dropna(subset=[metric]).copy()

# Group seeds
data_mtkd['group'] = data_mtkd['experiment'] + '/' + data_mtkd['config']

fig = px.box(
    data_frame=data_mtkd,
    x='group',           # <-- Use the shared group name here
    y=metric,
    title='Segmentation Performance (MT-KD Configurations)',
    labels={
        'group': 'Ablation Configuration',
        metric: 'Validation DICE Score'
    },
    template='plotly_white'
)

fig.show()

# %%
columns_df = pd.DataFrame({'df_final columns': df_final.columns})
columns_df.to_csv('output/mlflow_data_dictionary.csv', index=False)

# %%
tier_file = 'output/key_dictionary_v3.csv'
if os.path.exists(tier_file):
    df_tiers = pd.read_csv(tier_file)
    
    desired_tier_max = 2
    tier_columns = df_tiers.loc[df_tiers['Tier'] <= desired_tier_max, 'df_final columns'].tolist()
    
    valid_columns = [c for c in tier_columns if c in df_final.columns]
    
    df_final_filtered = df_final[valid_columns]
    
    print(f"Filtered df_final to {len(valid_columns)} columns (Tier <= {desired_tier_max})")
    display(df_final_filtered.head())
    
# %%
### THESIS ###

# %% Appendix Tables
print("--- Appendix Table 1: Aggregated Master Table ---")

metrics_mapping_app = {
    'DICE ($\\uparrow$)': ('metric.best_combined/performance/testing/segmentation/DICE_score/instrument_mean', 'metric.best_segmentation/performance/testing/segmentation/DICE_score/instrument_mean', True, 'seg'),
    'IoU ($\\uparrow$)': ('metric.best_combined/performance/testing/segmentation/IoU_score/instrument_mean', 'metric.best_segmentation/performance/testing/segmentation/IoU_score/instrument_mean', True, 'seg'),
    'AbsRel ($\\downarrow$)': ('metric.best_combined/performance/testing/disparity/AbsRel_rate', 'metric.best_disparity/performance/testing/disparity/AbsRel_rate', False, 'disp'),
    'Bad3 ($\\downarrow$)': ('metric.best_combined/performance/testing/disparity/Bad3_rate', 'metric.best_disparity/performance/testing/disparity/Bad3_rate', False, 'disp'),
    'EPE ($\\downarrow$)': ('metric.best_combined/performance/testing/disparity/EPE_px', 'metric.best_disparity/performance/testing/disparity/EPE_px', False, 'disp'),
    'MAE ($\\downarrow$)': ('metric.best_combined/performance/testing/disparity/MAE_mm', 'metric.best_disparity/performance/testing/disparity/MAE_mm', False, 'disp')
}

df_app = df_final.copy()
# Filter out 'train' mode and clean up Date:Time
df_app = df_app[df_app['mode'] != 'train'].copy()
df_app['Date:Time'] = df_app['run_name'].apply(lambda x: x.split('/')[0] if '/' in str(x) else x)

active_metrics = []
for short_name, (comb_col, sing_col, higher_is_better, task_type) in metrics_mapping_app.items():
    if comb_col in df_app.columns and sing_col in df_app.columns:
        df_app[short_name] = df_app[comb_col].fillna(df_app[sing_col])
        active_metrics.append(short_name)
    elif comb_col in df_app.columns:
        df_app[short_name] = df_app[comb_col]
        active_metrics.append(short_name)
    elif sing_col in df_app.columns:
        df_app[short_name] = df_app[sing_col]
        active_metrics.append(short_name)

config_map_app = {
    'SEG': 'SEG',
    'DISP': 'DISP',
    'wMT': 'MT',
    'wMT-KD': 'MT-KD',
    'wMT_KD': 'MT-KD'
}
df_app['config'] = df_app['config'].replace(config_map_app)

grouped_app = df_app.groupby(['experiment', 'config'])[active_metrics].agg(['mean', 'std']).reset_index()

melted_app_list = []
for idx, row in grouped_app.iterrows():
    exp = row[('experiment', '')]
    cfg = row[('config', '')]
    for m in active_metrics:
        mean_val = row[(m, 'mean')]
        std_val = row[(m, 'std')]
        val_str = f"{mean_val:05.2f} ± {std_val:05.2f}" if pd.notna(mean_val) else "-"
        melted_app_list.append({'experiment': exp, 'config': cfg, 'Metric': m, 'Value': val_str})

melted_app = pd.DataFrame(melted_app_list)

# Exclusion logic
for short_name, (_, _, _, task_type) in metrics_mapping_app.items():
    if task_type == 'seg':
        melted_app = melted_app[~((melted_app['Metric'] == short_name) & (melted_app['config'] == 'DISP'))]
    else:
        melted_app = melted_app[~((melted_app['Metric'] == short_name) & (melted_app['config'] == 'SEG'))]

melted_app['config'] = melted_app['config'].replace({'SEG': 'Single-Task', 'DISP': 'Single-Task'})
pivot_app = melted_app.pivot(index=['Metric', 'experiment'], columns='config', values='Value')

pivot_app.index = pivot_app.index.set_levels(
    pivot_app.index.levels[1].str.extract(r'(\d+)')[0].values, level='experiment'
)
pivot_app = pivot_app.fillna('N/A')

app_metrics_order = [m for m in metrics_mapping_app.keys() if m in active_metrics]
pivot_app = pivot_app.reindex(app_metrics_order, level=0)
pivot_app.index.names = ['Metric', 'ID']
pivot_app.columns.name = None

desired_order_app = ['Single-Task', 'MT', 'MT-KD']
ordered_cols_app = [c for c in desired_order_app if c in pivot_app.columns]
remaining_cols_app = [c for c in pivot_app.columns if c not in ordered_cols_app]
pivot_app = pivot_app[ordered_cols_app + remaining_cols_app]

def bold_multi_row_app(df):
    df_bolded = df.copy()
    for short_name, (_, _, higher_is_better, _) in metrics_mapping_app.items():
        if short_name not in df_bolded.index.get_level_values(0): continue
        sub_df = df_bolded.loc[short_name]
        
        for config in sub_df.columns:
            column_data = sub_df[config]
            means = []
            for idx, val in column_data.items():
                if val != "N/A" and isinstance(val, str):
                    try:
                        mean_val = float(val.split(" ± ")[0])
                        means.append((idx, mean_val))
                    except: pass
            
            if means:
                best_val_idx = max(means, key=lambda x: x[1])[0] if higher_is_better else min(means, key=lambda x: x[1])[0]
                df_bolded.loc[(short_name, best_val_idx), config] = f"\\textbf{{{column_data[best_val_idx]}}}"
    return df_bolded

pivot_app = bold_multi_row_app(pivot_app)

print(pivot_app.to_latex(
    escape=False, 
    index=True, 
    multirow=True, 
    index_names=True,
    column_format='ll' + 'l' * len(pivot_app.columns)
))


print("\n--- Appendix Table 2: Raw Run Log ---")
raw_cols = ['experiment', 'config', 'Date:Time'] + active_metrics
df_raw = df_app[raw_cols].copy()
# Extract experiment number (e.g., 'exp1' -> '1') and rename column
df_raw['experiment'] = df_raw['experiment'].str.extract(r'(\d+)')[0]
df_raw = df_raw.rename(columns={'experiment': 'ID'})

# Sort rows: ID (numeric), Config (custom order), Date:Time
config_sort_order = {'SEG': 0, 'DISP': 1, 'MT': 2, 'MT-KD': 3}
df_raw['config_sort'] = df_raw['config'].map(config_sort_order)
df_raw['ID_numeric'] = pd.to_numeric(df_raw['ID'])

df_raw = df_raw.sort_values(['ID_numeric', 'config_sort', 'Date:Time']).drop(columns=['config_sort', 'ID_numeric'])

latex_raw = df_raw.to_latex(
    index=False, 
    longtable=True, 
    escape=False,
    float_format="%.2f",
    na_rep="-"
)
latex_raw = "\\begin{landscape}\n" + latex_raw + "\\end{landscape}\n"
print(latex_raw)

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

# only needed for prototyping with current data
config_map_fig = {
    'SEG': 'SEG',
    'DISP': 'DISP',
    'wMT': 'MT',
    'wMT-KD': 'MT-KD',
    'wMT_KD': 'MT-KD'
}
df_bench['config'] = df_bench['config'].replace(config_map_fig)

# Group by experiment and config to get mean and std across seeds of config per experiment
grouped = df_bench.groupby(['experiment', 'config']).agg(
    DICE_mean=('DICE_score', 'mean'),
    DICE_std=('DICE_score', 'std'),
    AbsRel_mean=('AbsRel_rate', 'mean'),
    AbsRel_std=('AbsRel_rate', 'std')
).reset_index()

# %% Figure 1 - Slopegraph: Configuration Performance
from plotly.subplots import make_subplots

# Get baseline values from 'exp1' for fallback if missing
baseline_exp_data = grouped[grouped['experiment'] == 'exp1'].set_index('config') if 'exp1' in grouped['experiment'].values else None
baseline_seg_dice = baseline_exp_data.get('DICE_mean', {}).get('SEG', np.nan) if baseline_exp_data is not None else np.nan
baseline_disp_absrel = baseline_exp_data.get('AbsRel_mean', {}).get('DISP', np.nan) if baseline_exp_data is not None else np.nan

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

    # Dynamic fallback to exp1 if ST data is missing
    if pd.isna(seg_dice):
        seg_dice = baseline_seg_dice
    if pd.isna(disp_absrel):
        disp_absrel = baseline_disp_absrel

    fig1_data.append({
        'experiment': exp,
        'ST_DICE': seg_dice, 'MT_DICE': mt_dice, 'MT-KD_DICE': mtkd_dice,
        'ST_AbsRel': disp_absrel, 'MT_AbsRel': mt_absrel, 'MT-KD_AbsRel': mtkd_absrel
    })

df_fig1 = pd.DataFrame(fig1_data)

fig_config = make_subplots(rows=1, cols=2, subplot_titles=("DICE Score", "AbsRel Rate"))
x_cats = ['ST', 'MT', 'MT-KD']
colors = px.colors.qualitative.Plotly

for i, exp in enumerate(df_fig1['experiment']):
    row = df_fig1[df_fig1['experiment'] == exp].iloc[0]
    color = colors[i % len(colors)]
    
    # DICE
    y_dice = [row['ST_DICE'], row['MT_DICE'], row['MT-KD_DICE']]
    fig_config.add_trace(go.Scatter(x=x_cats, y=y_dice, mode='lines+markers',
                                    name=exp, legendgroup=exp, marker_color=color),
                         row=1, col=1)
    
    # AbsRel
    y_absrel = [row['ST_AbsRel'], row['MT_AbsRel'], row['MT-KD_AbsRel']]
    fig_config.add_trace(go.Scatter(x=x_cats, y=y_absrel, mode='lines+markers',
                                    name=exp, legendgroup=exp, showlegend=False, marker_color=color),
                         row=1, col=2)

fig_config.update_layout(
    title="Configuration Performance (MT-KD Narrative)",
    template='plotly_white',
    height=500
)

# Remove horizontal grids to keep slopes clear, reverse AbsRel axis
fig_config.update_yaxes(showgrid=False, row=1, col=1)
fig_config.update_yaxes(showgrid=False, autorange="reversed", row=1, col=2)

fig_config.show()



# %% TABLE 1: Global Results Matrix (LaTeX Table)
print("--- Global Results Matrix (LaTeX) ---")

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

# Extract experiment number (e.g., 'exp1' -> '1')
pivot_combined.index = pivot_combined.index.set_levels(
    pivot_combined.index.levels[1].str.extract(r'(\d+)')[0].values,
    level='experiment'
)

# Fill NaN with N/A
pivot_combined = pivot_combined.fillna('N/A')

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

# Bold best values per row/metric group
def bold_multi_row(df):
    df_bolded = df.copy()
    for metric in metrics_order:
        if metric not in df_bolded.index.get_level_values(0): continue
        higher_is_better = ('DICE' in metric)
        
        # Get slice for this metric
        sub_df = df_bolded.loc[metric]
        
        for config in sub_df.columns:
            column_data = sub_df[config]
            means = []
            for idx, val in column_data.items():
                if val != "N/A" and isinstance(val, str):
                    try:
                        mean_val = float(val.split(" ± ")[0])
                        means.append((idx, mean_val))
                    except: pass
            
            if means:
                best_val_idx = max(means, key=lambda x: x[1])[0] if higher_is_better else min(means, key=lambda x: x[1])[0]
                df_bolded.loc[(metric, best_val_idx), config] = f"\\textbf{{{column_data[best_val_idx]}}}"
    return df_bolded

pivot_combined = bold_multi_row(pivot_combined)

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

# %% 2. Ablation Delta Heatmap (Figure) -> Now a Slopegraph
fig_ablation = make_subplots(rows=1, cols=2, subplot_titles=("DICE Score", "AbsRel Rate"))

configs = ['ST', 'MT', 'MT-KD']
exps = grouped['experiment'].unique()
colors = px.colors.qualitative.Plotly

for i, cfg_label in enumerate(configs):
    color = colors[i % len(configs)]
    y_dice = []
    y_absrel = []
    for exp in exps:
        exp_data = grouped[grouped['experiment'] == exp].set_index('config')
        
        if cfg_label == 'ST':
            dice_val = exp_data.get('DICE_mean', {}).get('SEG', np.nan)
            absrel_val = exp_data.get('AbsRel_mean', {}).get('DISP', np.nan)
            
            # Dynamic fallback to exp1 if ST data is missing
            if pd.isna(dice_val):
                dice_val = baseline_seg_dice
            if pd.isna(absrel_val):
                absrel_val = baseline_disp_absrel
        else:
            dice_val = exp_data.get('DICE_mean', {}).get(cfg_label, np.nan)
            absrel_val = exp_data.get('AbsRel_mean', {}).get(cfg_label, np.nan)
            
        y_dice.append(dice_val)
        y_absrel.append(absrel_val)

    # DICE
    fig_ablation.add_trace(go.Scatter(x=exps, y=y_dice, mode='lines+markers',
                                      name=cfg_label, legendgroup=cfg_label, marker_color=color),
                           row=1, col=1)
    # AbsRel
    fig_ablation.add_trace(go.Scatter(x=exps, y=y_absrel, mode='lines+markers',
                                      name=cfg_label, legendgroup=cfg_label, showlegend=False, marker_color=color),
                           row=1, col=2)

fig_ablation.update_layout(
    title="Ablation Experiments Performance",
    template='plotly_white',
    height=500
)

# Remove horizontal grids to keep slopes clear, reverse AbsRel axis
fig_ablation.update_yaxes(showgrid=False, row=1, col=1)
fig_ablation.update_yaxes(showgrid=False, autorange="reversed", row=1, col=2)

fig_ablation.show()


# %%

