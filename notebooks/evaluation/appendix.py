# %%
import os, sys
sys.path.append(os.path.dirname('../../'))
os.chdir(os.path.dirname('../../'))

import pickle
import pandas as pd

# %% 
with open('./notebooks/evaluation/storage/dataframes.pkl', 'rb') as f:
    data = pickle.load(f)
    
    df_final = data['final']
    df_params = data['params']
    df_historic = data['historic']

# %% Appendix Table 1 - Performance Summary All metrics

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
pivot_app = pivot_app.fillna('-')

app_metrics_order = [m for m in metrics_mapping_app.keys() if m in active_metrics]
pivot_app = pivot_app.reindex(app_metrics_order, level=0)
pivot_app.index.names = ['Metric', 'ID']
pivot_app.columns.name = None

desired_order_app = ['Single-Task', 'MT', 'MT-KD']
ordered_cols_app = [c for c in desired_order_app if c in pivot_app.columns]
remaining_cols_app = [c for c in pivot_app.columns if c not in ordered_cols_app]
pivot_app = pivot_app[ordered_cols_app + remaining_cols_app]

print(pivot_app.to_latex(
    escape=False, 
    index=True, 
    multirow=True, 
    index_names=True,
    column_format='ll' + 'c' * len(pivot_app.columns)
))

# %% Appendix Table 2 - Raw Run Log

raw_cols = ['experiment', 'config', 'Date:Time'] + active_metrics
df_raw = df_app[raw_cols].copy()
# Extract experiment number (e.g., 'exp01' -> '1') and rename column
df_raw['experiment'] = df_raw['experiment'].str.extract(r'(\d+)')[0]
df_raw = df_raw.rename(columns={'experiment': 'ID', 'config': 'Config'})

# Sort rows: ID (numeric), Config (custom order), Date:Time
config_sort_order = {'SEG': 0, 'DISP': 1, 'MT': 2, 'MT-KD': 3}
df_raw['config_sort'] = df_raw['Config'].map(config_sort_order)
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

# %%