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
    'DICE [\\% $\\uparrow$]': ('metric.best_combined/performance/testing/segmentation/DICE_score/instrument_mean', 'metric.best_segmentation/performance/testing/segmentation/DICE_score/instrument_mean', True, 'seg'),
    'IoU [\\% $\\uparrow$]': ('metric.best_combined/performance/testing/segmentation/IoU_score/instrument_mean', 'metric.best_segmentation/performance/testing/segmentation/IoU_score/instrument_mean', True, 'seg'),
    'AbsRel [\\% $\\downarrow$]': ('metric.best_combined/performance/testing/disparity/AbsRel_rate', 'metric.best_disparity/performance/testing/disparity/AbsRel_rate', False, 'disp'),
    'Bad3 [\\% $\\downarrow$]': ('metric.best_combined/performance/testing/disparity/Bad3_rate', 'metric.best_disparity/performance/testing/disparity/Bad3_rate', False, 'disp'),
    'EPE [px $\\downarrow$]': ('metric.best_combined/performance/testing/disparity/EPE_px', 'metric.best_disparity/performance/testing/disparity/EPE_px', False, 'disp'),
    'MAE [mm $\\downarrow$]': ('metric.best_combined/performance/testing/disparity/MAE_mm', 'metric.best_disparity/performance/testing/disparity/MAE_mm', False, 'disp')
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

grouped_app = df_app.groupby(['experiment', 'config'])[active_metrics].agg(['median', 'min', 'max']).reset_index()

melted_app_list = []
for idx, row in grouped_app.iterrows():
    exp = row[('experiment', '')]
    cfg = row[('config', '')]
    for m in active_metrics:
        med = row[(m, 'median')]
        vmin = row[(m, 'min')]
        vmax = row[(m, 'max')]
        
        if pd.notna(med):
            val_str = f"${med:05.2f}_{{-{med - vmin:05.2f}}}^{{+{vmax - med:05.2f}}}$"
        else:
            val_str = "-"
            
        melted_app_list.append({'experiment': exp, 'config': cfg, 'Metric': m, 'Value': val_str})

melted_app = pd.DataFrame(melted_app_list)

# Exclusion logic
for short_name, (_, _, _, task_type) in metrics_mapping_app.items():
    if task_type == 'seg':
        melted_app = melted_app[~((melted_app['Metric'] == short_name) & (melted_app['config'] == 'DISP'))]
    else:
        melted_app = melted_app[~((melted_app['Metric'] == short_name) & (melted_app['config'] == 'SEG'))]

melted_app['config'] = melted_app['config'].replace({'SEG': 'ST', 'DISP': 'ST'})
pivot_app = melted_app.pivot(index=['Metric', 'experiment'], columns='config', values='Value')

pivot_app.index = pivot_app.index.set_levels(
    pivot_app.index.levels[1].str.extract(r'(\d+)')[0].values, level='experiment'
)
pivot_app = pivot_app.fillna('-')

app_metrics_order = [m for m in metrics_mapping_app.keys() if m in active_metrics]
pivot_app = pivot_app.reindex(app_metrics_order, level=0)
pivot_app.index.names = ['Metric', 'ID']
pivot_app.columns.name = None

desired_order_app = ['ST', 'MT', 'MT-KD']
ordered_cols_app = [c for c in desired_order_app if c in pivot_app.columns]
remaining_cols_app = [c for c in pivot_app.columns if c not in ordered_cols_app]
pivot_app = pivot_app[ordered_cols_app + remaining_cols_app]

print("\\renewcommand{{\\arraystretch}}{{1.4}}")
print(pivot_app.to_latex(
    escape=False, 
    index=True, 
    longtable=True, 
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

# %% Appendix Table 3 - Raw SERR and NLE for all experiments and configs in landscape table

with open('./notebooks/evaluation/storage/entropy_metrics.pkl', 'rb') as f:
    df_entropy = pickle.load(f)

df_app_ent = df_entropy.copy()
if 'mode' in df_app_ent.columns:
    df_app_ent = df_app_ent[df_app_ent['mode'] != 'train'].copy()

df_app_ent['Date:Time'] = df_app_ent['run_name'].apply(lambda x: str(x).split('/')[0] if '/' in str(x) else x)

rename_dict = {}
modules = [
    ('Enc', 'encoder.stages_', 4),
    ('Seg', 'decoders.segmentation.decoder.blocks.', 3),
    ('Seg', 'decoders.segmentation.decoder.final_block', 1),
    ('Disp', 'decoders.disparity.decoder.blocks.', 3),
    ('Disp', 'decoders.disparity.decoder.final_block', 1)
]

layer_cols = []
for m_name, m_prefix, count in modules:
    if 'final_block' in m_prefix:
        layers = [("", 4)]
    else:
        layers = [(str(i), i+1) for i in range(count)]
        
    for suff, num in layers:
        base = m_prefix + suff
        col_nle = base + '_norm_entropy'
        col_serr = base + '_erank_ratio'
        
        layer_col = f"{m_name}{num}"
        
        def combine(row, cnle=col_nle, cserr=col_serr):
            vnle = row.get(cnle, float('nan'))
            vserr = row.get(cserr, float('nan'))
            if pd.isna(vnle) and pd.isna(vserr):
                return '-'
            snle = f"{vnle*100:.2f}" if not pd.isna(vnle) else "-"
            sserr = f"{vserr*100:.2f}" if not pd.isna(vserr) else "-"
            return f"{sserr} / {snle}"

        df_app_ent[layer_col] = df_app_ent.apply(combine, axis=1)

df_app_ent['experiment'] = df_app_ent['experiment'].str.extract(r'(\d+)')[0]
df_app_ent = df_app_ent.rename(columns={'experiment': 'ID', 'config': 'Config'})

config_sort_order = {'SEG': 0, 'DISP': 1, 'MT': 2, 'MT-KD': 3}
df_app_ent['config_sort'] = df_app_ent['Config'].map(config_sort_order)
df_app_ent['ID_numeric'] = pd.to_numeric(df_app_ent['ID'])

df_app_ent = df_app_ent.sort_values(['ID_numeric', 'config_sort', 'Date:Time']).drop(columns=['config_sort', 'ID_numeric'])

new_rows = []
for _, row in df_app_ent.iterrows():
    cfg = row['Config']
    
    seg_row = {
        'ID': row['ID'], 'Config': row['Config'], 'Date:Time': row['Date:Time'],
        'Enc1': row['Enc1'], 'Enc2': row['Enc2'], 'Enc3': row['Enc3'], 'Enc4': row['Enc4'],
        'Dec.': 'SEG',
        'Dec1': row['Seg1'], 'Dec2': row['Seg2'], 'Dec3': row['Seg3'], 'Dec4': row['Seg4']
    }
    
    disp_row = {
        'ID': row['ID'], 'Config': row['Config'], 'Date:Time': row['Date:Time'],
        'Enc1': row['Enc1'], 'Enc2': row['Enc2'], 'Enc3': row['Enc3'], 'Enc4': row['Enc4'],
        'Dec.': 'DISP',
        'Dec1': row['Disp1'], 'Dec2': row['Disp2'], 'Dec3': row['Disp3'], 'Dec4': row['Disp4']
    }
    
    if cfg == 'SEG':
        new_rows.append(seg_row)
    elif cfg == 'DISP':
        new_rows.append(disp_row)
    else:  # MT or MT-KD
        disp_row['ID'] = ''
        disp_row['Config'] = ''
        disp_row['Date:Time'] = ''
        disp_row['Enc1'] = ''
        disp_row['Enc2'] = ''
        disp_row['Enc3'] = ''
        disp_row['Enc4'] = ''
        new_rows.append(seg_row)
        new_rows.append(disp_row)

df_stacked = pd.DataFrame(new_rows)

new_columns = []
for col in df_stacked.columns:
    if col in ['ID', 'Config', 'Date:Time', 'Dec.']:
        new_columns.append(('', col))
    else:
        new_columns.append(('SERR / NLE [\\%]', col))

df_stacked.columns = pd.MultiIndex.from_tuples(new_columns)

latex_raw_ent = df_stacked.to_latex(
    index=False, 
    longtable=True, 
    escape=False,
    na_rep="-",
    multicolumn=True,
    multicolumn_format='c'
)
latex_raw_ent = "\\begin{landscape}\n\\scriptsize\n" + latex_raw_ent + "\\normalsize\n\\end{landscape}\n"
print(latex_raw_ent)

# %%