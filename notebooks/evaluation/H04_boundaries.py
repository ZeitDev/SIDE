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

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment()

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Restrict to GPU 1 for this notebook


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
        'y1': dict(range=[30, 60], dtick=5),
        'y2': dict(range=[40, 0], dtick=5),
        'y3': dict(range=[105, 20], dtick=10),
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
    'AbsRel': ('disparity', 'AbsRel_rate'),
    'Bad3': ('disparity', 'Bad3_rate'),
    'DICE': ('segmentation', 'DICE_score/instrument_mean')
}

# Process metrics with fallback logic
for m_name, (m_task, m_col) in metrics.items():
    col = f"metric.best_combined/performance/testing/{m_task}/{m_col}"
    fallback = f"metric.best_{m_task}/performance/testing/{m_task}/{m_col}"
    df_f02[m_name] = df_f02[col].fillna(df_f02[fallback])

# Styling
colors_dict = {
    'ST': px.colors.qualitative.Plotly[0],
    'MT': px.colors.qualitative.Plotly[1],
    'MT-KD': px.colors.qualitative.Plotly[2]
}
regimes = ['01 (ON)', '02 (OFF)']
display_configs = ['ST', 'MT', 'MT-KD']

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)

for row, task in enumerate(['DICE', 'AbsRel', 'Bad3'], start=1):
    for config in display_configs:
        for regime in regimes:
            data = df_f02[(df_f02['regime'] == regime) & (df_f02['config_mapped'] == config)][task].dropna()
            
            is_inherited = False
            if len(data) == 0 and regime != regimes[0]:
                data = df_f02[(df_f02['regime'] == regimes[0]) & (df_f02['config_mapped'] == config)][task].dropna()
                is_inherited = True
            
            showlegend = True if row == 1 and regime == regimes[0] else False
            
            fig.add_trace(go.Box(
                y=data,
                x=[regime] * len(data),
                orientation='v',
                name=config,
                marker_color=colors_dict[config],
                boxpoints='all',
                jitter=0.5,
                pointpos=-2.0,
                showlegend=showlegend,
                legendgroup=config,
                offsetgroup=config,
                opacity=0.5 if is_inherited else 1.0
            ), row=row, col=1)

fig.update_layout(
    # template='plotly_white',
    height=800,
    width=500,
    boxmode='group',
    boxgroupgap=0.6,
    boxgap=0.3,
    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5, title_text="Config")
)

fig.update_yaxes(title_text="DICE Score [% ↑]", title_standoff=20, row=1, col=1)
fig.update_yaxes(title_text="AbsRel Rate [% ↓]", title_standoff=20, autorange="reversed", row=2, col=1)
fig.update_yaxes(title_text="Bad3 Rate [% ↓]", title_standoff=20, autorange="reversed", row=3, col=1)
fig.update_xaxes(title_text="", showticklabels=False, row=1, col=1)
fig.update_xaxes(title_text="", showticklabels=False, row=2, col=1)
fig.update_xaxes(title_text="Experiment (Disparity Gating)", row=3, col=1)
apply_chart_config(fig, 'H04F02', CHART_CONFIG)
save_figure(fig, height=800, name='H04F02', lrtb_margin=(40, 20, 0, 0), standoff=10, folder='results', skip_sync=skip_sync)

# %% H04F03
# H04F03

import yaml
import torch
import mlflow
import numpy as np
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from metrics.disparity import AbsRel, Bad3

device = torch.device('cuda')

# Color tools
ERROR_GREEN = np.array([0, 200, 0])
ERROR_YELLOW = np.array([255, 255, 0])
ERROR_RED = np.array([255, 0, 0])

def error_colormap(error_map):
    if hasattr(error_map, 'cpu'):
        error_map = error_map.cpu().numpy()
    error_map = np.squeeze(error_map)
    color = np.zeros((*error_map.shape, 3), dtype=np.uint8)
    mask1 = error_map <= 0.5
    mask2 = error_map > 0.5

    color[mask1] = (
        ERROR_GREEN * (1 - 2 * error_map[mask1, None]) +
        ERROR_YELLOW * (2 * error_map[mask1, None])
    ).astype(np.uint8)

    color[mask2] = (
        ERROR_YELLOW * (2 - 2 * error_map[mask2, None]) +
        ERROR_RED * (2 * error_map[mask2, None] - 1)
    ).astype(np.uint8)

    return color

def prepare_rgb(img_tensor):
    if hasattr(img_tensor, 'cpu'):
        img = img_tensor.cpu().numpy()
    else:
        img = np.array(img_tensor)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.moveaxis(img, 0, -1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return (img * 255).astype(np.uint8)

def load_model_from_run(experiment, config_name, run_name, task_mode):
    mlflow.set_tracking_uri(f'/data/Zeitler/code/SIDE/mlruns_experiments/{experiment}')
    mlflow_experiment = mlflow.get_experiment_by_name(config_name)
    train_run_name = run_name.replace('/test', '/train')
    
    # Get model_run_id from tags
    model_run_id = mlflow.search_runs(
        experiment_ids=[mlflow_experiment.experiment_id],
        filter_string=f'tags.mlflow.runName = "{train_run_name}"',
        order_by=['attributes.start_time DESC'],
        max_results=1
    ).iloc[0].run_id

    model_path = f'runs:/{model_run_id}/best_model_{task_mode}'
    model = mlflow.pytorch.load_model(model_path, map_location=device)
    model.eval()

    base_config_filepath = './configs/base.yaml'
    experiment_config_filepath = f'./configs/{experiment}/{config_name}.yaml'

    with open(base_config_filepath, 'r') as f: base_config = yaml.safe_load(f)
    with open(experiment_config_filepath, 'r') as f: experiment_config = yaml.safe_load(f)
    from utils import helpers
    cfg = helpers.deep_merge(experiment_config, base_config)

    return model, cfg

# Find median runs for MT and MT-KD in exp01
df_exp01 = df_final[df_final['experiment'] == 'exp01'].copy()
df_exp01 = df_exp01.dropna(subset=['metric.best_combined/performance/testing/disparity/AbsRel_rate'])
df_exp01['AbsRel'] = df_exp01['metric.best_combined/performance/testing/disparity/AbsRel_rate']
df_exp01['Bad3'] = df_exp01['metric.best_combined/performance/testing/disparity/Bad3_rate']

# Harmonic Mean of success rates for median seed selection
succ_abs_exp = (100.0 - df_exp01['AbsRel']).clip(0, 100) / 100.0
succ_bad_exp = (100.0 - df_exp01['Bad3']).clip(0, 100) / 100.0
df_exp01['HarmonicScore'] = (2 * succ_abs_exp * succ_bad_exp) / (succ_abs_exp + succ_bad_exp + 1e-8)

best_runs = {}
for conf in ['MT', 'MT-KD']:
    df_conf = df_exp01[(df_exp01['config'] == conf) & (df_exp01['HarmonicScore'].notna())].sort_values('HarmonicScore')
    median_run = df_conf.iloc[(len(df_conf) - 1) // 2]
    best_runs[conf] = median_run
    
# Load MT model and config
model_mt, cfg_mt = load_model_from_run('exp01', 'MT', best_runs['MT']['run_name'], 'combined')

# Load MT-KD model
model_mtkd, _ = load_model_from_run('exp01', 'MT-KD', best_runs['MT-KD']['run_name'], 'combined')

# %%
# Load Dataloader
from utils import helpers
from data.transforms import build_transforms
test_transforms = build_transforms(cfg_mt, mode='test')
dataset_class = helpers.load(cfg_mt['data']['dataset'])
dataset_test = dataset_class(mode='test', config=cfg_mt, transforms=test_transforms)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=cfg_mt['general']['num_workers'],
    pin_memory=cfg_mt['general']['pin_memory'],
    persistent_workers=False
)

absrel_metric = AbsRel(max_disparity=cfg_mt['data']['max_disparity'], device=device)
bad3_metric = Bad3(max_disparity=cfg_mt['data']['max_disparity'], device=device)

calculate_median = False
chosen_idx = 186 # 91 (HM), 186 (Mean) 72, 61, 52, 186, 498, 187, 15

if calculate_median:
    sample_metrics = []
    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader_test):
            left_images = data_batch['image'].to(device)
            right_images = data_batch.get('right_image').to(device) if 'right_image' in data_batch else None
            
            targets_disp = data_batch.get('disparity')
            targets_disp = targets_disp.to(device) if targets_disp is not None else None
            
            baseline = data_batch.get('baseline')
            baseline = baseline.to(device) if baseline is not None else None
            
            focal_length = data_batch.get('focal_length')
            focal_length = focal_length.to(device) if focal_length is not None else None
            
            # MT
            outputs_mt = model_mt(left_images, right_images)
            mt_disp = outputs_mt['disparity']
            absrel_metric.reset()
            absrel_metric.update(mt_disp, targets_disp, baseline, focal_length)
            mt_absrel = absrel_metric.compute()['AbsRel_rate']
            bad3_metric.reset()
            bad3_metric.update(mt_disp, targets_disp, baseline, focal_length)
            mt_bad3 = bad3_metric.compute()['Bad3_rate']
            
            # MT-KD
            outputs_mtkd = model_mtkd(left_images, right_images)
            mtkd_disp = outputs_mtkd['disparity']
            absrel_metric.reset()
            absrel_metric.update(mtkd_disp, targets_disp, baseline, focal_length)
            mtkd_absrel = absrel_metric.compute()['AbsRel_rate']
            bad3_metric.reset()
            bad3_metric.update(mtkd_disp, targets_disp, baseline, focal_length)
            mtkd_bad3 = bad3_metric.compute()['Bad3_rate']
            
            # Harmonic Mean of success rates for sample selection
            succ_abs_mt = max(0.0, 1.0 - mt_absrel)
            succ_bad_mt = max(0.0, 1.0 - mt_bad3)
            score_mt = (2 * succ_abs_mt * succ_bad_mt) / (succ_abs_mt + succ_bad_mt + 1e-8)
            score_mt_mean = (succ_abs_mt + succ_bad_mt) / 2.0
            
            succ_abs_mtkd = max(0.0, 1.0 - mtkd_absrel)
            succ_bad_mtkd = max(0.0, 1.0 - mtkd_bad3)
            score_mtkd = (2 * succ_abs_mtkd * succ_bad_mtkd) / (succ_abs_mtkd + succ_bad_mtkd + 1e-8)
            score_mtkd_mean = (succ_abs_mtkd + succ_bad_mtkd) / 2.0
            
            # Take the harmonic mean of the two models' scores to find the sample where BOTH models perform most typically
            sample_joint_score = (2 * score_mt * score_mtkd) / (score_mt + score_mtkd + 1e-8)
            sample_joint_score_mean = (score_mt_mean + score_mtkd_mean) / 2.0
            
            sample_metrics.append({
                'index': idx,
                'harmonic_score': sample_joint_score,
                'harmonic_score_mt': score_mt,
                'harmonic_score_mtkd': score_mtkd,
                'mean_score_mt': score_mt_mean,
                'mean_score_mtkd': score_mtkd_mean,
                'mean_score': sample_joint_score_mean,
                'absrel_mt': mt_absrel,
                'bad3_mt': mt_bad3,
                'absrel_mtkd': mtkd_absrel,
                'bad3_mtkd': mtkd_bad3,
                'left_image': left_images[0].cpu(),
                'right_image': right_images[0].cpu() if right_images is not None else None,
                'target_disp': targets_disp[0].cpu(),
                'mt_disp': mt_disp[0].cpu(),
                'mtkd_disp': mtkd_disp[0].cpu(),
                'baseline': baseline[0].cpu(),
                'focal_length': focal_length[0].cpu()
            })

    # Sort samples to find the median
    sorted_samples = sorted(sample_metrics, key=lambda x: x['harmonic_score'])
    median_sample = sorted_samples[(len(sorted_samples) - 1) // 2]
    print(f"Median Sample Index: {median_sample['index']} with harmonic score: {median_sample['harmonic_score']:.4f}")
    
    sorted_samples_mean = sorted(sample_metrics, key=lambda x: x['mean_score'])
    median_sample_mean = sorted_samples_mean[(len(sorted_samples_mean) - 1) // 2]
    print(f"Median Sample Index (Mean Score): {median_sample_mean['index']} with mean score: {median_sample_mean['mean_score']:.4f}")
else:
    print(f"Using pre-selected Sample Index: {chosen_idx}")
    dataset_sample = dataset_test[chosen_idx]
    
    with torch.no_grad():
        left_images = dataset_sample['image'].unsqueeze(0).to(device)
        right_images = dataset_sample.get('right_image').unsqueeze(0).to(device) if 'right_image' in dataset_sample else None
        
        targets_disp = dataset_sample.get('disparity')
        targets_disp = targets_disp.unsqueeze(0).to(device) if targets_disp is not None else None
        
        baseline = dataset_sample.get('baseline')
        baseline = baseline.unsqueeze(0).to(device) if baseline is not None else None
        
        focal_length = dataset_sample.get('focal_length')
        focal_length = focal_length.unsqueeze(0).to(device) if focal_length is not None else None
        
        # MT
        outputs_mt = model_mt(left_images, right_images)
        mt_disp = outputs_mt['disparity']
        absrel_metric.reset()
        absrel_metric.update(mt_disp, targets_disp, baseline, focal_length)
        mt_absrel = absrel_metric.compute()['AbsRel_rate']
        bad3_metric.reset()
        bad3_metric.update(mt_disp, targets_disp, baseline, focal_length)
        mt_bad3 = bad3_metric.compute()['Bad3_rate']
        
        # MT-KD
        outputs_mtkd = model_mtkd(left_images, right_images)
        mtkd_disp = outputs_mtkd['disparity']
        absrel_metric.reset()
        absrel_metric.update(mtkd_disp, targets_disp, baseline, focal_length)
        mtkd_absrel = absrel_metric.compute()['AbsRel_rate']
        bad3_metric.reset()
        bad3_metric.update(mtkd_disp, targets_disp, baseline, focal_length)
        mtkd_bad3 = bad3_metric.compute()['Bad3_rate']
        
        # Harmonic Mean of success rates
        succ_abs_mt = max(0.0, 1.0 - mt_absrel)
        succ_bad_mt = max(0.0, 1.0 - mt_bad3)
        score_mt = (2 * succ_abs_mt * succ_bad_mt) / (succ_abs_mt + succ_bad_mt + 1e-8)
        
        succ_abs_mtkd = max(0.0, 1.0 - mtkd_absrel)
        succ_bad_mtkd = max(0.0, 1.0 - mtkd_bad3)
        score_mtkd = (2 * succ_abs_mtkd * succ_bad_mtkd) / (succ_abs_mtkd + succ_bad_mtkd + 1e-8)
        
        # Take the harmonic mean of the two models' scores to find the sample where BOTH models perform most typically
        sample_joint_score = (2 * score_mt * score_mtkd) / (score_mt + score_mtkd + 1e-8)
        median_sample = {
            'index': chosen_idx,
            'harmonic_score': sample_joint_score,
            'harmonic_score_mt': score_mt,
            'harmonic_score_mtkd': score_mtkd,
            'absrel_mt': mt_absrel,
            'bad3_mt': mt_bad3,
            'absrel_mtkd': mtkd_absrel,
            'bad3_mtkd': mtkd_bad3,
            'left_image': left_images[0].cpu(),
            'right_image': right_images[0].cpu() if right_images is not None else None,
            'target_disp': targets_disp[0].cpu(),
            'mt_disp': mt_disp[0].cpu(),
            'mtkd_disp': mtkd_disp[0].cpu(),
            'baseline': baseline[0].cpu(),
            'focal_length': focal_length[0].cpu()
        }

# %%
# Recompute depths and error maps for the median sample
import cv2
import matplotlib.pyplot as plt

max_disparity = 512.0
max_depth_viz = 150.0
error_alpha = 1.0

target_disp = median_sample['target_disp'].squeeze().numpy() * max_disparity
f = median_sample['focal_length'].item()
B = median_sample['baseline'].item()

rgb_img = prepare_rgb(median_sample['left_image'])
right_img_rgb = prepare_rgb(median_sample['right_image']) if median_sample['right_image'] is not None else np.full_like(rgb_img, 255)

# GT Depth
target_depth = np.divide(f * B, target_disp, out=np.zeros_like(target_disp), where=(target_disp > 0))
target_depth[target_disp <= 0] = np.nan
valid_depth = target_disp > 0

def get_depth_viz_rgb(depth, max_val):
    d_norm = np.clip(depth, 0, max_val) / max_val
    cmap = plt.get_cmap('magma_r')
    rgba = cmap(d_norm)
    rgba[np.isnan(depth)] = [1.0, 1.0, 1.0, 1.0]
    return (rgba[..., :3] * 255).astype(np.uint8)

target_depth_viz = get_depth_viz_rgb(target_depth, max_depth_viz)

def get_error_viz(pred_disp_raw, target_disp_raw, f, B, rgb_img):
    pred_disp = pred_disp_raw.squeeze().numpy() * max_disparity
    target_disp = target_disp_raw.squeeze().numpy() * max_disparity
    
    pred_depth = np.divide(f * B, pred_disp, out=np.zeros_like(pred_disp), where=(pred_disp > 0))
    pred_depth[pred_disp <= 0] = np.nan
    
    valid = target_disp > 0
    
    # Absolute Depth Error (Replacing relative AbsRel calculation for consistent comparability)
    depth_err_abs = np.zeros_like(target_depth)
    depth_err_abs[valid] = np.abs(pred_depth[valid] - target_depth[valid])
    depth_err_abs[np.isnan(depth_err_abs)] = 0.0
    depth_err_scaled = np.clip(depth_err_abs / max_depth_viz, 0, 1.0) # scaled by strict max depth
    depth_color = error_colormap(depth_err_scaled)
    
    blended_absrel = np.full_like(rgb_img, 255) # Reusing variable name for compatibility below
    blended_absrel[valid] = (rgb_img[valid] * (1 - error_alpha) + depth_color[valid] * error_alpha).astype(np.uint8)
    
    # Bad3 (Strict binary visualization > 3px)
    bad3_err = np.zeros_like(target_disp)
    bad3_err[valid] = np.where(np.abs(pred_disp[valid] - target_disp[valid]) > 3.0, 1.0, 0.0) # 1.0 = Red, 0.0 = Green
    bad3_color = error_colormap(bad3_err)
    
    blended_bad3 = np.full_like(rgb_img, 255)
    blended_bad3[valid] = (rgb_img[valid] * (1 - error_alpha) + bad3_color[valid] * error_alpha).astype(np.uint8)
    
    pred_depth_viz = get_depth_viz_rgb(pred_depth, max_depth_viz)
    
    return pred_depth_viz, blended_bad3, blended_absrel

# MT
mt_depth_viz, mt_bad3_img, mt_absrel_img = get_error_viz(median_sample['mt_disp'], median_sample['target_disp'], f, B, rgb_img)

# MT-KD
mtkd_depth_viz, mtkd_bad3_img, mtkd_absrel_img = get_error_viz(median_sample['mtkd_disp'], median_sample['target_disp'], f, B, rgb_img)

# Plotting
fig = make_subplots(
    rows=3, cols=3, column_widths=[1, 1, 1],
    shared_xaxes=True, shared_yaxes=True,
    vertical_spacing=0.01, horizontal_spacing=0.01
)

# Row 1 (Ground Truth / Inputs)
fig.add_trace(go.Image(z=target_depth_viz), row=1, col=1)
fig.add_trace(go.Image(z=rgb_img), row=1, col=2)
fig.add_trace(go.Image(z=right_img_rgb), row=1, col=3)

# Row 2 (MT)
fig.add_trace(go.Image(z=mt_depth_viz), row=2, col=1)
fig.add_trace(go.Image(z=mt_bad3_img), row=2, col=2)
fig.add_trace(go.Image(z=mt_absrel_img), row=2, col=3)

# Row 3 (MT-KD)
fig.add_trace(go.Image(z=mtkd_depth_viz), row=3, col=1)
fig.add_trace(go.Image(z=mtkd_bad3_img), row=3, col=2)
fig.add_trace(go.Image(z=mtkd_absrel_img), row=3, col=3)

fig.update_layout(
    height=800, width=1200,
    margin=dict(l=100, r=40, t=80, b=140),
    plot_bgcolor='white', paper_bgcolor='white'
)
fig.update_xaxes(showticklabels=False, visible=False)
fig.update_yaxes(showticklabels=False, visible=False)

for idx in range(1, 10):
    axis_suffix = "" if idx == 1 else str(idx)
    if f"yaxis{axis_suffix}" in fig.layout:
        fig.layout[f"yaxis{axis_suffix}"].update(scaleanchor=f"x{axis_suffix}", scaleratio=1)

fig.add_annotation(
    text=f"<span style='font-size: 14px'><b>MT</b><br>(AbsRel: {median_sample['absrel_mt']*100:.2f}%, Bad3: {median_sample['bad3_mt']*100:.2f}%)</span>",
    xref="paper", yref="paper", x=-0.09, y=0.5, textangle=-90, showarrow=False, font=dict(size=16)
)
fig.add_annotation(
    text=f"<span style='font-size: 14px'><b>MT-KD</b><br>(AbsRel: {median_sample['absrel_mtkd']*100:.2f}%, Bad3: {median_sample['bad3_mtkd']*100:.2f}%)</span>",
    xref="paper", yref="paper", x=-0.09, y=0, textangle=-90, showarrow=False, font=dict(size=16)
)

# New Layout Annotations like H10
fig.add_annotation(
    text="<b>Ground Truth</b>", xref="paper", yref="paper", x=-0.06, y=0.93,
    textangle=-90, showarrow=False, font=dict(size=22)
)

fig.add_annotation(
    text="<b>Depth</b>", xref="x domain", yref="paper", x=0.5, y=1.03,
    showarrow=False, font=dict(size=16)
)
fig.add_annotation(
    text="<b>Left Image</b>", xref="x2 domain", yref="paper", x=0.5, y=1.03,
    showarrow=False, font=dict(size=16)
)

fig.add_annotation(
    text="<b>Bad3 Map</b>", xref="x2 domain", yref="paper", x=0.5, y=0.7,
    showarrow=False, font=dict(size=16)
)

fig.add_annotation(
    text="<b>Right Image</b>", xref="x3 domain", yref="paper", x=0.5, y=1.03,
    showarrow=False, font=dict(size=16)
)

fig.add_annotation(
    text="<b>Error Map</b>", xref="x3 domain", yref="paper", x=0.5, y=0.7,
    showarrow=False, font=dict(size=16)
)

# Dummy traces for colormaps
fig.add_trace(go.Heatmap(
    z=[[0]], opacity=0, zmin=0, zmax=max_depth_viz,
    colorscale='Magma_r', showscale=True,
    colorbar=dict(title=dict(text='<b>Depth [mm]</b>', side='top', font=dict(size=18)), orientation='h', x=0.5, len=1.0, y=0.0, yanchor='top', xanchor='center', thickness=15)
), row=1, col=1)

fig.add_trace(go.Heatmap(
    z=[[0]], opacity=0, zmin=0, zmax=1.0,
    colorscale=[[0.0, 'rgb(0,200,0)'], [0.5, 'rgb(255,255,0)'], [1.0, 'rgb(255,0,0)']], 
    showscale=True,
    colorbar=dict(title=dict(text='<b>Error</b>', side='top', font=dict(size=18)), orientation='h', x=0.5, len=1.0, y=-0.13, yanchor='top', xanchor='center', thickness=15)
), row=2, col=3)

fig.update_traces(colorbar_tickfont_size=14, colorbar_title_font_size=16, selector=dict(type='heatmap'))

save_figure(fig, height=780, name='H04F03', lrtb_margin=(50, 20, 20, 40), folder='results', skip_sync=skip_sync)

# %%
