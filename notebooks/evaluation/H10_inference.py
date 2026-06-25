# * Inference Images and Pointclouds for relevant experiments
# * Best ST, MT and MT-KD seed with worst test sample vs. best test sample
# * Best Exp10 seed with worst test sample vs. best test sample

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

# %% Find best seeds
# Filter out exp10 for standard benchmarks
df_bench = df_final[df_final['experiment'] != 'exp10'].copy()

# Keep exp10 separately for H10F04
df_exp10 = df_final[df_final['experiment'] == 'exp10'].copy()

# Metric Columns
col_dice = 'metric.best_combined/performance/testing/segmentation/DICE_score/instrument_mean'
col_dice_fallback = 'metric.best_segmentation/performance/testing/segmentation/DICE_score/instrument_mean'

col_absrel = 'metric.best_combined/performance/testing/disparity/AbsRel_rate'
col_absrel_fallback = 'metric.best_disparity/performance/testing/disparity/AbsRel_rate'

# Merge fallback metrics if the combined ones are NaN (For bench)
df_bench['DICE'] = df_bench[col_dice].fillna(df_bench[col_dice_fallback])
df_bench['AbsRel'] = df_bench[col_absrel].fillna(df_bench[col_absrel_fallback])

# Normalize to 0-1 range (they are saved as percentages in MLflow results)
dice_norm = df_bench['DICE'] / 100.0
absrel_norm = df_bench['AbsRel'] / 100.0

# Calculate combined heuristic for MT / MT-KD ranking (Harmonic Mean)
absrel_clamped = (1.0 - absrel_norm).clip(0, 1)
df_bench['SCORE'] = (2 * dice_norm * absrel_clamped) / (dice_norm + absrel_clamped) * 100.0

# Do the same for exp10
df_exp10['DICE'] = df_exp10[col_dice].fillna(df_exp10[col_dice_fallback])
df_exp10['AbsRel'] = df_exp10[col_absrel].fillna(df_exp10[col_absrel_fallback])
dice_norm_10 = df_exp10['DICE'] / 100.0
absrel_norm_10 = df_exp10['AbsRel'] / 100.0
absrel_clamped_10 = (1.0 - absrel_norm_10).clip(0, 1)
df_exp10['SCORE'] = (2 * dice_norm_10 * absrel_clamped_10) / (dice_norm_10 + absrel_clamped_10) * 100.0

best_runs = {}
best_runs_exp10 = {}

# Best ST SEG (Median DICE)
df_seg = df_bench[(df_bench['config'] == 'SEG') & (df_bench['DICE'].notna())].sort_values('DICE')
best_runs['ST-SEG'] = df_seg.iloc[(len(df_seg) - 1) // 2]

df_seg_10 = df_exp10[(df_exp10['config'] == 'SEG') & (df_exp10['DICE'].notna())].sort_values('DICE')
best_runs_exp10['ST-SEG'] = df_seg_10.iloc[(len(df_seg_10) - 1) // 2]

# Best ST DISP (Median AbsRel)
df_disp = df_bench[(df_bench['config'] == 'DISP') & (df_bench['AbsRel'].notna())].sort_values('AbsRel')
best_runs['ST-DISP'] = df_disp.iloc[(len(df_disp) - 1) // 2]

df_disp_10 = df_exp10[(df_exp10['config'] == 'DISP') & (df_exp10['AbsRel'].notna())].sort_values('AbsRel')
# If exp10 doesn't have a DISP run (which is true based on the configs), fallback to the globally best ST-DISP
if not df_disp_10.empty:
    best_runs_exp10['ST-DISP'] = df_disp_10.iloc[(len(df_disp_10) - 1) // 2]
else:
    best_runs_exp10['ST-DISP'] = best_runs['ST-DISP']

# Calculate ST combined SCORE
st_dice_norm = best_runs['ST-SEG']['DICE'] / 100.0
st_absrel_clamped = max(0, 1.0 - (best_runs['ST-DISP']['AbsRel'] / 100.0))
st_combined_score = 0.0 if (st_dice_norm + st_absrel_clamped) == 0 else (2 * st_dice_norm * st_absrel_clamped) / (st_dice_norm + st_absrel_clamped) * 100.0

st_dice_norm_10 = best_runs_exp10['ST-SEG']['DICE'] / 100.0
st_absrel_clamped_10 = max(0, 1.0 - (best_runs_exp10['ST-DISP']['AbsRel'] / 100.0))
st_combined_score_10 = 0.0 if (st_dice_norm_10 + st_absrel_clamped_10) == 0 else (2 * st_dice_norm_10 * st_absrel_clamped_10) / (st_dice_norm_10 + st_absrel_clamped_10) * 100.0

# Best MT (Median SCORE)
df_mt = df_bench[(df_bench['config'] == 'MT') & (df_bench['SCORE'].notna())].sort_values('SCORE')
best_runs['MT'] = df_mt.iloc[(len(df_mt) - 1) // 2]

df_mt_10 = df_exp10[(df_exp10['config'] == 'MT') & (df_exp10['SCORE'].notna())].sort_values('SCORE')
best_runs_exp10['MT'] = df_mt_10.iloc[(len(df_mt_10) - 1) // 2]

# Best MT-KD (Median SCORE)
df_mt_kd = df_bench[(df_bench['config'] == 'MT-KD') & (df_bench['SCORE'].notna())].sort_values('SCORE')
best_runs['MT-KD'] = df_mt_kd.iloc[(len(df_mt_kd) - 1) // 2]

df_mt_kd_10 = df_exp10[(df_exp10['config'] == 'MT-KD') & (df_exp10['SCORE'].notna())].sort_values('SCORE')
best_runs_exp10['MT-KD'] = df_mt_kd_10.iloc[(len(df_mt_kd_10) - 1) // 2]

print("--- Best ST Pipeline (ST-SEG + ST-DISP) ---")
print(f"ST-SEG Run Name: {best_runs['ST-SEG'].get('run_name', 'N/A')} (Exp: {best_runs['ST-SEG'].get('experiment', 'N/A')})")
print(f"ST-DISP Run Name: {best_runs['ST-DISP'].get('run_name', 'N/A')} (Exp: {best_runs['ST-DISP'].get('experiment', 'N/A')})")
print(f"DICE: {best_runs['ST-SEG']['DICE']:.4f} | AbsRel: {best_runs['ST-DISP']['AbsRel']:.4f} | SCORE: {st_combined_score:.4f}\n")

for name in ['MT', 'MT-KD']:
    row = best_runs[name]
    print(f"--- Best {name} ---")
    print(f"Experiment: {row.get('experiment', 'N/A')} | Run Name: {row.get('run_name', 'N/A')}")
    print(f"DICE: {row['DICE']:.4f} | AbsRel: {row['AbsRel']:.4f} | SCORE: {row['SCORE']:.4f}\n")

# %% Load Models and Evaluate Sample Metrics
import yaml
import torch
import mlflow
from torch.utils.data import DataLoader
from metrics.segmentation import Dice
from metrics.disparity import AbsRel

device = torch.device('cuda')

def load_model_from_run(experiment, config_name, run_name, task_mode):
    mlflow.set_tracking_uri(f'/data/Zeitler/code/SIDE/mlruns_experiments/{experiment}')
    mlflow_experiment = mlflow.get_experiment_by_name(config_name)
    
    # The df contains the test run (e.g. '.../test'), but artifacts are saved in the train run
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
    
    # Also load config for data loading
    base_config_filepath = './configs/base.yaml'
    experiment_config_filepath = f'./configs/{experiment}/{config_name}.yaml'
    
    with open(base_config_filepath, 'r') as f: base_config = yaml.safe_load(f)
    with open(experiment_config_filepath, 'r') as f: experiment_config = yaml.safe_load(f)
    from utils import helpers
    cfg = helpers.deep_merge(experiment_config, base_config)
    
    return model, cfg

# Collect configs for the 4 setups we care about:
setups = {
    'ST-SEG': {'exp': best_runs['ST-SEG']['experiment'], 'config': 'SEG', 'run_name': best_runs['ST-SEG']['run_name'], 'task_mode': 'segmentation'},
    'ST-DISP': {'exp': best_runs['ST-DISP']['experiment'], 'config': 'DISP', 'run_name': best_runs['ST-DISP']['run_name'], 'task_mode': 'disparity'},
    'MT': {'exp': best_runs['MT']['experiment'], 'config': 'MT', 'run_name': best_runs['MT']['run_name'], 'task_mode': 'combined'},
    'MT-KD': {'exp': best_runs['MT-KD']['experiment'], 'config': 'MT-KD', 'run_name': best_runs['MT-KD']['run_name'], 'task_mode': 'combined'}
}

# Dictionary to hold the sample-level evaluations for each setup
evaluation_results = {}

testing_setups = ['ST-PIPELINE', 'MT', 'MT-KD']

if False:
    # Load MT config once to use its dataloader for ST-PIPELINE (since it provides both GTs)
    _, cfg_mt = load_model_from_run(setups['MT']['exp'], setups['MT']['config'], setups['MT']['run_name'], setups['MT']['task_mode'])

    for setup_key in testing_setups:
        print(f"\nEvaluating {setup_key} ...")
        
        if setup_key == 'ST-PIPELINE':
            model_seg, _ = load_model_from_run(setups['ST-SEG']['exp'], setups['ST-SEG']['config'], setups['ST-SEG']['run_name'], setups['ST-SEG']['task_mode'])
            model_disp, _ = load_model_from_run(setups['ST-DISP']['exp'], setups['ST-DISP']['config'], setups['ST-DISP']['run_name'], setups['ST-DISP']['task_mode'])
            cfg = cfg_mt
        else:
            info = setups[setup_key]
            model, cfg = load_model_from_run(info['exp'], info['config'], info['run_name'], info['task_mode'])
        
        # Load Dataloader
        from utils import helpers
        from data.transforms import build_transforms
        dataset_class = helpers.load(cfg['data']['dataset'])
        test_transforms = build_transforms(cfg, mode='test')
        
        dataset_test = dataset_class(
            mode='test',
            config=cfg,
            transforms=test_transforms,
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=cfg['general']['num_workers'],
            pin_memory=cfg['general']['pin_memory'],
            persistent_workers=False
        )
        
        # Set up metrics
        dice_metric = Dice(n_classes=cfg['data']['num_of_classes']['segmentation'], device=device)
        absrel_metric = AbsRel(max_disparity=cfg['data']['max_disparity'], device=device)
        
        sample_metrics = []
        
        with torch.no_grad():
            for idx, data_batch in enumerate(dataloader_test):
                left_images = data_batch['image'].to(device)
                right_images = data_batch['right_image'].to(device) if 'right_image' in data_batch else None
                
                targets_seg = data_batch.get('segmentation')
                targets_seg = targets_seg.to(device) if targets_seg is not None else None
                
                targets_disp = data_batch.get('disparity')
                targets_disp = targets_disp.to(device) if targets_disp is not None else None
                
                baseline = data_batch.get('baseline')
                baseline = baseline.to(device) if baseline is not None else None
                
                focal_length = data_batch.get('focal_length')
                focal_length = focal_length.to(device) if focal_length is not None else None
                
                if setup_key == 'ST-PIPELINE':
                    out_seg = model_seg(left_images, right_images)
                    out_disp = model_disp(left_images, right_images)
                    outputs = {
                        'segmentation': out_seg['segmentation'],
                        'disparity': out_disp['disparity']
                    }
                else:
                    outputs = model(left_images, right_images)
                
                # --- Segmentation ---
                val_dice = 0.0
                if 'segmentation' in outputs:
                    dice_metric.reset()
                    dice_metric.update(outputs['segmentation'], targets_seg)
                    dice_dict = dice_metric.compute() 
                    
                    valid_dices = [d for class_idx, d in dice_dict.items() if class_idx >= 1]
                    if valid_dices:
                        mean_dice = sum(valid_dices) / len(valid_dices)
                        # Convert to float if it's a tensor
                        val_dice = mean_dice.item() if hasattr(mean_dice, 'item') else mean_dice
                    else:
                        val_dice = 0.0
                
                # --- Disparity ---
                absrel_val = 0.0
                if 'disparity' in outputs:
                    absrel_metric.reset()
                    absrel_metric.update(outputs['disparity'], targets_disp, baseline, focal_length)
                    absrel_val = absrel_metric.compute()['AbsRel_rate']
                
                # --- Score (Harmonic / Standard Mean) ---
                val_heuristic = 0.0
                val_mean = 0.0
                if setup_key in ['ST-PIPELINE', 'MT', 'MT-KD']:
                    import math
                    if math.isnan(absrel_val):
                        absrel_clamped = 0.0
                    else:
                        absrel_clamped = max(0.0, min(1.0, 1.0 - absrel_val))
                        
                    denominator = val_dice + absrel_clamped
                    val_heuristic = 0.0 if denominator == 0 else (2 * val_dice * absrel_clamped) / denominator
                    val_mean = (val_dice + absrel_clamped) / 2.0
                
                # Exclude samples that inherently lack the ground truth for an instrument
                has_instrument = (targets_seg > 0).sum() > 0 if targets_seg is not None else False
                
                if has_instrument:
                    sample_metrics.append({
                        'index': idx,
                        'dice': val_dice,
                        'absrel': absrel_val,
                        'score': val_heuristic,
                        'score_mean': val_mean
                    })
                
                if idx % 100 == 0:
                    print(f"  Processed {idx}/{len(dataloader_test)} samples")
                    
        # Sort samples to find best, worst and median based on standard mean (score_mean)
        #sorted_samples = sorted(sample_metrics, key=lambda x: x['score_mean'])
        sorted_samples = sorted(sample_metrics, key=lambda x: x['score'])
        worst = sorted_samples[0]
        best = sorted_samples[-1]
        median = sorted_samples[(len(sorted_samples) - 1) // 2]
        
        evaluation_results[setup_key] = {
            'all_metrics': sample_metrics,
            'worst': worst,
            'best': best,
            'median': median
        }
        
        print(f"Finished {setup_key}!")
        print(f"  Best  : Index {best['index']} | DICE: {best['dice']*100:.2f} | AbsRel: {best['absrel']*100:.2f} | Score (Mean): {best['score']*100:.2f}")
        print(f"  Median: Index {median['index']} | DICE: {median['dice']*100:.2f} | AbsRel: {median['absrel']*100:.2f} | Score (Mean): {median['score_mean']*100:.2f}")
        print(f"  Worst : Index {worst['index']} | DICE: {worst['dice']*100:.2f} | AbsRel: {worst['absrel']*100:.2f} | Score (Mean): {worst['score_mean']*100:.2f}")

# %%
# * Results
# Best model
# ST-PIPELINE!
# Best : Index 268 | DICE: 98.92 | AbsRel: 2.93 | Score (Mean): 97.99
# Worst: Index 536 | DICE: 3.03 | AbsRel: 27.52 | Score (Mean): 37.75
# Best : Index 268 | DICE: 98.92 | AbsRel: 2.93 | Score (HM): 97.99
# Worst: Index 72 | DICE: 0.11 | AbsRel: 5.18 | Score (HM): 0.23 

# MT!
# Best : Index 178 | DICE: 97.93 | AbsRel: 3.57 | Score (Mean): 97.17
# Worst: Index 536 | DICE: 21.61 | AbsRel: 35.70 | Score (Mean): 42.95
# Best : Index 178 | DICE: 97.93 | AbsRel: 3.57 | Score (HM): 97.17
# Worst: Index 244 | DICE: 0.27 | AbsRel: 3.47 | Score (HM): 0.54 

# MT-KD!
# Best : Index 81 | DICE: 98.50 | AbsRel: 3.11 | Score (Mean): 97.69
# Worst: Index 352 | DICE: 14.64 | AbsRel: 34.80 | Score (Mean): 39.92
# Median: Index 417 | DICE: 74.73 | AbsRel: 7.33 | Score (Mean): 83.70
# Best : Index 81 | DICE: 98.50 | AbsRel: 3.11 | Score (HM): 97.69
# Worst: Index 162 | DICE: 0.17 | AbsRel: 3.19 | Score (HM): 0.35
# Median: Index 429 | DICE: 75.17 | AbsRel: 6.82 | Score (HM): 84.18


# Median Model
# ST-PIPELINE!
# Best  : Index 35 | DICE: 98.49 | AbsRel: 47.66 | Score (HM): 68.36
# Median: Index 403 | DICE: 62.51 | AbsRel: 51.20 | Score (HM): 55.66
# Worst : Index 447 | DICE: 0.45 | AbsRel: 53.10 | Score (HM): 23.67

# MT!
# Best  : Index 32 | DICE: 98.43 | AbsRel: 5.74 | Score (HM): 96.30
# Median: Index 368 | DICE: 55.02 | AbsRel: 16.85 | Score (HM): 69.08
# Worst : Index 550 | DICE: 0.22 | AbsRel: 17.15 | Score (HM): 41.54

# MT-KD!
# Best  : Index 81 | DICE: 98.73 | AbsRel: 2.73 | Score (HM): 97.99
# Median: Index 449 | DICE: 67.17 | AbsRel: 7.26 | Score (HM): 79.96
# Worst : Index 311 | DICE: 0.21 | AbsRel: 6.14 | Score (HM): 47.04


# Hardcode the metrics so we don't have to rerun the long loop every time while developing plotting
if not evaluation_results:
    evaluation_results = {
        'ST-PIPELINE': {
            'best': {'index': 35, 'dice': 98.49, 'absrel': 47.66, 'score': 68.36},
            'worst': {'index': 447, 'dice': 0.45, 'absrel': 53.10, 'score': 23.67},
            'median': {'index': 403, 'dice': 62.51, 'absrel': 51.20, 'score': 55.66}
        },
        'MT': {
            'best': {'index': 32, 'dice': 98.43, 'absrel': 5.74, 'score': 96.30},
            'worst': {'index': 550, 'dice': 0.22, 'absrel': 17.15, 'score': 41.54},
            'median': {'index': 368, 'dice': 55.02, 'absrel': 16.85, 'score': 69.08}
        },
        'MT-KD': {
            'best': {'index': 81, 'dice': 98.73, 'absrel': 2.73, 'score': 97.99},
            'worst': {'index': 311, 'dice': 0.21, 'absrel': 6.14, 'score': 47.04},
            'median': {'index': 449, 'dice': 67.17, 'absrel': 7.26, 'score': 79.96}
        }
    }

# %% Plot Rendering
import cv2

# Settings for Chart Config integration
CHART_CONFIG = {
    'H10_ST-PIPELINE': {},
    'H10_MT': {},
    'H10_MT-KD': {}
}

# Visualization Settings
max_disparity = 512.0
max_depth_viz = 150.0
seg_alpha = 0.75
error_alpha = 1.0
ERROR_GREEN = np.array([0, 200, 0])
ERROR_YELLOW = np.array([255, 255, 0])
ERROR_RED = np.array([255, 0, 0])

from plotly.colors import hex_to_rgb

# Manually specify which Plotly color index to use for each instrument (1-7)
# Choose from: Dark24 (24 colors), Light24 (24 colors), Alphabet (26 colors), 
# Pastel (11 colors), Set1/Set2/Set3 (9 colors each), Plotly (10 colors)
PLOTLY_PALETTE = "Light24"  # Change this to use a different palette
INSTRUMENT_COLOR_MAP = {
    1: 9,   # Bipolar Forceps
    2: 7,   # Prograsp Forceps
    3: 2,   # Large Needle Driver
    4: 3,   # Vessel Sealer
    5: 4,   # Grasping Retractor
    6: 5,   # Monopolar Curved Scissors
    7: 6,   # Other
}

# Build CLASS_COLORS using the manual mapping
plotly_colors = getattr(px.colors.qualitative, PLOTLY_PALETTE)
CLASS_COLORS = {
    instrument_id: np.array(hex_to_rgb(plotly_colors[color_idx]))
    for instrument_id, color_idx in INSTRUMENT_COLOR_MAP.items()
}

def error_colormap(error_map):
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
    img = np.array(img_tensor)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.moveaxis(img, 0, -1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return (img * 255).astype(np.uint8)

def apply_seg_overlay(image, mask, alpha=0.5, class_colors=None):
    if class_colors is None:
        class_colors = CLASS_COLORS
    overlay = image.copy()
    mask = np.squeeze(mask)
    for class_id, color in class_colors.items():
        valid = (mask == class_id)
        if valid.any():
            overlay[valid] = overlay[valid] * (1 - alpha) + color * alpha
    return overlay

# Load MT config once to use its dataloader (since it provides both GTs)
_, cfg_mt = load_model_from_run(setups['MT']['exp'], 'MT', setups['MT']['run_name'], 'combined')
from utils import helpers
from data.transforms import build_transforms
test_transforms = build_transforms(cfg_mt, mode='test')
dataset_class = helpers.load(cfg_mt['data']['dataset'])
dataset_test = dataset_class(mode='test', config=cfg_mt, transforms=test_transforms)

if 'all_plot_data' not in globals():
    all_plot_data = {}

for setup_key in ['ST-PIPELINE', 'MT', 'MT-KD']:
    if setup_key in all_plot_data:
        print(f"Skipping inference for {setup_key}, already cached.")
        continue
    
    print(f"Running inference for {setup_key}...")
    
    # 1. Load the corresponding model(s)
    if setup_key == 'ST-PIPELINE':
        model_seg, _ = load_model_from_run(setups['ST-SEG']['exp'], 'SEG', setups['ST-SEG']['run_name'], 'segmentation')
        model_disp, _ = load_model_from_run(setups['ST-DISP']['exp'], 'DISP', setups['ST-DISP']['run_name'], 'disparity')
    else:
        info = setups[setup_key]
        model, _ = load_model_from_run(info['exp'], info['config'], info['run_name'], info['task_mode'])
        
    worst_idx = evaluation_results[setup_key]['worst']['index']
    best_idx = evaluation_results[setup_key]['best']['index']
    
    # We use the MT-KD median index for all pipelines for H10F04
    median_idx = evaluation_results['MT-KD']['median']['index']
    
    # Pre-fetch the sample data dicts
    plot_data = {}
    for i, (sample_type, data_idx) in enumerate([('BestSample', best_idx), ('WorstSample', worst_idx), ('MedianSample', median_idx)]):
        dataset_sample = dataset_test[data_idx]
        left_image = dataset_sample['image'].unsqueeze(0).to(device)
        right_image = dataset_sample['right_image'].unsqueeze(0).to(device) if 'right_image' in dataset_sample else None
        
        target_segmentation = dataset_sample.get('segmentation')
        target_segmentation = target_segmentation.unsqueeze(0).to(device) if target_segmentation is not None else None
        
        target_disparity = dataset_sample.get('disparity')
        target_disparity = target_disparity.unsqueeze(0).to(device) if target_disparity is not None else None
        
        baseline = dataset_sample.get('baseline')
        baseline = baseline.unsqueeze(0).to(device) if baseline is not None else None
        
        focal_length = dataset_sample.get('focal_length')
        focal_length = focal_length.unsqueeze(0).to(device) if focal_length is not None else None
        
        with torch.no_grad():
            if setup_key == 'ST-PIPELINE':
                out_seg = model_seg(left_image, right_image)
                out_disp = model_disp(left_image, right_image)
                output = {'segmentation': out_seg['segmentation'], 'disparity': out_disp['disparity']}
            else:
                output = model(left_image, right_image)
            
        plot_data[i] = {
            'left_image': left_image.squeeze(0).cpu().numpy(),
            'right_image': right_image.squeeze(0).cpu().numpy() if right_image is not None else None,
            'target_segmentation': target_segmentation.squeeze(0).cpu().numpy() if target_segmentation is not None else None,
            'target_disparity': target_disparity.squeeze(0).cpu().numpy() if target_disparity is not None else None,
            'output_segmentation': output['segmentation'].squeeze(0).cpu().numpy(),
            'output_disparity': output['disparity'].squeeze(0).cpu().numpy(),
            'baseline': baseline.item() if baseline is not None else 1.0,
            'focal_length': focal_length.item() if focal_length is not None else 1.0,
            'type': sample_type,
            'metrics_dice': evaluation_results[setup_key].get(sample_type.lower().replace('sample', ''), {}).get('dice', 0.0),
            'metrics_absrel': evaluation_results[setup_key].get(sample_type.lower().replace('sample', ''), {}).get('absrel', 0.0),
        }
        
    all_plot_data[setup_key] = plot_data

# %% Format and Render Plots
if True:
    for setup_key, plot_data in all_plot_data.items():
        print(f"Generating plot for {setup_key}...")
        
        # 2. Setup the Plotly Figure
        fig = make_subplots(
            rows=6, cols=3, column_widths=[1, 1, 1],
            vertical_spacing=0.01, horizontal_spacing=0.0001,
            subplot_titles=("<b>Prediction</b>", "<b>Target</b>", "<b>Error Map</b>",
                            "", "", "",
                            "", "", "",
                            "", "", "",
                            "", "", "",
                            "", "", "")
        )
        
        # 3. Render loop equivalent to `inference_figure.py`
        for plot_row_idx, sample_idx in enumerate([0, 2, 1]): # 0=Best, 2=Median, 1=Worst
            row_offset = 1 + (plot_row_idx * 2) # Maps to rows 1, 3, 5
            sample = plot_data[sample_idx]
            rgb_img = prepare_rgb(sample['left_image'])
            f = sample['focal_length']
            B = sample['baseline']
            
            # --- Segmentation Row ---
            out_seg = sample['output_segmentation']
            if out_seg.ndim == 3 and out_seg.shape[0] < out_seg.shape[-1]: pred_seg_mask = out_seg.argmax(axis=0)
            elif out_seg.ndim == 3: pred_seg_mask = out_seg.argmax(axis=-1)
            else: pred_seg_mask = out_seg.squeeze()
                
            target_seg_mask = sample['target_segmentation'].squeeze()
            
            pred_seg_overlay = apply_seg_overlay(rgb_img, pred_seg_mask, seg_alpha)
            target_seg_overlay = apply_seg_overlay(rgb_img, target_seg_mask, seg_alpha)
            
            seg_match = (pred_seg_mask == target_seg_mask)
            valid_mask = target_seg_mask != 255
            
            seg_error = np.zeros_like(target_seg_mask, dtype=np.float32)
            seg_error[valid_mask & ~seg_match] = 1.0 
            seg_err_color = error_colormap(seg_error)
            
            seg_err_img = np.full_like(rgb_img, 255)
            seg_err_img[valid_mask] = (rgb_img[valid_mask] * (1 - error_alpha) + seg_err_color[valid_mask] * error_alpha).astype(np.uint8)
            
            # Anti-aliasing downscale before passing to Plotly (Kaleido uses Nearest-Neighbor which causes zig-zags)
            render_size = (340, 340) # Approx size of the subplot to prevent browser/kaleido aliasing
            pred_seg_overlay = cv2.resize(pred_seg_overlay, render_size, interpolation=cv2.INTER_AREA)
            target_seg_overlay = cv2.resize(target_seg_overlay, render_size, interpolation=cv2.INTER_AREA)
            seg_err_img = cv2.resize(seg_err_img, render_size, interpolation=cv2.INTER_AREA)

            fig.add_trace(go.Image(z=pred_seg_overlay), row=row_offset, col=1)
            fig.add_trace(go.Image(z=target_seg_overlay), row=row_offset, col=2)
            fig.add_trace(go.Image(z=seg_err_img), row=row_offset, col=3)
            
            # --- Depth Row ---
            pred_disp = sample['output_disparity'].squeeze() * max_disparity
            target_disp = sample['target_disparity'].squeeze() * max_disparity
            
            pred_depth = np.divide(f * B, pred_disp, out=np.zeros_like(pred_disp), where=(pred_disp > 0))
            target_depth = np.divide(f * B, target_disp, out=np.zeros_like(target_disp), where=(target_disp > 0))
            
            pred_depth[pred_disp <= 0] = np.nan
            target_depth[target_disp <= 0] = np.nan
            
            # Absolute Depth Error (Consistent Error limits)
            depth_err_abs = np.zeros_like(target_depth)
            valid_depth = target_depth > 0
            depth_err_abs[valid_depth] = np.abs(pred_depth[valid_depth] - target_depth[valid_depth])
            depth_err_abs[np.isnan(depth_err_abs)] = 0.0
            
            # Scale the error visualization so small errors are visible
            depth_err_scaled = np.clip(depth_err_abs / max_depth_viz, 0, 1.0)
            
            depth_err_color = error_colormap(depth_err_scaled)
            blended_depth_err = np.full_like(rgb_img, 255)
            blended_depth_err[valid_depth] = (rgb_img[valid_depth] * (1 - error_alpha) + depth_err_color[valid_depth] * error_alpha).astype(np.uint8)
            
            blended_depth_err = cv2.resize(blended_depth_err, render_size, interpolation=cv2.INTER_AREA)
            pred_depth_viz = cv2.resize(pred_depth, render_size, interpolation=cv2.INTER_NEAREST)[::-1]
            target_depth_viz = cv2.resize(target_depth, render_size, interpolation=cv2.INTER_NEAREST)[::-1]

            fig.add_trace(go.Heatmap(z=pred_depth_viz, colorscale='Magma_r', zmin=0, zmax=max_depth_viz, showscale=False), row=row_offset + 1, col=1)
            
            fig.add_trace(
                go.Heatmap(
                    z=target_depth_viz, colorscale='Magma_r', zmin=0, zmax=max_depth_viz, 
                    showscale=(sample_idx == 0), 
                    colorbar=dict(
                        title=dict(text="Depth [mm]", side="right", font=dict(size=18)), 
                        x=0.98, len=1.0, y=0.5, yanchor='middle', thickness=15
                    )
                ), 
                row=row_offset + 1, col=2
            )
            
            fig.add_trace(go.Image(z=blended_depth_err), row=row_offset + 1, col=3)
            
            green_red_scale = [
                [0.0, f'rgb({ERROR_GREEN[0]}, {ERROR_GREEN[1]}, {ERROR_GREEN[2]})'],
                [0.5, f'rgb({ERROR_YELLOW[0]}, {ERROR_YELLOW[1]}, {ERROR_YELLOW[2]})'],
                [1.0, f'rgb({ERROR_RED[0]}, {ERROR_RED[1]}, {ERROR_RED[2]})']
            ]
            fig.add_trace(
                go.Heatmap(
                    z=[[0]], opacity=0, colorscale=green_red_scale, zmin=0, zmax=1.0, 
                    showscale=(sample_idx == 0), 
                    colorbar=dict(
                        title=dict(text="Error", side="right", font=dict(size=18)), 
                        x=1.12, len=1.0, y=0.5, yanchor='middle', thickness=15
                    )
                ), 
                row=row_offset + 1, col=3
            )
            
        fig.update_xaxes(showticklabels=False, visible=False)
        fig.update_yaxes(showticklabels=False, visible=False)
        for i in range(1, 19): 
            axis_suffix = "" if i == 1 else str(i)
            fig.layout[f"yaxis{axis_suffix}"].update(scaleanchor=f"x{axis_suffix}", scaleratio=1)
            
        # --- Legend ---
        instrument_names = {
            1: "Bipolar Forceps",
            2: "Prograsp Forceps",
            3: "Large Needle Driver",
            4: "Vessel Sealer",
            5: "Grasping Retractor",
            6: "Monopolar Curved Scissors",
            7: "Other"
        }

        for class_id, color in CLASS_COLORS.items():
            if class_id in instrument_names:
                hex_color = f'rgb({color[0]}, {color[1]}, {color[2]})'
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(size=15, color=hex_color, symbol='square'),
                        name=instrument_names[class_id],
                        showlegend=True
                    )
                )
            
        # Thesis formatting
        fig.add_annotation(
            text=f"<span style='font-size: 14px'><b>Best Sample</b> (DICE: {plot_data[0]['metrics_dice']:.2f}%, AbsRel: {plot_data[0]['metrics_absrel']:.2f}%)</span>",
            xref="paper", yref="paper", x=-0.045, y=1.01, textangle=-90, showarrow=False, font=dict(size=16)
        )
        fig.add_annotation(
            text=f"<span style='font-size: 14px'><b>Median Sample</b> (DICE: {plot_data[2]['metrics_dice']:.2f}%, AbsRel: {plot_data[2]['metrics_absrel']:.2f}%)</span>",
            xref="paper", yref="paper", x=-0.045, y=0.5, textangle=-90, showarrow=False, font=dict(size=16)
        )
        fig.add_annotation(
            text=f"<span style='font-size: 14px'><b>Worst Sample</b> (DICE: {plot_data[1]['metrics_dice']:.2f}%, AbsRel: {plot_data[1]['metrics_absrel']:.2f}%)</span>",
            xref="paper", yref="paper", x=-0.045, y=-0.01, textangle=-90, showarrow=False, font=dict(size=16)
        )

        fig.update_layout(
            width=1000, 
            height=1725, 
            margin=dict(l=60, r=120, t=50, b=50),
            plot_bgcolor='white', 
            paper_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.02,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            )
            #title=dict(text=f"Inference Format: {setup_key}", x=0.5, xanchor='center', font=dict(size=24))
        )
        
        fig.update_traces(colorbar_tickfont_size=14, colorbar_title_font_size=16, selector=dict(type='heatmap'))
        
        apply_chart_config(fig, f'H10_{setup_key}', CHART_CONFIG)
        setup_f = {'ST-PIPELINE': 'F01', 'MT': 'F02', 'MT-KD': 'F03'}[setup_key]
        save_figure(fig, height=1050, name=f'H10{setup_f}', lrtb_margin=(20, 0, 20, 10), folder='results', skip_sync=False)
        # fig.show()
        # break

# %% H10F04 - Best MT-KD Sample of exp10 across ST, MT and MT-KD 
# Top row, the raw sample median pair (left image, right image)
# ST output (depth map with segmentation overlay (left), 3D Pointcloud Screenshot (right))
# MT output (same as ST)
# MT-KD output (same as ST)

# For H10F04, we specifically want to evaluate the best exp10 seeds and use the sample with the best mean score.
setups_exp10 = {
    'ST-PIPELINE_EXP10': {
        'seg': {'exp': best_runs_exp10['ST-SEG']['experiment'], 'config': 'SEG', 'run_name': best_runs_exp10['ST-SEG']['run_name'], 'task_mode': 'segmentation'},
        'disp': {'exp': best_runs_exp10['ST-DISP']['experiment'], 'config': 'DISP', 'run_name': best_runs_exp10['ST-DISP']['run_name'], 'task_mode': 'disparity'}
    },
    'MT_EXP10': {'exp': best_runs_exp10['MT']['experiment'], 'config': 'MT', 'run_name': best_runs_exp10['MT']['run_name'], 'task_mode': 'combined'},
    'MT-KD_EXP10': {'exp': best_runs_exp10['MT-KD']['experiment'], 'config': 'MT-KD', 'run_name': best_runs_exp10['MT-KD']['run_name'], 'task_mode': 'combined'}
}

evaluation_results_exp10 = {
    'MT-KD_EXP10': {
        'best': {'index': 12, 'score': 0.9809183767948502},
        'median': {'index': 506, 'score': 0.9610790237927049}
        }
}

# 1. Run evaluation to find the best mean score sample
if 'evaluation_results_exp10' not in globals():
    print("Evaluating exp10 setups to find the best mean score sample...")
    evaluation_results_exp10 = {}
    
    # Load MT_EXP10 config to use dataloader
    info = setups_exp10['MT_EXP10']
    _, cfg_exp10 = load_model_from_run(info['exp'], info['config'], info['run_name'], info['task_mode'])
    test_transforms_exp10 = build_transforms(cfg_exp10, mode='test')
    dataset_test_exp10 = dataset_class(mode='test', config=cfg_exp10, transforms=test_transforms_exp10)
    dataloader_test_exp10 = DataLoader(dataset_test_exp10, batch_size=1, shuffle=False, num_workers=cfg_exp10['general']['num_workers'], pin_memory=cfg_exp10['general']['pin_memory'])
    
    # We only need to find the best sample for MT-KD_EXP10 (or we could just use MT-KD_EXP10's best sample)
    info_mtkd = setups_exp10['MT-KD_EXP10']
    model_mtkd_exp10, _ = load_model_from_run(info_mtkd['exp'], info_mtkd['config'], info_mtkd['run_name'], info_mtkd['task_mode'])
    
    dice_metric = Dice(n_classes=cfg_exp10['data']['num_of_classes']['segmentation'], device=device)
    absrel_metric = AbsRel(max_disparity=cfg_exp10['data']['max_disparity'], device=device)
    
    sample_metrics_exp10 = []
    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader_test_exp10):
            left_images = data_batch['image'].to(device)
            right_images = data_batch.get('right_image').to(device) if 'right_image' in data_batch else None
            targets_seg = data_batch.get('segmentation')
            targets_seg = targets_seg.to(device) if targets_seg is not None else None
            targets_disp = data_batch.get('disparity')
            targets_disp = targets_disp.to(device) if targets_disp is not None else None
            baseline = data_batch.get('baseline')
            baseline = baseline.to(device) if baseline is not None else None
            focal_length = data_batch.get('focal_length')
            focal_length = focal_length.to(device) if focal_length is not None else None
            
            outputs = model_mtkd_exp10(left_images, right_images)
            
            val_dice = 0.0
            if 'segmentation' in outputs:
                dice_metric.reset()
                dice_metric.update(outputs['segmentation'], targets_seg)
                dice_dict = dice_metric.compute() 
                valid_dices = [d for class_idx, d in dice_dict.items() if class_idx >= 1]
                if valid_dices:
                    mean_dice = sum(valid_dices) / len(valid_dices)
                    # Convert to float if it's a tensor
                    val_dice = mean_dice.item() if hasattr(mean_dice, 'item') else mean_dice
                else:
                    val_dice = 0.0
            
            absrel_val = 0.0
            if 'disparity' in outputs:
                absrel_metric.reset()
                absrel_metric.update(outputs['disparity'], targets_disp, baseline, focal_length)
                absrel_val = absrel_metric.compute()['AbsRel_rate']
            
            import math
            if math.isnan(absrel_val):
                absrel_clamped = 0.0
            else:
                absrel_clamped = max(0.0, min(1.0, 1.0 - absrel_val))
                
            denominator = val_dice + absrel_clamped
            val_heuristic = 0.0 if denominator == 0 else (2 * val_dice * absrel_clamped) / denominator
            
            # Honestly check if ground truth is valid, rather than filtering by network score > 0
            has_instrument = (targets_seg > 0).sum() > 0 if targets_seg is not None else False
            
            # Keep sample if it inherently has instruments
            if has_instrument:
                sample_metrics_exp10.append({'index': idx, 'score': val_heuristic})
            
            if idx % 100 == 0:
                print(f"  exp10 Processed {idx}/{len(dataloader_test_exp10)} samples")
                
    sorted_samples_exp10 = sorted(sample_metrics_exp10, key=lambda x: x['score'])
    best_exp10_sample = sorted_samples_exp10[-1]
    median_exp10_sample = sorted_samples_exp10[(len(sorted_samples_exp10) - 1) // 2]
    evaluation_results_exp10['MT-KD_EXP10'] = {
        'best': best_exp10_sample,
        'median': median_exp10_sample
    }
    print(f"exp10 Best MT-KD Sample Index: {best_exp10_sample['index']}")
    print(f"exp10 Median MT-KD Sample Index: {median_exp10_sample['index']}")


# 2. Extract and Plot H10F04 using the median exp10 index
if 'evaluation_results_exp10' in globals():
    from notebooks.figures import helpers as temp_helpers
    import importlib
    importlib.reload(temp_helpers)
    from notebooks.figures.helpers import generate_pointcloud_screenshot
    from plotly.colors import hex_to_rgb
    import plotly.express as px
    import numpy as np
    
    # --- Custom Colors for exp10 ---
    EXP10_PLOTLY_PALETTE = "Light24"  # Change this to use a different palette
    EXP10_INSTRUMENT_COLOR_MAP = {
        1: 15,   # Change this index to select a different color (15 was chosen to not overlap with 1-7 configs above)
    }
    
    exp10_plotly_colors = getattr(px.colors.qualitative, EXP10_PLOTLY_PALETTE)
    EXP10_CLASS_COLORS = {
        instrument_id: np.array(hex_to_rgb(exp10_plotly_colors[color_idx]))
        for instrument_id, color_idx in EXP10_INSTRUMENT_COLOR_MAP.items()
    }
    
    print("Generating Figure H10F04...")
    
    # We explicitly take the Median Sample from the Best Model
    target_idx_exp10 = evaluation_results_exp10['MT-KD_EXP10']['median']['index']
    
    # Needs 4 rows: Raw Images (1), ST (2), MT (3), MT-KD (4)
    fig_best_exp10 = make_subplots(
        rows=4, cols=2,
        vertical_spacing=0.005, horizontal_spacing=0.01,
        specs=[[{"type": "image"}, {"type": "image"}],
               [{"type": "image"}, {"type": "image"}],
               [{"type": "image"}, {"type": "image"}],
               [{"type": "image"}, {"type": "image"}]]
    )
    
    info_mt = setups_exp10['MT_EXP10']
    _, cfg_exp10 = load_model_from_run(info_mt['exp'], info_mt['config'], info_mt['run_name'], info_mt['task_mode'])
    dataset_class_exp10 = helpers.load(cfg_exp10['data']['dataset'])
    test_transforms_exp10 = build_transforms(cfg_exp10, mode='test')
    dataset_test_exp10 = dataset_class_exp10(mode='test', config=cfg_exp10, transforms=test_transforms_exp10)
    
    target_sample_data = dataset_test_exp10[target_idx_exp10]
    left_img = prepare_rgb(target_sample_data['image'].numpy())
    raw_right = target_sample_data.get('right_image')
    
    # We will compute inference dynamically for this one sample instead of caching all
    f = target_sample_data.get('focal_length', torch.tensor([1.0])).item()
    B = target_sample_data.get('baseline', torch.tensor([1.0])).item()
    left_tensor = target_sample_data['image'].unsqueeze(0).to(device)
    right_tensor = raw_right.unsqueeze(0).to(device) if raw_right is not None else None

    target_seg = target_sample_data.get('segmentation').squeeze().numpy()
    target_disp = target_sample_data.get('disparity').squeeze().numpy() * max_disparity
    
    target_depth = np.divide(f * B, target_disp, out=np.zeros_like(target_disp), where=(target_disp > 0))
    target_depth[target_disp <= 0] = np.nan
    
    depth_viz_gt = np.clip(target_depth, 0, max_depth_viz) / max_depth_viz
    depth_gray_gt = (depth_viz_gt * 255).astype(np.uint8)
    depth_gray_gt = 255 - depth_gray_gt
    target_depth_colored = np.stack([depth_gray_gt]*3, axis=-1)
    
    target_seg_overlay = apply_seg_overlay(target_depth_colored, target_seg, seg_alpha, class_colors=EXP10_CLASS_COLORS)
    fig_best_exp10.add_trace(go.Image(z=target_seg_overlay), row=1, col=1)
    
    target_seg_rgb_overlay = apply_seg_overlay(left_img, target_seg, seg_alpha, class_colors=EXP10_CLASS_COLORS)
    target_pc_img = generate_pointcloud_screenshot(
        pred_disp=target_disp, rgb_image=target_seg_rgb_overlay, focal_length=f, baseline=B, 
        max_depth=300, denoise=False
    )
    fig_best_exp10.add_trace(go.Image(z=target_pc_img), row=1, col=2)

    # Iter over ST, MT, MT-KD
    models_to_run = [
        ('ST-PIPELINE_EXP10', setups_exp10['ST-PIPELINE_EXP10']),
        ('MT_EXP10', setups_exp10['MT_EXP10']),
        ('MT-KD_EXP10', setups_exp10['MT-KD_EXP10'])
    ]
    
    for row_idx, (setup_key, info) in enumerate(models_to_run, start=2):
        with torch.no_grad():
            if setup_key == 'ST-PIPELINE_EXP10':
                model_s, _ = load_model_from_run(info['seg']['exp'], info['seg']['config'], info['seg']['run_name'], info['seg']['task_mode'])
                model_d, _ = load_model_from_run(info['disp']['exp'], info['disp']['config'], info['disp']['run_name'], info['disp']['task_mode'])
                s_out = model_s(left_tensor, right_tensor)
                d_out = model_d(left_tensor, right_tensor)
                out_seg = s_out['segmentation'].squeeze(0).cpu().numpy()
                pred_disp = d_out['disparity'].squeeze(0).cpu().numpy().squeeze() * max_disparity
            else:
                model, _ = load_model_from_run(info['exp'], info['config'], info['run_name'], info['task_mode'])
                out = model(left_tensor, right_tensor)
                out_seg = out['segmentation'].squeeze(0).cpu().numpy()
                pred_disp = out['disparity'].squeeze(0).cpu().numpy().squeeze() * max_disparity
                
        if out_seg.ndim == 3 and out_seg.shape[0] < out_seg.shape[-1]: pred_seg_mask = out_seg.argmax(axis=0)
        elif out_seg.ndim == 3: pred_seg_mask = out_seg.argmax(axis=-1)
        else: pred_seg_mask = out_seg.squeeze()
        
        pred_depth = np.divide(f * B, pred_disp, out=np.zeros_like(pred_disp), where=(pred_disp > 0))
        pred_depth[pred_disp <= 0] = np.nan
        
        depth_viz = np.clip(pred_depth, 0, max_depth_viz) / max_depth_viz
        depth_gray = (depth_viz * 255).astype(np.uint8)
        depth_gray = 255 - depth_gray
        depth_colored = np.stack([depth_gray]*3, axis=-1)
        
        pred_seg_overlay = apply_seg_overlay(depth_colored, pred_seg_mask, seg_alpha, class_colors=EXP10_CLASS_COLORS)
        fig_best_exp10.add_trace(go.Image(z=pred_seg_overlay), row=row_idx, col=1)
        
        pred_seg_rgb_overlay = apply_seg_overlay(left_img, pred_seg_mask, seg_alpha, class_colors=EXP10_CLASS_COLORS)
        pc_img = generate_pointcloud_screenshot(
            pred_disp=pred_disp, rgb_image=pred_seg_rgb_overlay, focal_length=f, baseline=B, 
            max_depth=300, denoise=False
        )
        fig_best_exp10.add_trace(go.Image(z=pc_img), row=row_idx, col=2)
        
    fig_best_exp10.add_annotation(
        text="<b>Ground Truth</b>", xref="paper", yref="paper", x=-0.055, y=0.92,
        textangle=-90, showarrow=False, font=dict(size=22)
    )
    fig_best_exp10.add_annotation(
        text="<b>ST</b>", xref="paper", yref="paper", x=-0.055, y=0.625,
        textangle=-90, showarrow=False, font=dict(size=22)
    )
    fig_best_exp10.add_annotation(
        text="<b>MT</b>", xref="paper", yref="paper", x=-0.055, y=0.37,
        textangle=-90, showarrow=False, font=dict(size=22)
    )
    fig_best_exp10.add_annotation(
        text="<b>MT-KD</b>", xref="paper", yref="paper", x=-0.055, y=0.095,
        textangle=-90, showarrow=False, font=dict(size=22)
    )
    
    fig_best_exp10.add_annotation(
        text="<b>Joint Output</b>", xref="paper", yref="paper", x=0.155, y=-0.02,
        showarrow=False, font=dict(size=22)
    )
    fig_best_exp10.add_annotation(
        text="<b>3D Reconstruction</b>", xref="paper", yref="paper", x=0.9, y=-0.02,
        showarrow=False, font=dict(size=22)
    )

    fig_best_exp10.update_layout(
        width=1200, height=800,
        margin=dict(l=60, r=20, t=20, b=60),
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False
    )
            
    fig_best_exp10.update_xaxes(showticklabels=False, visible=False)
    fig_best_exp10.update_yaxes(showticklabels=False, visible=False)
    
    save_figure(fig_best_exp10, height=1250, lrtb_margin=(30, 10, 0, 40), name='H10F04', folder='results', skip_sync=False)

# Uses best seed for each config of experiment 10 according to mean 
# because ST run had no disparity trained, the best disp seed across all experiments is used

# %%

