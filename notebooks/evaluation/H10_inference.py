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
# Filter out exp10
df_bench = df_final[df_final['experiment'] != 'exp10'].copy()

# Metric Columns
col_dice = 'metric.best_combined/performance/testing/segmentation/DICE_score/instrument_mean'
col_dice_fallback = 'metric.best_segmentation/performance/testing/segmentation/DICE_score/instrument_mean'

col_absrel = 'metric.best_combined/performance/testing/disparity/AbsRel_rate'
col_absrel_fallback = 'metric.best_disparity/performance/testing/disparity/AbsRel_rate'

# Merge fallback metrics if the combined ones are NaN
df_bench['DICE'] = df_bench[col_dice].fillna(df_bench[col_dice_fallback])
df_bench['AbsRel'] = df_bench[col_absrel].fillna(df_bench[col_absrel_fallback])

# Normalize to 0-1 range (they are saved as percentages in MLflow results)
dice_norm = df_bench['DICE'] / 100.0
absrel_norm = df_bench['AbsRel'] / 100.0

# Calculate combined heuristic for MT / MT-KD ranking (Harmonic Mean)
absrel_clamped = (1.0 - absrel_norm).clip(0, 1)
df_bench['SCORE'] = (2 * dice_norm * absrel_clamped) / (dice_norm + absrel_clamped) * 100.0

best_runs = {}

# Best ST SEG (Max DICE)
df_seg = df_bench[df_bench['config'] == 'SEG']
best_runs['ST-SEG'] = df_seg.loc[df_seg['DICE'].idxmax()]

# Best ST DISP (Min AbsRel)
df_disp = df_bench[df_bench['config'] == 'DISP']
best_runs['ST-DISP'] = df_disp.loc[df_disp['AbsRel'].idxmin()]

# Calculate ST combined SCORE
st_dice_norm = best_runs['ST-SEG']['DICE'] / 100.0
st_absrel_clamped = max(0, 1.0 - (best_runs['ST-DISP']['AbsRel'] / 100.0))
st_combined_score = (2 * st_dice_norm * st_absrel_clamped) / (st_dice_norm + st_absrel_clamped) * 100.0

# Best MT (Max SCORE)
df_mt = df_bench[df_bench['config'] == 'MT']
best_runs['MT'] = df_mt.loc[df_mt['SCORE'].idxmax()]

# Best MT-KD (Max SCORE)
df_mt_kd = df_bench[df_bench['config'] == 'MT-KD']
best_runs['MT-KD'] = df_mt_kd.loc[df_mt_kd['SCORE'].idxmax()]

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
                
                # --- Score (Harmonic Mean) ---
                val_heuristic = 0.0
                if setup_key in ['ST-PIPELINE', 'MT', 'MT-KD']:
                    absrel_clamped = max(0.0, min(1.0, 1.0 - absrel_val))
                    denominator = val_dice + absrel_clamped
                    val_heuristic = 0.0 if denominator == 0 else (2 * val_dice * absrel_clamped) / denominator
                
                # Exclude samples that have 0.0 DICE (meaning no instrument is present to be evaluated)
                if val_dice > 0.0:
                    sample_metrics.append({
                        'index': idx,
                        'dice': val_dice,
                        'absrel': absrel_val,
                        'score': val_heuristic
                    })
                
                if idx % 100 == 0:
                    print(f"  Processed {idx}/{len(dataloader_test)} samples")
                    
        # Sort samples to find best and worst
        sample_metrics.sort(key=lambda x: x['score']) # lowest SCORE is worst [0]
        
        evaluation_results[setup_key] = {
            'all_metrics': sample_metrics,
            'worst': sample_metrics[0],
            'best': sample_metrics[-1]
        }
        
        worst = sample_metrics[0]
        best = sample_metrics[-1]
        print(f"Finished {setup_key}!")
        print(f"  Worst: Index {worst['index']} | DICE: {worst['dice']*100:.2f} | AbsRel: {worst['absrel']*100:.2f} | Score: {worst['score']*100:.2f}")
        print(f"  Best : Index {best['index']} | DICE: {best['dice']*100:.2f} | AbsRel: {best['absrel']*100:.2f} | Score: {best['score']*100:.2f}")

# * Results
# ST-PIPELINE!
# Worst: Index 72 | DICE: 0.11 | AbsRel: 5.18 | Score: 0.23
# Best : Index 268 | DICE: 98.92 | AbsRel: 2.93 | Score: 97.99

# MT!
# Worst: Index 244 | DICE: 0.27 | AbsRel: 3.47 | Score: 0.54
# Best : Index 178 | DICE: 97.93 | AbsRel: 3.57 | Score: 97.17

# MT-KD!
# Worst: Index 162 | DICE: 0.17 | AbsRel: 3.19 | Score: 0.35
# Best : Index 81 | DICE: 98.50 | AbsRel: 3.11 | Score: 97.69

# Hardcode the metrics so we don't have to rerun the long loop every time while developing plotting
if not evaluation_results:
    evaluation_results = {
        'ST-PIPELINE': {
            'worst': {'index': 72, 'dice': 0.11, 'absrel': 5.18, 'score': 0.23}, # Fill in actual scores if needed for annotations
            'best': {'index': 268, 'dice': 98.92, 'absrel': 2.93, 'score': 97.99}
        },
        'MT': {
            'worst': {'index': 244, 'dice': 0.27, 'absrel': 3.47, 'score': 0.54},
            'best': {'index': 178, 'dice': 97.93, 'absrel': 3.57, 'score': 97.17}
        },
        'MT-KD': {
            'worst': {'index': 162, 'dice': 0.17, 'absrel': 3.19, 'score': 0.35},
            'best': {'index': 81, 'dice': 98.50, 'absrel': 3.11, 'score': 97.69}
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
seg_color = np.array([0, 255, 255])  # Cyan in RGB
seg_alpha = 0.4
error_alpha = 1.0
ERROR_GREEN = np.array([0, 200, 0])
ERROR_YELLOW = np.array([255, 255, 0])
ERROR_RED = np.array([255, 0, 0])

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

def apply_seg_overlay(image, mask, color, alpha=0.5):
    overlay = image.copy()
    mask = np.squeeze(mask)
    valid = (mask > 0)
    overlay[valid] = overlay[valid] * (1 - alpha) + np.array(color) * alpha
    return overlay

# Load MT config once to use its dataloader (since it provides both GTs)
_, cfg_mt = load_model_from_run(setups['MT']['exp'], 'MT', setups['MT']['run_name'], 'combined')
from utils import helpers
from data.transforms import build_transforms
test_transforms = build_transforms(cfg_mt, mode='test')
dataset_class = helpers.load(cfg_mt['data']['dataset'])
dataset_test = dataset_class(mode='test', config=cfg_mt, transforms=test_transforms)

for setup_key in ['ST-PIPELINE', 'MT', 'MT-KD']:
    print(f"Generating plot for {setup_key}...")
    
    # 1. Load the corresponding model(s)
    if setup_key == 'ST-PIPELINE':
        model_seg, _ = load_model_from_run(setups['ST-SEG']['exp'], 'SEG', setups['ST-SEG']['run_name'], 'segmentation')
        model_disp, _ = load_model_from_run(setups['ST-DISP']['exp'], 'DISP', setups['ST-DISP']['run_name'], 'disparity')
    else:
        info = setups[setup_key]
        model, _ = load_model_from_run(info['exp'], info['config'], info['run_name'], info['task_mode'])
        
    worst_idx = evaluation_results[setup_key]['worst']['index']
    best_idx = evaluation_results[setup_key]['best']['index']
    
    # Pre-fetch the sample data dicts
    plot_data = {}
    for i, (sample_type, data_idx) in enumerate([('BestSample', best_idx), ('WorstSample', worst_idx)]):
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
                out_seg = model_seg(left_images, right_images)
                out_disp = model_disp(left_images, right_images)
                output = {'segmentation': out_seg['segmentation'], 'disparity': out_disp['disparity']}
            else:
                output = model(left_image, right_image)
            
        plot_data[i] = {
            'left_image': left_image.squeeze(0).cpu().numpy(),
            'target_segmentation': target_segmentation.squeeze(0).cpu().numpy() if target_segmentation is not None else None,
            'target_disparity': target_disparity.squeeze(0).cpu().numpy() if target_disparity is not None else None,
            'output_segmentation': output['segmentation'].squeeze(0).cpu().numpy(),
            'output_disparity': output['disparity'].squeeze(0).cpu().numpy(),
            'baseline': baseline.item() if baseline is not None else 1.0,
            'focal_length': focal_length.item() if focal_length is not None else 1.0,
            'type': sample_type,
            'metrics_dice': evaluation_results[setup_key]['best']['dice'] if i==0 else evaluation_results[setup_key]['worst']['dice'],
            'metrics_absrel': evaluation_results[setup_key]['best']['absrel'] if i==0 else evaluation_results[setup_key]['worst']['absrel'],
        }
        
    # 2. Setup the Plotly Figure
    fig = make_subplots(
        rows=4, cols=3, column_widths=[1, 1, 1],
        vertical_spacing=0.01, horizontal_spacing=0.0001,
        subplot_titles=("<b>Prediction</b>", "<b>Target</b>", "<b>Error</b>",
                        "", "", "",
                        "", "", "",
                        "", "", "")
    )
    
    # 3. Render loop equivalent to `inference_figure.py`
    for sample_idx in [0, 1]: # 0=Best, 1=Worst
        row_offset = 1 if sample_idx == 0 else 3
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
        
        pred_seg_overlay = apply_seg_overlay(rgb_img, pred_seg_mask, seg_color, seg_alpha)
        target_seg_overlay = apply_seg_overlay(rgb_img, target_seg_mask, seg_color, seg_alpha)
        
        seg_match = (pred_seg_mask == target_seg_mask)
        valid_mask = target_seg_mask != 255
        
        seg_error = np.zeros_like(target_seg_mask, dtype=np.float32)
        seg_error[valid_mask & ~seg_match] = 1.0 
        seg_err_color = error_colormap(seg_error)
        
        seg_err_img = rgb_img.copy()
        seg_err_img[valid_mask] = (rgb_img[valid_mask] * (1 - error_alpha) + seg_err_color[valid_mask] * error_alpha).astype(np.uint8)
        
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
        
        absrel_err = np.abs(pred_depth - target_depth) / np.where(target_depth > 0, target_depth, 1e-8)
        absrel_err[np.isnan(absrel_err)] = 0.0
        absrel_err = np.clip(absrel_err, 0, 1.0)
        
        depth_err_color = error_colormap(absrel_err)
        valid_depth = target_depth > 0
        blended_depth_err = rgb_img.copy()
        blended_depth_err[valid_depth] = (rgb_img[valid_depth] * (1 - error_alpha) + depth_err_color[valid_depth] * error_alpha).astype(np.uint8)
        
        fig.add_trace(go.Heatmap(z=pred_depth[::-1], colorscale='Magma_r', zmin=0, zmax=max_depth_viz, showscale=False), row=row_offset + 1, col=1)
        
        fig.add_trace(
            go.Heatmap(
                z=target_depth[::-1], colorscale='Magma_r', zmin=0, zmax=max_depth_viz, 
                showscale=(sample_idx == 0), 
                colorbar=dict(
                    title=dict(text="Depth [mm]", side="right", font=dict(size=18)), 
                    x=1.005, len=1.0, y=0.5, yanchor='middle'
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
                    x=1.1, len=1.0, y=0.5, yanchor='middle'
                )
            ), 
            row=row_offset + 1, col=3
        )
        
    fig.update_xaxes(showticklabels=False, visible=False)
    fig.update_yaxes(showticklabels=False, visible=False)
    for i in range(1, 13): 
        axis_suffix = "" if i == 1 else str(i)
        fig.layout[f"yaxis{axis_suffix}"].update(scaleanchor=f"x{axis_suffix}", scaleratio=1)
        
    # Thesis formatting
    fig.add_annotation(
        text=f"<b>Best Sample</b><br><span style='font-size: 12px'>(DICE: {plot_data[0]['metrics_dice']*100:.1f}%, AbsRel: {plot_data[0]['metrics_absrel']*100:.1f}%)</span>",
        xref="paper", yref="paper", x=-0.034, y=0.96, textangle=-90, showarrow=False, font=dict(size=16)
    )
    fig.add_annotation(
        text=f"<b>Worst Sample</b><br><span style='font-size: 12px'>(DICE: {plot_data[1]['metrics_dice']*100:.1f}%, AbsRel: {plot_data[1]['metrics_absrel']*100:.1f}%)</span>",
        xref="paper", yref="paper", x=-0.034, y=0.04, textangle=-90, showarrow=False, font=dict(size=16)
    )

    fig.update_layout(
        width=1000, 
        height=1100, 
        margin=dict(l=60, r=120, t=50, b=10),
        plot_bgcolor='white', 
        paper_bgcolor='white',
        title=dict(text=f"Inference Format: {setup_key}", x=0.5, xanchor='center', font=dict(size=24))
    )
    
    fig.update_traces(colorbar_tickfont_size=14, colorbar_title_font_size=16, selector=dict(type='heatmap'))
    
    apply_chart_config(fig, f'H10_{setup_key}', CHART_CONFIG)
    save_figure(fig, name=f'H10_Inference_{setup_key}', lrtb_margin=(60, 120, 50, 10), folder='results', skip_sync=False)
    # fig.show()

# %%