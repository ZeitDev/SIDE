# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))
import sys
import yaml
import cv2
import torch
import numpy as np
import subprocess
from tqdm import tqdm


from utils import helpers
from processors.tester import Tester
from utils.helpers import load
from data.transforms import build_transforms
import mlflow
import mlflow.artifacts

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment()

arch = 'convnext'
run = 'wMT-KD/260406:2036/train'
task_mode = 'combined'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SAMPLES = 100
FPS = 5
SEG_COLOR = [255, 255, 0] # Cyan in BGR

def overlay_segmentation(base_image, mask, color, alpha=0.5):
    """Overlays a binary mask onto a base BGR image."""
    overlay = base_image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, base_image, 1 - alpha, 0)

def tensor2depth(tensor, baseline, focal_length, max_disp):
    """Converts a (1, H, W) float disparity tensor into a MAGMA BGR image representing Depth."""
    # The dataloader & model return normalized disparities [0, 1]. Multiply by max_disp (512.0)
    disp = tensor.squeeze().cpu().numpy() * max_disp
    
    valid = disp > 0
    depth = np.zeros_like(disp)
    if isinstance(baseline, torch.Tensor): baseline = baseline.item()
    if isinstance(focal_length, torch.Tensor): focal_length = focal_length.item()
    
    # baseline in dataloader is in meters. We need millimeters for visualization.
    # depth calculation matches the user's prompt / generate_dataset_video (with * 1000.0 for mm)
    depth[valid] = (focal_length * baseline) / disp[valid]
    
    # Typical depth visualization scaling (max 15cm = 150mm)
    max_depth_mm = 150.0
    
    # Ensure invalid areas are black and scale depth
    depth_norm = np.zeros(depth.shape, dtype=np.uint8)
    depth_norm[valid] = 255 - np.clip(depth[valid] * (255.0 / max_depth_mm), 0, 255).astype(np.uint8)
    
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    depth_colored[~valid] = [0, 0, 0]
    
    return depth_colored

def inference_video(run, task_mode, arch):
    state_path_parts = run.split('/')
    experiment = state_path_parts[0]
    run_path = '/'.join(state_path_parts[1:])
    
    mlflow_experiment = mlflow.get_experiment_by_name(experiment)
    mlflow_run = mlflow.search_runs(
        experiment_ids=[mlflow_experiment.experiment_id], 
        filter_string=f"run_name = '{state_path_parts[1]}'"
    ).iloc[0] 

    base_config_filepath = mlflow.artifacts.download_artifacts(
        run_id=mlflow_run.run_id, artifact_path='configs/base.yaml', dst_path='../.temp'
    )
    
    # In model_speed_evaluation, it uses arch and experiment like so:
    experiment_config_filepath = mlflow.artifacts.download_artifacts(
        run_id=mlflow_run.run_id, artifact_path=f'configs/{arch}/{experiment}.yaml', dst_path='../.temp'
    )

    with open(base_config_filepath, 'r') as f: base_config = yaml.safe_load(f)
    with open(experiment_config_filepath, 'r') as f: experiment_config = yaml.safe_load(f)
    config = helpers.deep_merge(experiment_config, base_config)
    config['logging']['notebook_mode'] = True
    config['training']['batch_size'] = 1 # Force batch size to 1 for video creation
    
    # Get Run ID & Initialize Tester
    model_run_id = mlflow.search_runs(
        experiment_ids=[mlflow_experiment.experiment_id], 
        filter_string=f"tags.mlflow.runName = '{run_path}'", 
        order_by=["attributes.start_time DESC"], 
        max_results=1
    ).iloc[0].run_id

    tester = Tester(config, run_id=model_run_id)
    tester._load_models(task_mode=task_mode)
    
    model = tester.model
    dataloader = tester.dataloader_test
    max_disp = config['data'].get('max_disparity', 512.0)
    
    output_video_path = f'/data/Zeitler/Visualization/videos/inference/{run.replace("/", "_")}_{task_mode}'
    os.makedirs(output_video_path, exist_ok=True)
    temp_video_path = os.path.join(output_video_path, 'inference_comparison_temp.mp4')
    final_video_path = os.path.join(output_video_path, 'inference_comparison_web.mp4')
    
    count = 0
    print(f"Generating inference video on {MAX_SAMPLES} samples...")
    video_out = None
    
    with torch.no_grad():
        for data in tqdm(dataloader, total=min(MAX_SAMPLES, len(dataloader))):
            if count >= MAX_SAMPLES:
                break
                
            left_images = data['image'].to(DEVICE)
            right_images = data['right_image'].to(DEVICE) if 'right_image' in data else None
            
            outputs = model(left_images, right_images)
            
            h, w = left_images.shape[2:]
            baseline = data['baseline']
            focal_length = data['focal_length']
            
            # Predictions
            pred_disp = outputs.get('disparity', torch.zeros((1,1,h,w)).to(DEVICE))
            pred_seg_logits = outputs.get('segmentation', torch.zeros((1,2,h,w)).to(DEVICE))
            pred_seg = torch.argmax(pred_seg_logits, dim=1)
            
            pred_disp_vis = tensor2depth(pred_disp, baseline, focal_length, max_disp)
            pred_seg_np = pred_seg.squeeze().cpu().numpy()
            
            pred_vis = overlay_segmentation(pred_disp_vis, pred_seg_np, SEG_COLOR)
            
            # Targets
            t_disp = data.get('disparity', torch.zeros((1,1,h,w)))
            t_seg = data.get('segmentation', torch.zeros((1,h,w)))
            
            target_disp_vis = tensor2depth(t_disp, baseline, focal_length, max_disp)
            target_seg_np = t_seg.squeeze().cpu().numpy()
            
            target_vis = overlay_segmentation(target_disp_vis, target_seg_np, SEG_COLOR)
            
            # Add text headers before stacking
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(pred_vis, 'Prediction - Test Set', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(target_vis, 'Target - Test Set', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

            combined = np.hstack([pred_vis, target_vis])
            
            if video_out is None:
                out_h, out_w = combined.shape[:2]
                video_out = cv2.VideoWriter(
                    temp_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    FPS, (out_w, out_h)
                )
            
            video_out.write(combined)
            count += 1
            
    if video_out is not None:
        video_out.release()
        
    print(f"Converting to H.264 for web optimization using FFmpeg...")
    if os.path.exists(final_video_path):
        os.remove(final_video_path)
    
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_video_path,
        # Video settings
        '-c:v', 'libx264', 
        '-profile:v', 'baseline', 
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',                # Lighter compression, easier on the CPU
        '-maxrate', '1.5M',          # Slightly lowered to ensure smooth streaming
        '-bufsize', '3M',
        '-vf', 'scale=-2:480',       # CRITICAL: Force height to 480p (or 720 max)
        '-r', '30',                  # CRITICAL: Force max 30 fps
        # Audio settings (CRITICAL)
        '-c:a', 'aac',               # Universally supported audio
        '-b:a', '128k', 
        '-ac', '2',                  # Force stereo
        # Container settings
        '-movflags', '+faststart',
        final_video_path
    ], check=True, capture_output=True)
    
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        
    print(f"\nFinished! Web-ready video saved to {final_video_path}")

if __name__ == '__main__':
    inference_video(run, task_mode, arch)

# %%
