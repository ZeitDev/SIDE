# %%
import os
import sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

import mlflow
import plotly.graph_objects as go

import open3d as o3d
import pyvista as pv
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils import helpers
from utils.helpers import upsample_logits, logits2disparity

from data.transforms import build_transforms
from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)

# %%
# Settings
arch = 'convnext'
run = 'wMT-KD/260406:2036/train'
task_mode = 'combined'
sample_indices = [226, 81] # Bad / Good
sample_metrics = {
    0: {'dice': 0.744957685470581, 'absrel': 0.13829538226127625, 'score': 0.7990894867050663},
    1: {'dice': 0.9868193864822388, 'absrel': 0.03192799538373947, 'score': 0.9773558019806275}
}

# Final composite controls
# Set either width or height (or both) to resize every panel before stitching.
panel_width = 420
panel_height = None

# Optional overall figure scaling after stitching.
figure_scale = 1.0



# %%
# Load model
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model_run_id = helpers.get_model_run_id(run)
model = mlflow.pytorch.load_model(f'runs:/{model_run_id}/best_model_{task_mode}').to(device)
model.eval()

# %%
# Load data
config_name = run.split('/')[0]

with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
with open(os.path.join('configs', arch, config_name + '.yaml'), 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)

dataset_class = helpers.load(config['data']['dataset'])

test_transforms = build_transforms(config, mode='test')
dataset_test = dataset_class(
    mode='test',
    config=config,
    transforms=test_transforms,
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=config['general']['num_workers'],
    pin_memory=config['general']['pin_memory'],
    persistent_workers=False
)

# %%
if not sample_indices:
    from metrics.segmentation import Dice
    from metrics.disparity import AbsRel

    dice_metric = Dice(n_classes=config['data']['num_of_classes']['segmentation'], ignore_index=255, device=device)
    absrel_metric = AbsRel(max_disparity=512, device=device)

    sample_metrics = []

    with torch.no_grad():
        for idx, data in enumerate(dataloader_test):
            left_images = data['image'].to(device)
            right_images = data['right_image'].to(device) if 'right_image' in data else None
            
            targets_seg = data['segmentation'].to(device)
            targets_disp = data['disparity'].to(device)
            
            baseline = data['baseline'].to(device)
            focal_length = data['focal_length'].to(device)
            
            outputs = model(left_images, right_images)
            
            dice_metric.reset()
            dice_metric.update(outputs['segmentation'], targets_seg)
            dice_dict = dice_metric.compute() 
            
            val_dice = dice_dict[1]
            
            absrel_metric.reset()
            absrel_metric.update(outputs['disparity'], targets_disp, baseline, focal_length)
            absrel_val = absrel_metric.compute()['AbsRel_rate']
                
            absrel_clamped = max(0.0, min(1.0, 1.0 - absrel_val))
            denominator = val_dice + absrel_clamped
            if denominator == 0:
                val_heuristic = 0.0
            else:
                val_heuristic = (2 * val_dice * absrel_clamped) / denominator
            
            sample_metrics.append({
                'index': idx,
                'dice': val_dice,
                'absrel': absrel_val,
                'score': val_heuristic
            })

    sample_metrics.sort(key=lambda x: x['score'])
    bad_sample_idx = sample_metrics[0]['index']
    good_sample_idx = sample_metrics[-1]['index']

    print(f"Bad Sample Index: {bad_sample_idx} (DICE: {sample_metrics[0]['dice']:.4f}, AbsRel: {sample_metrics[0]['absrel']:.4f})")
    print(f"Good Sample Index: {good_sample_idx} (DICE: {sample_metrics[-1]['dice']:.4f}, AbsRel: {sample_metrics[-1]['absrel']:.4f})")
else:
    bad_sample_idx, good_sample_idx = sample_indices
    
data = {}
for i, idx in enumerate([bad_sample_idx, good_sample_idx]):
    dataset = dataset_test[idx]
    left_image = dataset['image'].unsqueeze(0).to(device)
    right_image = dataset['right_image'].unsqueeze(0).to(device) if 'right_image' in dataset else None
    
    target_segmentation = dataset['segmentation'].unsqueeze(0).to(device)
    target_disparity = dataset['disparity'].unsqueeze(0).to(device)
    
    baseline = dataset['baseline'].unsqueeze(0).to(device)
    focal_length = dataset['focal_length'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(left_image, right_image)
        
    data[i] = {
        'left_image': left_image.squeeze(0).cpu().numpy(),
        'target_segmentation': target_segmentation.squeeze(0).cpu().numpy(),
        'target_disparity': target_disparity.squeeze(0).cpu().numpy(),
        'output_segmentation': output['segmentation'].squeeze(0).cpu().numpy(),
        'output_disparity': output['disparity'].squeeze(0).cpu().numpy(),
        'teacher_segmentation': dataset['teacher_segmentation'].squeeze(0).cpu().numpy() if 'teacher_segmentation' in dataset else None,
        'teacher_disparity': dataset['teacher_disparity'].squeeze(0).cpu().numpy() if 'teacher_disparity' in dataset else None,
        'baseline': baseline.item(),
        'focal_length': focal_length.item()
    }
    
# %%
max_disparity = 512.0
seg_color = np.array([0, 255, 255])  # Cyan in RGB
seg_alpha = 0.4

focus = (0.0, 0.0, 0.0)
show_endoscope = False

radius = 0.01
distance = -0.1
duration = 4
fps = 30
resolution = [512, 512]
grid = False
point_cloud_scale_factor = 1.0

scale = 1
max_depth = 300

denoise_cloud = False
denoise_num_points = 30
denoise_radius = 0.03
    
endoscope_path = Path('/data/Zeitler/Visualization/3d_reconstructions/long_endoscope.ply')

# %%
def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud

def depth2xyzmap(depth:np.ndarray, K, uvs:np.ndarray=None, zmin=0):
    #invalid_mask = (depth < zmin)
    H,W = depth.shape[:2]
    if uvs is None:
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]
    zs = depth[vs,us]
    xs = (us-K[0,2]) * zs / K[0,0]
    ys = (vs-K[1,2]) * zs / K[1,1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float32)
    xyz_map[vs,us] = pts
    # if invalid_mask.any():
    #     xyz_map[invalid_mask] = 0
    return xyz_map

# %%
for sample_idx, data_idx in enumerate([1, 0]): # Good first (1), Bad second (0)
    sample = data[data_idx]
    prefix = "GoodSample" if sample_idx == 0 else "BadSample"
    
    left_image = sample['left_image'].transpose(1,2,0) #(H,W,3)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    left_image = (left_image * std) + mean
    left_image = np.clip(left_image * 255.0, 0, 255).astype(np.float32)
    
    out_seg = sample['output_segmentation']
    pred_seg_mask = out_seg.argmax(axis=0)
    target_seg_mask = sample['target_segmentation'].squeeze()
    
    pred_disp = sample['output_disparity'].squeeze() * max_disparity
    target_disp = sample['target_disparity'].squeeze() * max_disparity
    
    teach_seg = torch.tensor(sample['teacher_segmentation'])
    teach_seg_up = upsample_logits(teach_seg.unsqueeze(0), (pred_seg_mask.shape[0], pred_seg_mask.shape[1]))
    teach_seg_mask = teach_seg_up.argmax(dim=1).squeeze().cpu().numpy()
        
    teach_disp = torch.tensor(sample['teacher_disparity'])
    teach_disp_up = logits2disparity(teach_disp.unsqueeze(0), (pred_seg_mask.shape[0], pred_seg_mask.shape[1])) * 512.0
    teach_disp_np = np.nan_to_num(teach_disp_up.squeeze().cpu().numpy(), nan=0.0)
    
    ###
    
    # pred_seg_mask = target_seg_mask
    # pred_disp = target_disp
    
    ###
    
    solid_seg_colors = np.zeros_like(left_image)
    mask = pred_seg_mask > 0
    solid_seg_colors[mask] = seg_color
    
    final_overlay_image = left_image.copy().astype(np.float32)
    foreground_mask = (pred_seg_mask > 0)
    if foreground_mask.any():
        blended_region = (left_image[foreground_mask].astype(np.float32) * (1.0 - seg_alpha)) + \
                        (solid_seg_colors[foreground_mask].astype(np.float32) * seg_alpha)
        final_overlay_image[foreground_mask] = blended_region
    final_overlay_image = np.clip(final_overlay_image, 0, 255).astype(np.uint8)
    
    # ! CARE LAZY APPROACH AHEAD !
    with open('/data/Zeitler/SIDED/EndoVis17/processed/test/instrument_dataset_10/calibration/foundation_stereo_calibration.txt', 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
        
    depth = K[0,0] * baseline / pred_disp
    xyz_map = depth2xyzmap(depth, K)
    color_input = final_overlay_image.reshape(-1, 3) / 255.0
    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), color_input)
    keep_mask = (np.asarray(pcd.points)[:,2] <= max_depth)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    
    if denoise_cloud:
        cl, ind = pcd.remove_radius_outlier(nb_points=denoise_num_points, radius=denoise_radius)
        inlier_cloud = pcd.select_by_index(ind)
        pcd = inlier_cloud
        
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # pt_cloud = pv.PolyData(points)
    # pt_cloud.point_data['RGB'] = colors
    # pt_cloud.points = pt_cloud.points * point_cloud_scale_factor
    
    # plotter = pv.Plotter()
    # plotter.set_background('black')
    # plotter.add_mesh(pt_cloud, point_size=1, rgb=True)
    # plotter.reset_camera()
    # img = plotter.screenshot(return_img=True)
    # plt.imshow(img)
    # plt.show()
    # plotter.close()
    
    # Pointcloud
    step = 5 
    sub_points = points[::step]
    sub_colors = colors[::step]
    sub_colors = sub_colors * 255.0

    plotly_colors = [f'rgb({int(r)}, {int(g)}, {int(b)})' for r, g, b in sub_colors]

    fig = go.Figure(data=[go.Scatter3d(
        x=sub_points[:, 0],
        y=sub_points[:, 1],
        z=sub_points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=plotly_colors,
            opacity=1.0
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data' # Keeps the true aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='black'
    )

    fig.show()
    
    #export_pcd = o3d.geometry.PointCloud()
    # export_pcd.points = o3d.utility.Vector3dVector(points[::step].astype(np.float64))
    # export_pcd.colors = o3d.utility.Vector3dVector(colors[::step].astype(np.float64))
    #o3d.io.write_point_cloud(f'{prefix}_cloud.ply', export_pcd)
    #o3d.io.write_point_cloud(f'./notebooks/output/pointclouds/model.ply', pcd)
    

# %%
depth.max()


# %%
