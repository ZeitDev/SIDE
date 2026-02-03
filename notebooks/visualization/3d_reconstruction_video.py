# %% Imports
# Imports
import os
import cv2
import imageio
from pathlib import Path
import numpy as np
import open3d as o3d
import pyvista as pv
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cpu_cores = list(range(24))
os.sched_setaffinity(os.getpid(), cpu_cores)

# %% Settings
# Settings
# sequence_name = 'instrument_dataset_1'
# frame_num = '199'

# left_image_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/train/{sequence_name}/input/left_images/image{frame_num}.png')
# segmentation_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/train/{sequence_name}/ground_truth/segmentation/image{frame_num}.png')
# disparity_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/train/{sequence_name}/ground_truth/disparity/image{frame_num}.png')
# intrinsic_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/train/{sequence_name}/input/foundation_stereo_calibration.txt')
# output_path = Path('/data/Zeitler/debug')
# 


# %% Helpers
# Helpers

def generate_3d_reconstruction_video(
    left_image_path:Path,
    segmentation_path:Path,
    disparity_path:Path,
    intrinsic_path:Path,
    output_path:Path,
    name:str):
    
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

    # % Generate Point Cloud
    # Generate Point Cloud
    left_image = imageio.imread(left_image_path)
    left_image = left_image[:, :, :3]
    left_image = cv2.resize(left_image, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)

    seg_mask = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)

    disp_scaled = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED)
    disp = disp_scaled / 128.0

    # invalid = disp == 0.0
    # disp[invalid] = np.inf

    solid_seg_colors = np.zeros_like(left_image)
    palette = [
        [255, 0, 0],    # Class 1: Red
        [0, 255, 0],    # Class 2: Green
        [0, 0, 255],    # Class 3: Blue
        [255, 255, 0],  # Class 4: Yellow
        [0, 255, 255],  # Class 5: Cyan
        [255, 0, 255],  # Class 6: Magenta
        [255, 128, 0],  # Class 7: Orange
        [128, 0, 255],  # Class 8: Purple
    ]
    unique_ids = np.unique(seg_mask)
    for uid in unique_ids:
        if uid == 0: continue
        
        color = palette[(uid - 1) % len(palette)]
        mask_boolean = (seg_mask == uid)
        solid_seg_colors[mask_boolean] = color
        
    final_overlay_image = left_image.copy().astype(np.float32)
    foreground_mask = (seg_mask > 0)
    if foreground_mask.any():
        blended_region = (left_image[foreground_mask].astype(np.float32) * (1.0 - 0.4)) + \
                        (solid_seg_colors[foreground_mask].astype(np.float32) * 0.4)
        final_overlay_image[foreground_mask] = blended_region
    final_overlay_image = np.clip(final_overlay_image, 0, 255).astype(np.uint8)


    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
        
    K[:2] *= scale
    depth = K[0,0] * baseline / disp
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), final_overlay_image.reshape(-1, 3))
    # keep_mask = (np.asarray(pcd.points)[:,2] > 0) & (np.asarray(pcd.points)[:,2] <= max_depth)
    keep_mask = (np.asarray(pcd.points)[:,2] <= max_depth)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    #o3d.io.write_point_cloud(output_path / 'cloud.ply', pcd)

    # is_finite = np.isfinite(pcd.points).all(axis=1) # No NaNs or Infs

    # # Combine them
    # clean_mask = is_finite

    # # Apply the mask
    # pcd = pcd.select_by_index(np.where(clean_mask)[0])

    # TODO: FIX MISSING INSTRUMENTS IN POINT CLOUD

    if denoise_cloud:
        cl, ind = pcd.remove_radius_outlier(nb_points=denoise_num_points, radius=denoise_radius)
        inlier_cloud = pcd.select_by_index(ind)
        pcd = inlier_cloud
        #o3d.io.write_point_cloud(output_path.parent / 'cloud_denoise.ply', inlier_cloud)

    # % Render Video
    # Render Video
    def render_camera_axis(pv_plotter, scale=0.015):
        x = pv.Arrow(direction=(1,0,0), scale=scale)
        y = pv.Arrow(direction=(0,1,0), scale=scale)
        z = pv.Arrow(direction=(0,0,1), scale=scale)

        pv_plotter.add_mesh(x, color='red')
        pv_plotter.add_mesh(y, color='green')
        pv_plotter.add_mesh(z, color='blue')
        
    plotter = pv.Plotter(window_size=resolution)
    plotter.open_movie(output_path / f'{name}_reconstruction.mp4', fps)
    plotter.set_background('black')
    if grid: plotter.show_grid()
    plotter.show_axes()
    plotter.camera_set = True
    #render_camera_axis(plotter, 0.015)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    pt_cloud = pv.PolyData(points)
    pt_cloud.point_data['RGB'] = colors
    pt_cloud.points = pt_cloud.points * point_cloud_scale_factor
    plotter.add_mesh(pt_cloud, point_size=1, rgb=True)

    if show_endoscope:
        endoscope = pv.read(endoscope_path)
        plotter.add_mesh(endoscope, rgb=True)
        
    if focus is None: focus = pt_cloud.center
    else: focus = focus

    for f in tqdm(range(0, fps * duration), total=fps * duration, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        r = radius
        x = r * np.cos((2 * np.pi * f) / (fps * duration))
        y = -r * np.sin((2 * np.pi * f) / (fps * duration))
        
        plotter.camera.SetPosition(x, y, distance)
        plotter.camera.SetViewUp(0, -1, 0)
        plotter.camera.SetFocalPoint(focus)
        plotter.reset_camera_clipping_range()
        plotter.write_frame()

    plotter.close()

    # %
    norm_disp = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color_disp = cv2.applyColorMap(norm_disp, cv2.COLORMAP_JET)
                    
    # Option B: Overlay
    disp_overlay = cv2.addWeighted(left_image, 0.6, color_disp, 0.4, 0)
    cv2.imwrite(output_path / f'{name}_overlay.png', disp_overlay)

# %%
# %% Loop
for mode in ['train', 'test']:
    for sequence_name in sorted(os.listdir(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}')):
        output_path = Path(f'/data/Zeitler/Visualization/3d_reconstructions/EndoVis17/{mode}/{sequence_name}')
        output_path.mkdir(parents=True, exist_ok=True)
        if Path.is_dir(Path(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}/{sequence_name}')):
            for frame_num in tqdm(range(0, 200)):
                frame_num_str = str(frame_num).zfill(3)
                left_image_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}/{sequence_name}/input/left_images/image{frame_num_str}.png')
                segmentation_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}/{sequence_name}/ground_truth/segmentation/image{frame_num_str}.png')
                disparity_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}/{sequence_name}/ground_truth/disparity/image{frame_num_str}.png')
                intrinsic_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}/{sequence_name}/input/foundation_stereo_calibration.txt')
                
                name = f'{mode}_{sequence_name}_{frame_num_str}'
                
                generate_3d_reconstruction_video(
                    left_image_path,
                    segmentation_path,
                    disparity_path,
                    intrinsic_path,
                    output_path,
                    name
                )