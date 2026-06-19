import os
import shutil
import subprocess
import plotly.express as px
import cairosvg

def save_figure(fig, height=400, name='test', lrtb_margin=(0, 0, 0, 0), standoff=5, fallback=True, folder='methods', skip_sync=False):
    font_size = 15.5
    width = 600
    family = 'Latin Modern Roman, Computer Modern Roman, serif'
    
    # 1. Primary local output path
    base_path = f'notebooks/output/{folder}/'
    os.makedirs(base_path, exist_ok=True)
    
    axis_kwargs = dict(title_font=dict(size=font_size+2, family=family), tickfont=dict(size=font_size, family=family))
    if standoff is not None:
        axis_kwargs['title_standoff'] = standoff
        
    fig.update_xaxes(**axis_kwargs)
    fig.update_yaxes(**axis_kwargs)
    fig.update_annotations(font=dict(size=font_size+2, family=family)) # Subplot titles
    
    # Standardize Colorbars if they exist
    fig.update_coloraxes(
        colorbar_title_font_size=font_size+2,
        colorbar_title_font_family=family,
        colorbar_tickfont_size=font_size,
        colorbar_tickfont_family=family
    )
    
    for trace in fig.data:
        if 'colorbar' in trace:
            trace.update(
                colorbar_title_font_size=font_size+2,
                colorbar_title_font_family=family,
                colorbar_tickfont_size=font_size,
                colorbar_tickfont_family=family
            )

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=lrtb_margin[0], r=lrtb_margin[1], t=lrtb_margin[2], b=lrtb_margin[3]),
        font=dict(
            family=family,
            size=font_size,
            color='black'
        ),
        legend=dict(
            title_font=dict(family=family, size=font_size+2),
            font=dict(family=family, size=font_size)
        )
    )
    
    fig.show(config={'toImageButtonOptions': {'format': 'svg', 'filename': f'{base_path}{name}'}})
    
    # 2. Save the file locally
    generated_file_path = ""
    if not fallback:
        svg_bytes = fig.to_image(format='svg', width=width, height=height)
        generated_file_path = os.path.join(base_path, f'{name}.pdf')
        cairosvg.svg2pdf(bytestring=svg_bytes, write_to=generated_file_path)
    else:
        generated_file_path = os.path.join(base_path, f'{name}.png')
        fig.write_image(generated_file_path, width=width, height=height, scale=3)
        
    # 3. Automatically sync to Overleaf if a repo path is provided
    if not skip_sync: sync_to_overleaf(generated_file_path, folder, name)


def apply_chart_config(fig, name, config):
    """Applies axis ranges and tick frequencies from a config dictionary."""
    if name not in config:
        return
    
    settings = config[name]
    
    # Handle x-axes
    for x_key in [k for k in settings.keys() if k.startswith('x')]:
        suffix = x_key[1:]
        row = None
        col = None
        
        if suffix.isdigit():
            col = int(suffix)
        elif 'r' in suffix:
            parts = suffix.split('r')
            if parts[0].isdigit(): col = int(parts[0])
            if parts[1].isdigit(): row = int(parts[1])
        
        axis_settings = settings[x_key].copy()
        if 'range' in axis_settings:
            axis_settings['autorange'] = False
            
        fig.update_xaxes(**axis_settings, row=row, col=col)
        
    # Handle y-axes
    for y_key in [k for k in settings.keys() if k.startswith('y')]:
        suffix = y_key[1:]
        row = None
        col = None
        
        if suffix.isdigit():
            row = int(suffix)
        elif 'c' in suffix:
            parts = suffix.split('c')
            if parts[0].isdigit(): row = int(parts[0])
            if parts[1].isdigit(): col = int(parts[1])
        
        axis_settings = settings[y_key].copy()
        if 'range' in axis_settings:
            axis_settings['autorange'] = False
            
        fig.update_yaxes(**axis_settings, row=row, col=col)


def sync_to_overleaf(source_file, folder, name):
    """Copies the generated image to a local Overleaf Git repo and pushes it."""
    # Define and create target directory inside the Overleaf repository
    repo_path = '/data/Zeitler/masterthesis'
    target_dir = os.path.join(repo_path, 'figures', folder)
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy file over
    filename = os.path.basename(source_file)
    target_file = os.path.join(target_dir, filename)
    shutil.copy2(source_file, target_file)
    
    #subprocess.run(['git', 'add', target_file], cwd=repo_path, check=True, capture_output=True)

import numpy as np

def toOpen3dCloud(points, colors=None, normals=None):
    import open3d as o3d
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
    return xyz_map

def get_calibration_K(cx=None, cy=None, fx=1251.4659480063228, fy=1251.4659480063228):
    """Returns the K matrix. If cx/cy not provided, typical Endovis 17 defaults are used."""
    if cx is None: cx = 633.2722702026367
    if cy is None: cy = 514.8552856445312
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)

def generate_pointcloud_trace(pred_disp, rgb_image, focal_length, baseline, max_depth=300, step=1, denoise=False):
    """Generates a Plotly go.Scatter3d trace for a pointcloud given disparity and RGB."""
    import plotly.graph_objects as go
    
    K = get_calibration_K(fx=focal_length, fy=focal_length)
    depth = K[0,0] * baseline / pred_disp
    xyz_map = depth2xyzmap(depth, K)
    
    color_input = rgb_image.reshape(-1, 3) / 255.0
    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), color_input)
    
    keep_mask = (np.asarray(pcd.points)[:,2] <= max_depth)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    
    if denoise:
        pcd, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.03)
        
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    sub_points = points[::step]
    sub_colors = colors[::step] * 255.0
    plotly_colors = [f'rgb({int(r)}, {int(g)}, {int(b)})' for r, g, b in sub_colors]
    
    trace = go.Scatter3d(
        x=sub_points[:, 0],
        y=sub_points[:, 1],
        z=sub_points[:, 2],
        mode='markers',
        marker=dict(size=2, color=plotly_colors, opacity=1.0)
    )
    return trace

def generate_pointcloud_screenshot(pred_disp, rgb_image, focal_length, baseline, max_depth=300, denoise=False):
    """Generates a 2D RGB screenshot of the pointcloud using PyVista and Off-screen rendering."""
    import pyvista as pv
    K = get_calibration_K(fx=focal_length, fy=focal_length)
    
    pred_disp_safe = np.where(pred_disp <= 0, 1e-6, pred_disp)
    depth = K[0,0] * baseline / pred_disp_safe
    
    xyz_map = depth2xyzmap(depth, K)
    
    color_input = rgb_image.reshape(-1, 3) / 255.0
    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), color_input)
    
    points_z = np.asarray(pcd.points)[:, 2]
    # Filter bounds properly (e.g. baseline might be in mm or m, 
    # but depths typically 0 < z <= max_depth)
    keep_mask = (points_z > 0) & (points_z <= max_depth)
    keep_ids = np.arange(len(points_z))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    
    if denoise and len(pcd.points) > 30:
        # If depth is in mm, 0.03 is too tiny. Scale radius if baseline > 0.1 (mm) 
        # vs if baseline < 0.1 (meters)
        radius = 30.0 if baseline > 0.1 else 0.03
        pcd, ind = pcd.remove_radius_outlier(nb_points=30, radius=radius)
        
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255.0
    
    # Fallback for empty pointclouds to prevent crashing PyVista
    if len(points) == 0:
        return np.zeros((1024, 1024, 3), dtype=np.uint8)
        
    pt_cloud = pv.PolyData(points)
    pt_cloud.point_data['RGB'] = colors.astype(np.uint8)
    
    # pv.start_xvfb() # Necessary for headless servers
    plotter = pv.Plotter(off_screen=True, window_size=[1024, 1024])
    plotter.set_background('black')
    plotter.add_mesh(pt_cloud, point_size=5, rgb=True) # Increased point size
    plotter.view_xy() # Face straight onto the subject to avoid empty side-view!
    
    # ---------------------------------------------------------
    # CAMERA SETTINGS (Tweak these variables to change the view)
    # ---------------------------------------------------------
    plotter.camera_position = 'xy'
    plotter.camera.azimuth = 0      # Rotate left/right
    plotter.camera.roll = 0       # Roll image (often needed b/c image Y is flipped vs 3D Plot Y)
    plotter.camera.elevation = -180 # Angle up/down
    
    plotter.reset_camera()          # Automatically frame the object
    plotter.camera.zoom(2.0)        # Zoom in (>1.0) or out (<1.0)
    # ---------------------------------------------------------
    
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img

