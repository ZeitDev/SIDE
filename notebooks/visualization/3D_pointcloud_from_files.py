# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import cv2
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import matplotlib.pyplot as plt

os.chdir('/data/Zeitler/code/SIDE')

# %% Settings
idx = '03'

max_depth = 300

denoise_cloud = False
denoise_num_points = 30
denoise_radius = 0.03

# %%
def load_rectified_calibration(yaml_path):
    # Standard yaml.safe_load will fail on OpenCV matrices, use FileStorage
    cv_file = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    
    # After rectification, P1 (left) and P2 (right) contain the projection matrices
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    cv_file.release()

    # The intrinsic matrix K is the left 3x3 block of P1 (or P2)
    K = P1[:, :3]
    fx = P1[0, 0]

    # The baseline is embedded in P2[0, 3] = -fx * baseline
    baseline = abs(P2[0, 3] / fx)
    
    return K, baseline

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
right_img_path = f'./notebooks/input/martini/{idx}.png'
disp_npy_path = f'./notebooks/input/martini/{idx}_disparity_px.npy'
calib_yaml_path = './notebooks/input/martini/stereo_calib.yaml'

# 1. Load the data
# Load image and convert BGR to RGB
image = cv2.imread(right_img_path)
image_cropped = image[:540, :715, ::-1]
image_float = image_cropped.astype(np.float32)
#plt.imshow(image_float.astype(np.uint8))

# Load disparity and ensure it's 2D (H, W)
pred_disp = np.load(disp_npy_path).squeeze() 
#plt.imshow(pred_disp, cmap='plasma')

# Load calibration
K, baseline = load_rectified_calibration(calib_yaml_path)

print(f"Loaded Intrinsics: \n{K}")
print(f"Loaded Baseline: {baseline:.5f}")

# 2. Calculate Depth
# Avoid division by zero by creating a safe mask
disp_mask = pred_disp > 0
depth = np.zeros_like(pred_disp, dtype=np.float32)
depth[disp_mask] = (K[0,0] * baseline) / pred_disp[disp_mask]

# 3. Create XYZ Map
xyz_map = depth2xyzmap(depth, K)

# 4. Generate Point Cloud
color_input = image_float.reshape(-1, 3) / 255.0
pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), color_input)

# Filter out points with zero depth (background / invalid disparity)
valid_points_mask = depth.reshape(-1) > 0
pcd = pcd.select_by_index(np.where(valid_points_mask)[0])

# Filter by max_depth as in your original script
keep_mask = (np.asarray(pcd.points)[:,2] <= max_depth)
keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
pcd = pcd.select_by_index(keep_ids)

# Optional Denoising
if denoise_cloud:
    cl, ind = pcd.remove_radius_outlier(nb_points=denoise_num_points, radius=denoise_radius)
    pcd = pcd.select_by_index(ind)

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 5. Plotly Visualization
step = 5 
sub_points = points[::step]
sub_colors = colors[::step] * 255.0

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
        aspectmode='data' 
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    paper_bgcolor='black'
)

fig.show()
o3d.io.write_point_cloud(f'./notebooks/output/pointclouds/{idx}.ply', pcd)

# %%