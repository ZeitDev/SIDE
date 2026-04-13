# %%
import os
import sys
# Ensure we can import from `utils` if running this notebook specifically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np
from tqdm import tqdm
import torch
from utils.helpers import upsample_logits, logits2disparity
import torch.nn.functional as F

# Create a bright Red-Yellow-Green colormap LUT (0=Red, 127=Yellow, 255=Green)
_lut_x = np.linspace(0, 1, 256)
_lut_r = np.clip(2.0 - 2.0 * _lut_x, 0, 1) * 255
_lut_g = np.clip(2.0 * _lut_x, 0, 1) * 255
_lut_b = np.zeros_like(_lut_x)
BRIGHT_RYG_LUT = np.stack([_lut_b, _lut_g, _lut_r], axis=1).astype(np.uint8) # BGR format

# Global setting for visualization colormap scaling
max_disparity = 512.0 #128.0

# Global setting for segmentation color (BGR format)
seg_color = [255, 255, 0]

# Global setting to toggle disparity overlay (True = blend with image, False = solid colormap where valid)
overlay_disparity = False

fps = 5


# %%
# Dataset Videos
mode = 'test' # 'train' or 'test'
dataset_path = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}'
output_video_path = f'/data/Zeitler/Visualization/videos/dataset/{mode}'
os.makedirs(output_video_path, exist_ok=True)

for subset in sorted(os.listdir(dataset_path)):
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.isdir(subset_path):
        continue
    
    left_dir = os.path.join(subset_path, 'input', 'left_images')
    right_dir = os.path.join(subset_path, 'input', 'right_images')
    seg_dir = os.path.join(subset_path, 'target', 'segmentation')
    disp_dir = os.path.join(subset_path, 'target', 'disparity')
    
    left_images = sorted(os.listdir(left_dir))
    if not left_images:
        continue
        
    sample_img = cv2.imread(os.path.join(left_dir, left_images[0]))
    h, w = sample_img.shape[:2]
    
    video_out = cv2.VideoWriter(
        os.path.join(output_video_path, f'{subset}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, # Adjust FPS to your preference
        (w * 2, h * 2)
    )
    
    for image_name in tqdm(left_images, desc=f'Processing {subset}'): 
        left_img = cv2.imread(os.path.join(left_dir, image_name))
        right_img = cv2.imread(os.path.join(right_dir, image_name))
        
        # Read GTs and enforce reading as single channel where appropriate
        seg_img = cv2.imread(os.path.join(seg_dir, image_name), cv2.IMREAD_GRAYSCALE)
        disp_img = cv2.imread(os.path.join(disp_dir, image_name), cv2.IMREAD_UNCHANGED)
        
        if seg_img is None or disp_img is None:
            continue
            
        # --- Segmentation Overlay ---
        # Segmentation mask has values 0 (background) and 1 (foreground)
        seg_mask = (seg_img == 1)
        
        if seg_mask.any():
            # Apply a solid color to the segmented area
            color_mask = np.zeros_like(left_img)
            color_mask[seg_mask] = seg_color
            # Safely blend the full image and selectively apply the mask
            blended = cv2.addWeighted(left_img, 0.6, color_mask, 0.4, 0)
            seg_overlay = np.where(seg_mask[:, :, None], blended, left_img)
        else:
            seg_overlay = left_img.copy()
        
        # --- Disparity Overlay ---
        # Disparity mask is scaled (raw/128) and valid pixels are > 0
        valid_disp_mask = (disp_img > 0)
        
        if valid_disp_mask.any():
            # Get absolute disparity values in pixels
            valid_vals = disp_img[valid_disp_mask].astype(np.float32) / 128.0
            
            # Fixed absolute scaling using a max threshold
            disp_norm_full = np.zeros(disp_img.shape, dtype=np.uint8)
            disp_norm_full[valid_disp_mask] = np.clip(valid_vals * (255.0 / max_disparity), 0, 255).astype(np.uint8)
                
            disp_color = cv2.applyColorMap(disp_norm_full, cv2.COLORMAP_MAGMA)
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, disp_color, 0.4, 0)
                disp_overlay = np.where(valid_disp_mask[:, :, None], blended, left_img)
            else:
                disp_overlay = np.where(valid_disp_mask[:, :, None], disp_color, np.zeros_like(left_img))
        else:
            disp_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)
            
        # Add labels to the images before stacking
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(left_img, 'Left Image', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right_img, 'Right Image', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(seg_overlay, 'Segmentation Target', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp_overlay, 'Disparity FoundationStereo Pseudo Target', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Make a 2x2 grid view
        top_row = np.hstack((left_img, right_img))
        bottom_row = np.hstack((seg_overlay, disp_overlay))
        combined_frame = np.vstack((top_row, bottom_row))
        
        video_out.write(combined_frame)
        
    video_out.release()
    
# %%
# Teacher Segmentation and Disparity Videos

mode = 'train' # 'train' or 'test'
dataset_path = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}'
teacher_output_video_path = f'/data/Zeitler/Visualization/videos/Teacher_SegFormer_FoundationStereo/{mode}'
os.makedirs(teacher_output_video_path, exist_ok=True)

for subset in sorted(os.listdir(dataset_path)):
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.isdir(subset_path):
        continue
    
    left_dir = os.path.join(subset_path, 'input', 'left_images')
    seg_dir = os.path.join(subset_path, 'target', 'segmentation')
    disp_dir = os.path.join(subset_path, 'target', 'disparity')
    
    teach_seg_dir = os.path.join(subset_path, 'teacher', 'segmentation_2_256_256')
    teach_disp_dir = os.path.join(subset_path, 'teacher', 'disparity_128_256_256')
    
    left_images = sorted(os.listdir(left_dir))
    if not left_images:
        continue
        
    sample_img = cv2.imread(os.path.join(left_dir, left_images[0]))
    h, w = sample_img.shape[:2]
    crop_h, crop_w = 1024, 1024
    
    # Offsets for center crop
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    
    video_out = cv2.VideoWriter(
        os.path.join(teacher_output_video_path, f'{subset}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (crop_w * 2, crop_h * 2)
    )
    
    for image_name in tqdm(left_images, desc=f'Processing {subset} Teachers'): 
        left_img = cv2.imread(os.path.join(left_dir, image_name))
        
        # Load Raw GTs
        seg_img = cv2.imread(os.path.join(seg_dir, image_name), cv2.IMREAD_GRAYSCALE)
        disp_img = cv2.imread(os.path.join(disp_dir, image_name), cv2.IMREAD_UNCHANGED)
        
        if seg_img is None or disp_img is None:
            continue
            
        # Center crop the loaded images
        left_img = left_img[top:top+crop_h, left:left+crop_w]
        seg_img = seg_img[top:top+crop_h, left:left+crop_w]
        disp_img = disp_img[top:top+crop_h, left:left+crop_w]
            
        # Load Raw Teachers
        pt_filename = image_name.replace('.png', '.pt')
        teach_seg_path = os.path.join(teach_seg_dir, pt_filename)
        teach_disp_path = os.path.join(teach_disp_dir, pt_filename)
        
        if not os.path.exists(teach_seg_path) or not os.path.exists(teach_disp_path):
            continue
            
        teach_seg_pt = torch.load(teach_seg_path, weights_only=True).float()
        teach_disp_pt = torch.load(teach_disp_path, weights_only=True).float()
        
        # Add batch dim if missing
        if teach_seg_pt.dim() == 3: teach_seg_pt = teach_seg_pt.unsqueeze(0)
        if teach_disp_pt.dim() == 3: teach_disp_pt = teach_disp_pt.unsqueeze(0)
            
        # Process and un-normalize format using helper functions at crop size
        teach_seg_up = upsample_logits(teach_seg_pt, (crop_h, crop_w))
        teach_seg_mask = teach_seg_up.argmax(dim=1).squeeze().cpu().numpy()
        
        teach_disp_up = logits2disparity(teach_disp_pt, (crop_h, crop_w)) * 512.0
        teach_disp_np = np.nan_to_num(teach_disp_up.squeeze().cpu().numpy(), nan=0.0)

        # --- 1. Segmentation GT Overlay ---
        seg_gt_mask = (seg_img == 1)
        if seg_gt_mask.any():
            color_mask = np.zeros_like(left_img)
            color_mask[seg_gt_mask] = seg_color
            blended = cv2.addWeighted(left_img, 0.6, color_mask, 0.4, 0)
            seg_gt_overlay = np.where(seg_gt_mask[:, :, None], blended, left_img)
        else:
            seg_gt_overlay = left_img.copy()

        # --- 2. Segmentation Teacher Overlay ---
        seg_t_mask = (teach_seg_mask == 1)
        if seg_t_mask.any():
            color_mask_t = np.zeros_like(left_img)
            color_mask_t[seg_t_mask] = seg_color
            blended = cv2.addWeighted(left_img, 0.6, color_mask_t, 0.4, 0)
            seg_t_overlay = np.where(seg_t_mask[:, :, None], blended, left_img)
        else:
            seg_t_overlay = left_img.copy()
            
        # --- 3. Disparity GT Overlay ---
        valid_disp_gt = (disp_img > 0)
        if valid_disp_gt.any():
            valid_vals = disp_img[valid_disp_gt].astype(np.float32) / 128.0
            
            disp_norm_full = np.zeros(disp_img.shape, dtype=np.uint8)
            # Use max_disparity scaling to make disparity visible
            disp_norm_full[valid_disp_gt] = np.clip(valid_vals * (255.0 / max_disparity), 0, 255).astype(np.uint8)
            
            disp_color = cv2.applyColorMap(disp_norm_full, cv2.COLORMAP_MAGMA)
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, disp_color, 0.4, 0)
                disp_gt_overlay = np.where(valid_disp_gt[:, :, None], blended, left_img)
            else:
                disp_gt_overlay = np.where(valid_disp_gt[:, :, None], disp_color, np.zeros_like(left_img))
        else:
            disp_gt_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)
            
        # --- 4. Disparity Teacher Overlay ---
        valid_disp_t = (teach_disp_np > 0)
        if valid_disp_t.any():
            valid_vals_t = teach_disp_np[valid_disp_t]
            
            disp_norm_full_t = np.zeros(teach_disp_np.shape, dtype=np.uint8)
            # Use max_disparity scaling to make disparity visible
            disp_norm_full_t[valid_disp_t] = np.clip(valid_vals_t * (255.0 / max_disparity), 0, 255).astype(np.uint8)
            
            disp_color_t = cv2.applyColorMap(disp_norm_full_t, cv2.COLORMAP_MAGMA)
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, disp_color_t, 0.4, 0)
                disp_t_overlay = np.where(valid_disp_t[:, :, None], blended, left_img)
            else:
                disp_t_overlay = np.where(valid_disp_t[:, :, None], disp_color_t, np.zeros_like(left_img))
        else:
            disp_t_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)

        # Add labels to the images before stacking
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(seg_gt_overlay, 'Segmentation Target', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(seg_t_overlay, 'Segmentation SegFormer-B5 Teacher Logits', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp_gt_overlay, 'Disparity FoundationStereo Pseudo Target', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp_t_overlay, 'Disparity FoundationStereo Teacher Logits', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # 2x2 Grid: [Seg GT | Seg Teacher] \n [Disp GT | Disp Teacher]
        top_row = np.hstack((seg_gt_overlay, seg_t_overlay))
        bottom_row = np.hstack((disp_gt_overlay, disp_t_overlay))
        combined_frame = np.vstack((top_row, bottom_row))
        
        video_out.write(combined_frame)
        
    video_out.release()
    
# %%
# STTR disparity and confidence comparison videos

mode = 'train' # 'train' or 'test'
dataset_path = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}'
sttr_output_video_path = f'/data/Zeitler/Visualization/videos/Teacher_FS_vs_STTR/{mode}'
os.makedirs(sttr_output_video_path, exist_ok=True)

for subset in sorted(os.listdir(dataset_path)):
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.isdir(subset_path):
        continue
    
    left_dir = os.path.join(subset_path, 'input', 'left_images')
    
    teach_disp_dir = os.path.join(subset_path, 'teacher', 'disparity_128_256_256')
    fs_disp_dir = os.path.join(subset_path, 'target', 'disparity')
    sttr_disp_dir = os.path.join(subset_path, 'target', 'disparity_sttr')
    sttr_conf_dir = os.path.join(subset_path, 'teacher', 'disparity_confidence_sttr')
    
    left_images = sorted(os.listdir(left_dir))
    if not left_images:
        continue
        
    sample_img = cv2.imread(os.path.join(left_dir, left_images[0]))
    h, w = sample_img.shape[:2]
    crop_h, crop_w = 1024, 1024
    
    # Offsets for center crop
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    
    video_out = cv2.VideoWriter(
        os.path.join(sttr_output_video_path, f'{subset}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (crop_w * 2, crop_h * 2)
    )
    
    for image_name in tqdm(left_images, desc=f'Processing {subset} STTR Comparison'): 
        left_img = cv2.imread(os.path.join(left_dir, image_name))
        
        if left_img is None:
            continue
            
        # Center crop the image
        left_img = left_img[top:top+crop_h, left:left+crop_w]
        
        # Load FoundationStereo Target disparity (not teacher)
        fs_disp_path = os.path.join(fs_disp_dir, image_name)
        fs_disp = None
        if os.path.exists(fs_disp_path):
            fs_disp = cv2.imread(fs_disp_path, cv2.IMREAD_UNCHANGED)
            if fs_disp is not None:
                fs_disp = fs_disp[top:top+crop_h, left:left+crop_w]
        
        # Load FoundationStereo teacher disparity
        pt_filename = image_name.replace('.png', '.pt')
        teach_disp_path = os.path.join(teach_disp_dir, pt_filename)
        
        if not os.path.exists(teach_disp_path):
            continue
            
        teach_disp_pt = torch.load(teach_disp_path, weights_only=True).float()
        if teach_disp_pt.dim() == 3: teach_disp_pt = teach_disp_pt.unsqueeze(0)
        
        teach_disp_up = logits2disparity(teach_disp_pt, (crop_h, crop_w)) * 512.0
        teach_disp_np = np.nan_to_num(teach_disp_up.squeeze().cpu().numpy(), nan=0.0)
        
        # Load STTR disparity and confidence maps
        sttr_disp_path = os.path.join(sttr_disp_dir, image_name)
        sttr_conf_path = os.path.join(sttr_conf_dir, image_name)
        
        sttr_disp = None
        sttr_conf = None
        
        if os.path.exists(sttr_disp_path):
            sttr_disp = cv2.imread(sttr_disp_path, cv2.IMREAD_UNCHANGED)
            if sttr_disp is not None:
                sttr_disp = sttr_disp[top:top+crop_h, left:left+crop_w]
        
        if os.path.exists(sttr_conf_path):
            sttr_conf = cv2.imread(sttr_conf_path, cv2.IMREAD_GRAYSCALE)
            if sttr_conf is not None:
                sttr_conf = sttr_conf[top:top+crop_h, left:left+crop_w]
        
        if sttr_disp is None or sttr_conf is None:
            continue
        
        # --- 1. FoundationStereo Disparity Overlay ---
        valid_disp_fs = (fs_disp > 0) if fs_disp is not None else np.zeros((crop_h, crop_w), dtype=bool)
        if valid_disp_fs.any():
            valid_vals_fs = fs_disp[valid_disp_fs].astype(np.float32) / 128.0
            disp_norm_fs = np.zeros(fs_disp.shape, dtype=np.uint8)
            disp_norm_fs[valid_disp_fs] = np.clip(valid_vals_fs * (255.0 / max_disparity), 0, 255).astype(np.uint8)
            
            disp_color_fs = cv2.applyColorMap(disp_norm_fs, cv2.COLORMAP_MAGMA)
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, disp_color_fs, 0.4, 0)
                fs_disp_overlay = np.where(valid_disp_fs[:, :, None], blended, left_img)
            else:
                fs_disp_overlay = np.where(valid_disp_fs[:, :, None], disp_color_fs, np.zeros_like(left_img))
        else:
            fs_disp_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)
        
        # --- 2. STTR Disparity Overlay ---
        valid_disp_sttr = (sttr_disp > 0)
        if valid_disp_sttr.any():
            valid_vals_sttr = sttr_disp[valid_disp_sttr].astype(np.float32) / 128.0
            disp_norm_sttr = np.zeros(sttr_disp.shape, dtype=np.uint8)
            disp_norm_sttr[valid_disp_sttr] = np.clip(valid_vals_sttr * (255.0 / max_disparity), 0, 255).astype(np.uint8)
            
            disp_color_sttr = cv2.applyColorMap(disp_norm_sttr, cv2.COLORMAP_MAGMA)
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, disp_color_sttr, 0.4, 0)
                sttr_disp_overlay = np.where(valid_disp_sttr[:, :, None], blended, left_img)
            else:
                sttr_disp_overlay = np.where(valid_disp_sttr[:, :, None], disp_color_sttr, np.zeros_like(left_img))
        else:
            sttr_disp_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)
        
        # --- 3. FoundationStereo Disparity Logit Overlay ---
        # Same as disparity teacher logits in other videos - use MAGMA colormap
        valid_disp_fs = (teach_disp_np > 0)
        if valid_disp_fs.any():
            valid_vals_fs = teach_disp_np[valid_disp_fs]
            disp_norm_fs = np.zeros(teach_disp_np.shape, dtype=np.uint8)
            disp_norm_fs[valid_disp_fs] = np.clip(valid_vals_fs * (255.0 / max_disparity), 0, 255).astype(np.uint8)
            
            disp_color_fs = cv2.applyColorMap(disp_norm_fs, cv2.COLORMAP_MAGMA)
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, disp_color_fs, 0.4, 0)
                fs_logit_overlay = np.where(valid_disp_fs[:, :, None], blended, left_img)
            else:
                fs_logit_overlay = np.where(valid_disp_fs[:, :, None], disp_color_fs, np.zeros_like(left_img))
        else:
            fs_logit_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)
        
        # --- 4. STTR Confidence Map Overlay ---
        # Use our custom bright Red/Yellow/Green LUT
        sttr_conf_color = BRIGHT_RYG_LUT[sttr_conf]
        
        sttr_conf_overlay = cv2.addWeighted(left_img, 0.6, sttr_conf_color, 0.4, 0)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(fs_disp_overlay, 'FoundationStereo Prediction', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(sttr_disp_overlay, 'STTR Prediction', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(fs_logit_overlay, 'FoundationStereo Teacher Logits', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(sttr_conf_overlay, 'STTR Confidence Map', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 2x2 Grid: [FS Disp | STTR Disp] \n [FS Logit | STTR Conf]
        top_row = np.hstack((fs_disp_overlay, sttr_disp_overlay))
        bottom_row = np.hstack((fs_logit_overlay, sttr_conf_overlay))
        combined_frame = np.vstack((top_row, bottom_row))
        
        video_out.write(combined_frame)
        
    video_out.release()
    

# %%
# FoundationStereo Teacher Confidence Comparison

mode = 'test' # 'train' 'val' 'test'
dataset_path = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}'
fs_conf_output_video_path = f'/data/Zeitler/Visualization/videos/FoundationStereo_Confidence/{mode}'
os.makedirs(fs_conf_output_video_path, exist_ok=True)

for subset in sorted(os.listdir(dataset_path)):
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.isdir(subset_path):
        continue
    
    left_dir = os.path.join(subset_path, 'input', 'left_images')
    disp_dir = os.path.join(subset_path, 'target', 'disparity')
    teach_disp_dir = os.path.join(subset_path, 'teacher', 'disparity_128_256_256')
    
    left_images = sorted(os.listdir(left_dir))
    if not left_images:
        continue
        
    sample_img = cv2.imread(os.path.join(left_dir, left_images[0]))
    h, w = sample_img.shape[:2]
    crop_h, crop_w = 1024, 1024
    
    # Offsets for center crop
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    
    video_out = cv2.VideoWriter(
        os.path.join(fs_conf_output_video_path, f'{subset}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (crop_w * 2, crop_h * 2)
    )
    
    for image_name in tqdm(left_images, desc=f'Processing {subset} FS Confidence'): 
        left_img = cv2.imread(os.path.join(left_dir, image_name))
        disp_img = cv2.imread(os.path.join(disp_dir, image_name), cv2.IMREAD_UNCHANGED)
        
        if left_img is None or disp_img is None:
            continue
            
        # Center crop the images
        left_img = left_img[top:top+crop_h, left:left+crop_w]
        disp_img = disp_img[top:top+crop_h, left:left+crop_w]
            
        # Load teacher logits
        pt_filename = image_name.replace('.png', '.pt')
        teach_disp_path = os.path.join(teach_disp_dir, pt_filename)
        
        if not os.path.exists(teach_disp_path):
            continue
            
        teach_disp_pt = torch.load(teach_disp_path, weights_only=True).float()
        if teach_disp_pt.dim() == 3: teach_disp_pt = teach_disp_pt.unsqueeze(0)
        
        # Calculate disparity and confidence
        teach_disp_up = logits2disparity(teach_disp_pt, (crop_h, crop_w)) * 512.0
        teach_disp_np = np.nan_to_num(teach_disp_up.squeeze().cpu().numpy(), nan=0.0)
        
        # Confidence map
        teach_disp_logits_up = F.interpolate(teach_disp_pt, size=(crop_h, crop_w), mode='bilinear', align_corners=False)
        conf_pt = F.softmax(teach_disp_logits_up, dim=1).max(dim=1)[0]
        conf_np = conf_pt.squeeze().cpu().numpy()
        
        # --- 1. Left Image Overlay ---
        # Actually just left_img 
        
        # --- 2. FoundationStereo Disparity Overlay ---
        valid_disp_fs = (disp_img > 0)
        if valid_disp_fs.any():
            valid_vals_fs = disp_img[valid_disp_fs].astype(np.float32) / 128.0
            disp_norm_fs = np.zeros(disp_img.shape, dtype=np.uint8)
            disp_norm_fs[valid_disp_fs] = np.clip(valid_vals_fs * (255.0 / max_disparity), 0, 255).astype(np.uint8)
            
            disp_color_fs = cv2.applyColorMap(disp_norm_fs, cv2.COLORMAP_MAGMA)
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, disp_color_fs, 0.4, 0)
                disp_overlay = np.where(valid_disp_fs[:, :, None], blended, left_img)
            else:
                disp_overlay = np.where(valid_disp_fs[:, :, None], disp_color_fs, np.zeros_like(left_img))
        else:
            disp_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)
            
        # --- 3. FoundationStereo Teacher Logits Overlay ---
        valid_disp_t = (teach_disp_np > 0)
        if valid_disp_t.any():
            valid_vals_t = teach_disp_np[valid_disp_t]
            disp_norm_t = np.zeros(teach_disp_np.shape, dtype=np.uint8)
            disp_norm_t[valid_disp_t] = np.clip(valid_vals_t * (255.0 / max_disparity), 0, 255).astype(np.uint8)
            
            disp_color_t = cv2.applyColorMap(disp_norm_t, cv2.COLORMAP_MAGMA)
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, disp_color_t, 0.4, 0)
                logit_overlay = np.where(valid_disp_t[:, :, None], blended, left_img)
            else:
                logit_overlay = np.where(valid_disp_t[:, :, None], disp_color_t, np.zeros_like(left_img))
        else:
            logit_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)
            
        # --- 4. FoundationStereo Confidence Map Overlay ---
        # conf_np is in [0, 1]
        conf_scaled = np.clip(conf_np * 255, 0, 255).astype(np.uint8)
        conf_color = BRIGHT_RYG_LUT[conf_scaled]
        conf_overlay = cv2.addWeighted(left_img, 0.6, conf_color, 0.4, 0)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_labeled = left_img.copy()
        mean_conf = conf_np.mean()
        cv2.putText(left_labeled, 'Left Image', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp_overlay, 'FoundationStereo Disparity Pseudo Target', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(logit_overlay, 'FoundationStereo Teacher Logits', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(conf_overlay, f'FoundationStereo Confidence Map (Mean: {mean_conf:.0%})', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 2x2 Grid: [Left Image | FS Disp] \n [FS Logit | FS Conf]
        top_row = np.hstack((left_labeled, disp_overlay))
        bottom_row = np.hstack((logit_overlay, conf_overlay))
        combined_frame = np.vstack((top_row, bottom_row))
        
        video_out.write(combined_frame)
        
    video_out.release()
    

# %%
# Consistency Check Videos

def left_right_consistency_check(left_disp, right_disp, threshold=3.0):
    B, C, H, W = left_disp.shape
    x_grid = torch.linspace(-1, 1, W, device=left_disp.device)
    y_grid = torch.linspace(-1, 1, H, device=left_disp.device)
    y, x = torch.meshgrid(y_grid, x_grid, indexing='ij')
    grid = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    
    normalized_disp = (left_disp.squeeze(1) / (W / 2)).unsqueeze(-1)
    shifted_grid = grid.clone()
    shifted_grid[..., 0] -= normalized_disp[..., 0]
    
    warped_disp_right = torch.nn.functional.grid_sample(right_disp, shifted_grid, align_corners=True, padding_mode='zeros')
    
    diff = torch.abs(left_disp - warped_disp_right)
    
    valid_mask = (diff < threshold).float()
    
    left_disp_valid = left_disp > 0
    valid_mask[~left_disp_valid] = torch.nan
    valid_warped_right = warped_disp_right > 0
    valid_mask[~valid_warped_right] = torch.nan
    
    return valid_mask

mode = 'test' # 'train' or 'test'
dataset_path = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}'
consistency_output_video_path = f'/data/Zeitler/Visualization/videos/FoundationStereo_Consistency/{mode}'
os.makedirs(consistency_output_video_path, exist_ok=True)

for subset in sorted(os.listdir(dataset_path)):
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.isdir(subset_path):
        continue
    
    left_dir = os.path.join(subset_path, 'input', 'left_images')
    right_dir = os.path.join(subset_path, 'input', 'right_images')
    disp_dir = os.path.join(subset_path, 'target', 'disparity')
    disp_right_dir = os.path.join(subset_path, 'target', 'disparity_right')
    
    left_images = sorted(os.listdir(left_dir))
    if not left_images:
        continue
        
    sample_img = cv2.imread(os.path.join(left_dir, left_images[0]))
    h, w = sample_img.shape[:2]
    
    video_out = cv2.VideoWriter(
        os.path.join(consistency_output_video_path, f'{subset}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w * 2, h * 2)
    )
    
    for image_name in tqdm(left_images, desc=f'Processing {subset} Consistency'): 
        left_img = cv2.imread(os.path.join(left_dir, image_name))
        right_img = cv2.imread(os.path.join(right_dir, image_name))
        disp_img = cv2.imread(os.path.join(disp_dir, image_name), cv2.IMREAD_UNCHANGED)
        disp_right_img = cv2.imread(os.path.join(disp_right_dir, image_name), cv2.IMREAD_UNCHANGED)
        
        if left_img is None or right_img is None or disp_img is None or disp_right_img is None:
            continue
            
        # Format left and right disparity for PyTorch
        left_disp_pt = torch.from_numpy(np.where(disp_img > 0, disp_img.astype(np.float32) / 128.0, 0.0)).unsqueeze(0).unsqueeze(0)
        right_disp_pt = torch.from_numpy(np.where(disp_right_img > 0, disp_right_img.astype(np.float32) / 128.0, 0.0)).unsqueeze(0).unsqueeze(0)

        # Calculate consistency map
        valid_mask = left_right_consistency_check(left_disp_pt, right_disp_pt, threshold=3.0)
        agreed_count = valid_mask.nansum()
        possible_count = (~torch.isnan(valid_mask)).sum()
        
        if possible_count > 0:
            mean_consistency = (agreed_count / possible_count).item()
        else:
            mean_consistency = float('nan')
            
        # --- 1. Left Image Overlay ---
        left_labeled = left_img.copy()
        
        # --- 2. Right Image Overlay ---
        right_labeled = right_img.copy()
        
        # --- 3. Left Disparity Overlay ---
        valid_disp_left = (disp_img > 0)
        if valid_disp_left.any():
            valid_vals_left = disp_img[valid_disp_left].astype(np.float32) / 128.0
            disp_norm_left = np.zeros(disp_img.shape, dtype=np.uint8)
            disp_norm_left[valid_disp_left] = np.clip(valid_vals_left * (255.0 / max_disparity), 0, 255).astype(np.uint8)
            
            disp_color_left = cv2.applyColorMap(disp_norm_left, cv2.COLORMAP_MAGMA)
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, disp_color_left, 0.4, 0)
                disp_left_overlay = np.where(valid_disp_left[:, :, None], blended, left_img)
            else:
                disp_left_overlay = np.where(valid_disp_left[:, :, None], disp_color_left, np.zeros_like(left_img))
        else:
            disp_left_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)
            
        # --- 4. Right Disparity Overlay ---
        valid_disp_right = (disp_right_img > 0)
        if valid_disp_right.any():
            valid_vals_right = disp_right_img[valid_disp_right].astype(np.float32) / 128.0
            disp_norm_right = np.zeros(disp_right_img.shape, dtype=np.uint8)
            disp_norm_right[valid_disp_right] = np.clip(valid_vals_right * (255.0 / max_disparity), 0, 255).astype(np.uint8)
            
            disp_color_right = cv2.applyColorMap(disp_norm_right, cv2.COLORMAP_MAGMA)
            if overlay_disparity:
                blended = cv2.addWeighted(right_img, 0.6, disp_color_right, 0.4, 0)
                disp_right_overlay = np.where(valid_disp_right[:, :, None], blended, right_img)
            else:
                disp_right_overlay = np.where(valid_disp_right[:, :, None], disp_color_right, np.zeros_like(right_img))
        else:
            disp_right_overlay = right_img.copy() if overlay_disparity else np.zeros_like(right_img)
            
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(left_labeled, 'Left Image', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right_labeled, 'Right Image', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        if np.isnan(mean_consistency):
            cv2.putText(disp_left_overlay, 'FoundationStereo Pseudo GT Left-Sided', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(disp_left_overlay, f'FoundationStereo Pseudo GT Left-Sided (3px-Consistency: {mean_consistency:.0%})', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.putText(disp_right_overlay, 'FoundationStereo Pseudo GT Right-Sided', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 2x2 Grid: [Left Image | Right Image] \n [Left Disp | Right Disp]
        top_row = np.hstack((left_labeled, right_labeled))
        bottom_row = np.hstack((disp_left_overlay, disp_right_overlay))
        combined_frame = np.vstack((top_row, bottom_row))
        
        video_out.write(combined_frame)
        
    video_out.release()

# %%
# Single Dataset Left/Right Video (Depth on Left, Confidence on Right)
import torch.nn.functional as F

import subprocess

fps = 4
mode = 'val'
dataset_name = 'instrument_dataset_5'
dataset_path = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}/{dataset_name}'
conference_output_video_path = f'/data/Zeitler/Visualization/videos/conference/{mode}/{dataset_name}'
os.makedirs(conference_output_video_path, exist_ok=True)

left_dir = os.path.join(dataset_path, 'input', 'left_images')
teach_disp_dir = os.path.join(dataset_path, 'teacher', 'disparity_128_256_256')

left_images = sorted(os.listdir(left_dir))

if left_images:
    sample_img = cv2.imread(os.path.join(left_dir, left_images[0]))
    h, w = sample_img.shape[:2]
    crop_h, crop_w = 1024, 1024
    
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)

    # Read calibration
    with open(os.path.join(dataset_path, 'calibration', 'foundation_stereo_calibration.txt'), 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
    
    temp_output_path = os.path.join(conference_output_video_path, f'{dataset_name}_depth_conf_temp.mp4')
    final_output_path = os.path.join(conference_output_video_path, f'{dataset_name}_depth_conf.mp4')
    
    video_out = cv2.VideoWriter(
        temp_output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (crop_w * 2, crop_h)
    )
    
    max_depth_cm = 15.0
    max_depth_mm = max_depth_cm * 10.0 # 150.0 mm
    
    for image_name in tqdm(left_images, desc=f'Processing Conference Video {dataset_name}'): 
        left_img = cv2.imread(os.path.join(left_dir, image_name))
        
        if left_img is None:
            continue
            
        left_img = left_img[top:top+crop_h, left:left+crop_w]
            
        pt_filename = image_name.replace('.png', '.pt')
        teach_disp_path = os.path.join(teach_disp_dir, pt_filename)
        
        if not os.path.exists(teach_disp_path):
            continue
            
        teach_disp_pt = torch.load(teach_disp_path, weights_only=True).float()
        if teach_disp_pt.dim() == 3: teach_disp_pt = teach_disp_pt.unsqueeze(0)
        
        teach_disp_up = logits2disparity(teach_disp_pt, (crop_h, crop_w)) * 512.0
        teach_disp_np = np.nan_to_num(teach_disp_up.squeeze().cpu().numpy(), nan=0.0)
        
        # Confidence map
        teach_disp_logits_up = F.interpolate(teach_disp_pt, size=(crop_h, crop_w), mode='bilinear', align_corners=False)
        conf_pt = F.softmax(teach_disp_logits_up, dim=1).max(dim=1)[0]
        conf_np = conf_pt.squeeze().cpu().numpy()
        
        # --- Left: Depth Overlay ---
        valid_disp_t = (teach_disp_np > 0)
        depth = np.zeros_like(teach_disp_np)
        depth[valid_disp_t] = (K[0,0] * baseline * 1000.0) / teach_disp_np[valid_disp_t]
        
        if valid_disp_t.any():
            depth_vals = depth[valid_disp_t]
            depth_norm = np.zeros(depth.shape, dtype=np.uint8)
            depth_norm[valid_disp_t] = 255 - np.clip(depth_vals * (255.0 / max_depth_mm), 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
            
            # Force invalid/NaN pixels to pure black background if not overlaying
            heatmap[~valid_disp_t] = [0, 0, 0]
            
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, heatmap, 0.4, 0)
                depth_overlay = np.where(valid_disp_t[:, :, None], blended, left_img)
            else:
                depth_overlay = heatmap
        else:
            if overlay_disparity:
                depth_overlay = left_img.copy()
            else:
                depth_overlay = np.zeros((*depth.shape, 3), dtype=np.uint8)
            
        # --- Right: Confidence Map Overlay ---
        conf_scaled = np.clip(conf_np * 255, 0, 255).astype(np.uint8)
        conf_color = BRIGHT_RYG_LUT[conf_scaled]
        conf_overlay = cv2.addWeighted(left_img, 0.6, conf_color, 0.4, 0)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        mean_conf = conf_np.mean()
        
        cv2.putText(depth_overlay, 'FoundationStereo Teacher Logits as Depth', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(conf_overlay, f'FoundationStereo Teacher Confidence (Mean: {mean_conf:.0%})', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        combined_frame = np.hstack((depth_overlay, conf_overlay))
        video_out.write(combined_frame)
        
    video_out.release()
    
    # Convert to H.264 using FFmpeg for web compatibility
    print("Converting to H.264 for web optimization using FFmpeg...")
    if os.path.exists(final_output_path):
        os.remove(final_output_path)
    
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_output_path,
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
        final_output_path
    ], check=True, capture_output=True)
    
    # Clean up temp file
    os.remove(temp_output_path)
    print(f"Finished! Web-ready video saved to: {final_output_path}")


# %%
# Conference Video 1: Segmentation Teacher Logits (Left) vs Segmentation Target (Right)
dataset_path = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}/{dataset_name}'
conference_output_video_path = f'/data/Zeitler/Visualization/videos/conference/{mode}/{dataset_name}'
os.makedirs(conference_output_video_path, exist_ok=True)

left_dir = os.path.join(dataset_path, 'input', 'left_images')
seg_dir = os.path.join(dataset_path, 'target', 'segmentation')
teach_seg_dir = os.path.join(dataset_path, 'teacher', 'segmentation_2_256_256')

left_images = sorted(os.listdir(left_dir))

if left_images:
    sample_img = cv2.imread(os.path.join(left_dir, left_images[0]))
    h, w = sample_img.shape[:2]
    crop_h, crop_w = 1024, 1024
    
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)

    temp_output_path = os.path.join(conference_output_video_path, f'{dataset_name}_seg.mp4')
    final_output_path = os.path.join(conference_output_video_path, f'{dataset_name}_seg_web.mp4')
    
    video_out = cv2.VideoWriter(
        temp_output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (crop_w * 2, crop_h)
    )
    
    for image_name in tqdm(left_images, desc=f'Processing Conference Seg Video {dataset_name}'): 
        left_img = cv2.imread(os.path.join(left_dir, image_name))
        seg_img = cv2.imread(os.path.join(seg_dir, image_name), cv2.IMREAD_GRAYSCALE)
        
        if left_img is None or seg_img is None:
            continue
            
        left_img = left_img[top:top+crop_h, left:left+crop_w]
        seg_img = seg_img[top:top+crop_h, left:left+crop_w]
            
        pt_filename = image_name.replace('.png', '.pt')
        teach_seg_path = os.path.join(teach_seg_dir, pt_filename)
        
        if not os.path.exists(teach_seg_path):
            continue
            
        teach_seg_pt = torch.load(teach_seg_path, weights_only=True).float()
        if teach_seg_pt.dim() == 3: teach_seg_pt = teach_seg_pt.unsqueeze(0)
        
        teach_seg_up = upsample_logits(teach_seg_pt, (crop_h, crop_w))
        teach_seg_mask = teach_seg_up.argmax(dim=1).squeeze().cpu().numpy()
        
        # --- Left: Segmentation Target ---
        seg_gt_mask = (seg_img == 1)
        if seg_gt_mask.any():
            color_mask = np.zeros_like(left_img)
            color_mask[seg_gt_mask] = seg_color
            blended = cv2.addWeighted(left_img, 0.6, color_mask, 0.4, 0)
            seg_gt_overlay = np.where(seg_gt_mask[:, :, None], blended, left_img)
        else:
            seg_gt_overlay = left_img.copy()
            
        # --- Right: Segmentation Teacher ---
        seg_t_mask = (teach_seg_mask == 1)
        if seg_t_mask.any():
            color_mask = np.zeros_like(left_img)
            color_mask[seg_t_mask] = seg_color
            blended = cv2.addWeighted(left_img, 0.6, color_mask, 0.4, 0)
            seg_t_overlay = np.where(seg_t_mask[:, :, None], blended, left_img)
        else:
            seg_t_overlay = left_img.copy()
            
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(seg_gt_overlay, 'Segmentation Target', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(seg_t_overlay, 'SegFormer-B5 Teacher Logits as Mask', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        combined_frame = np.hstack((seg_t_overlay, seg_gt_overlay))
        video_out.write(combined_frame)
        
    video_out.release()
    
    print("Converting to H.264 for web optimization using FFmpeg...")
    if os.path.exists(final_output_path):
        os.remove(final_output_path)
    
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_output_path,
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
        final_output_path
    ], check=True, capture_output=True)
    os.remove(temp_output_path)
    print(f"Finished! Web-ready video saved to: {final_output_path}")

# %%
# Conference Video 2: FS Teacher Logits as Depth (Left) vs FoundationStereo Pseudo Targets (Right)

disp_dir = os.path.join(dataset_path, 'target', 'disparity')
teach_disp_dir = os.path.join(dataset_path, 'teacher', 'disparity_128_256_256')

if left_images:
    sample_img = cv2.imread(os.path.join(left_dir, left_images[0]))
    h, w = sample_img.shape[:2]
    crop_h, crop_w = 1024, 1024
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)

    with open(os.path.join(dataset_path, 'calibration', 'foundation_stereo_calibration.txt'), 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
    
    temp_output_path = os.path.join(conference_output_video_path, f'{dataset_name}_pseudo_vs_teach_depth_temp.mp4')
    final_output_path = os.path.join(conference_output_video_path, f'{dataset_name}_pseudo_vs_teach_depth_web.mp4')

    video_out = cv2.VideoWriter(
        temp_output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (crop_w * 2, crop_h)
    )
    
    max_depth_cm = 15.0
    max_depth_mm = max_depth_cm * 10.0 # 150.0 mm
    
    for image_name in tqdm(left_images, desc=f'Processing Conference Disp vs Teach Video {dataset_name}'): 
        left_img = cv2.imread(os.path.join(left_dir, image_name))
        disp_img = cv2.imread(os.path.join(disp_dir, image_name), cv2.IMREAD_UNCHANGED)
        
        if left_img is None or disp_img is None:
            continue
            
        left_img = left_img[top:top+crop_h, left:left+crop_w]
        disp_img = disp_img[top:top+crop_h, left:left+crop_w]
            
        pt_filename = image_name.replace('.png', '.pt')
        teach_disp_path = os.path.join(teach_disp_dir, pt_filename)
        
        if not os.path.exists(teach_disp_path):
            continue
            
        # Target Disparity to Depth
        disp_img_np = np.where(disp_img > 0, disp_img.astype(np.float32) / 128.0, 0.0)
        valid_disp_gt = (disp_img_np > 0)
        
        depth_gt = np.zeros_like(disp_img_np)
        depth_gt[valid_disp_gt] = (K[0,0] * baseline * 1000.0) / disp_img_np[valid_disp_gt]

        # Teacher Disparity to Depth
        teach_disp_pt = torch.load(teach_disp_path, weights_only=True).float()
        if teach_disp_pt.dim() == 3: teach_disp_pt = teach_disp_pt.unsqueeze(0)
        
        teach_disp_up = logits2disparity(teach_disp_pt, (crop_h, crop_w)) * 512.0
        teach_disp_np = np.nan_to_num(teach_disp_up.squeeze().cpu().numpy(), nan=0.0)
        
        valid_disp_t = (teach_disp_np > 0)
        depth_t = np.zeros_like(teach_disp_np)
        depth_t[valid_disp_t] = (K[0,0] * baseline * 1000.0) / teach_disp_np[valid_disp_t]
        
        
        # --- Left: FS Pseudo Target Depth Overlay ---
        if valid_disp_gt.any():
            depth_vals_gt = depth_gt[valid_disp_gt]
            depth_norm_gt = np.zeros(depth_gt.shape, dtype=np.uint8)
            depth_norm_gt[valid_disp_gt] = 255 - np.clip(depth_vals_gt * (255.0 / max_depth_mm), 0, 255).astype(np.uint8)
            heatmap_gt = cv2.applyColorMap(depth_norm_gt, cv2.COLORMAP_MAGMA)
            
            heatmap_gt[~valid_disp_gt] = [0, 0, 0]
            
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, heatmap_gt, 0.4, 0)
                depth_gt_overlay = np.where(valid_disp_gt[:, :, None], blended, left_img)
            else:
                depth_gt_overlay = heatmap_gt
        else:
            if overlay_disparity:
                depth_gt_overlay = left_img.copy()
            else:
                depth_gt_overlay = np.zeros((*depth_gt.shape, 3), dtype=np.uint8)
            
        # --- Right: FS Teacher Logits Depth Overlay ---
        if valid_disp_t.any():
            depth_vals_t = depth_t[valid_disp_t]
            depth_norm_t = np.zeros(depth_t.shape, dtype=np.uint8)
            depth_norm_t[valid_disp_t] = 255 - np.clip(depth_vals_t * (255.0 / max_depth_mm), 0, 255).astype(np.uint8)
            heatmap_t = cv2.applyColorMap(depth_norm_t, cv2.COLORMAP_MAGMA)
            
            heatmap_t[~valid_disp_t] = [0, 0, 0]
            
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, heatmap_t, 0.4, 0)
                depth_t_overlay = np.where(valid_disp_t[:, :, None], blended, left_img)
            else:
                depth_t_overlay = heatmap_t
        else:
            if overlay_disparity:
                depth_t_overlay = left_img.copy()
            else:
                depth_t_overlay = np.zeros((*depth_t.shape, 3), dtype=np.uint8)
            
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(depth_gt_overlay, 'FoundationStereo Pseudo Target Depth', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_t_overlay, 'FoundationStereo Teacher Logits as Depth', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        combined_frame = np.hstack((depth_t_overlay, depth_gt_overlay))
        video_out.write(combined_frame)
        
    video_out.release()
    
    print("Converting to H.264 for web optimization using FFmpeg...")
    if os.path.exists(final_output_path):
        os.remove(final_output_path)
    
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_output_path,
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
        final_output_path
    ], check=True, capture_output=True)
    
    os.remove(temp_output_path)
    print(f"Finished! Web-ready video saved to: {final_output_path}")

# %%
