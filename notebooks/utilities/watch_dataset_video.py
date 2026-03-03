# %%
import os
import cv2
import numpy as np

from tqdm import tqdm

# %%

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
    seg_dir = os.path.join(subset_path, 'ground_truth', 'segmentation')
    disp_dir = os.path.join(subset_path, 'ground_truth', 'disparity')
    
    left_images = sorted(os.listdir(left_dir))
    if not left_images:
        continue
        
    sample_img = cv2.imread(os.path.join(left_dir, left_images[0]))
    h, w = sample_img.shape[:2]
    
    video_out = cv2.VideoWriter(
        os.path.join(output_video_path, f'{subset}.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),
        15, # Adjust FPS to your preference
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
            # Apply a solid color (e.g., Red) to the segmented area
            color_mask = np.zeros_like(left_img)
            color_mask[seg_mask] = [0, 0, 255] # Red in BGR
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
            
            # Fixed absolute scaling using a max threshold (e.g. 128 pixels maps to 255)
            disp_norm_full = np.zeros(disp_img.shape, dtype=np.uint8)
            disp_norm_full[valid_disp_mask] = np.clip(valid_vals * (255.0 / 128.0), 0, 255).astype(np.uint8)
                
            disp_color = cv2.applyColorMap(disp_norm_full, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(left_img, 0.6, disp_color, 0.4, 0)
            disp_overlay = np.where(valid_disp_mask[:, :, None], blended, left_img)
        else:
            disp_overlay = left_img.copy()
            
        # Add labels to the images before stacking
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(left_img, 'Left Image', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right_img, 'Right Image', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(seg_overlay, 'Segmentation Ground Truth', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp_overlay, 'Disparity FoundationStereo Pseudo Ground Truth', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Make a 2x2 grid view
        top_row = np.hstack((left_img, right_img))
        bottom_row = np.hstack((seg_overlay, disp_overlay))
        combined_frame = np.vstack((top_row, bottom_row))
        
        video_out.write(combined_frame)
        
    video_out.release()
    
# %% new videos
# with 'teacher', 'segmentation_2_256_256', 'disparity_128_256_256' overlays
import sys
import torch

# Ensure we can import from `utils` if running this notebook specifically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.helpers import upsample_logits, logits2disparity

mode = 'test' # 'train' or 'test'
dataset_path = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}'
teacher_output_video_path = f'/data/Zeitler/Visualization/videos/teacher/{mode}'
os.makedirs(teacher_output_video_path, exist_ok=True)

for subset in sorted(os.listdir(dataset_path)):
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.isdir(subset_path):
        continue
    
    left_dir = os.path.join(subset_path, 'input', 'left_images')
    seg_dir = os.path.join(subset_path, 'ground_truth', 'segmentation')
    disp_dir = os.path.join(subset_path, 'ground_truth', 'disparity')
    
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
        15,
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
            color_mask[seg_gt_mask] = [0, 0, 255] # Red
            blended = cv2.addWeighted(left_img, 0.6, color_mask, 0.4, 0)
            seg_gt_overlay = np.where(seg_gt_mask[:, :, None], blended, left_img)
        else:
            seg_gt_overlay = left_img.copy()

        # --- 2. Segmentation Teacher Overlay ---
        seg_t_mask = (teach_seg_mask == 1)
        if seg_t_mask.any():
            color_mask_t = np.zeros_like(left_img)
            color_mask_t[seg_t_mask] = [0, 0, 255] # Red
            blended = cv2.addWeighted(left_img, 0.6, color_mask_t, 0.4, 0)
            seg_t_overlay = np.where(seg_t_mask[:, :, None], blended, left_img)
        else:
            seg_t_overlay = left_img.copy()
            
        # --- 3. Disparity GT Overlay ---
        valid_disp_gt = (disp_img > 0)
        if valid_disp_gt.any():
            valid_vals = disp_img[valid_disp_gt].astype(np.float32) / 128.0
            
            disp_norm_full = np.zeros(disp_img.shape, dtype=np.uint8)
            # Use 128.0 max scaling to make disparity visible
            disp_norm_full[valid_disp_gt] = np.clip(valid_vals * (255.0 / 128.0), 0, 255).astype(np.uint8)
            
            disp_color = cv2.applyColorMap(disp_norm_full, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(left_img, 0.6, disp_color, 0.4, 0)
            disp_gt_overlay = np.where(valid_disp_gt[:, :, None], blended, left_img)
        else:
            disp_gt_overlay = left_img.copy()
            
        # --- 4. Disparity Teacher Overlay ---
        valid_disp_t = (teach_disp_np > 0)
        if valid_disp_t.any():
            valid_vals_t = teach_disp_np[valid_disp_t]
            
            disp_norm_full_t = np.zeros(teach_disp_np.shape, dtype=np.uint8)
            # Use 128.0 max scaling to make disparity visible
            disp_norm_full_t[valid_disp_t] = np.clip(valid_vals_t * (255.0 / 128.0), 0, 255).astype(np.uint8)
            
            disp_color_t = cv2.applyColorMap(disp_norm_full_t, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(left_img, 0.6, disp_color_t, 0.4, 0)
            disp_t_overlay = np.where(valid_disp_t[:, :, None], blended, left_img)
        else:
            disp_t_overlay = left_img.copy()

        # Add labels to the images before stacking
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(seg_gt_overlay, 'Segmentation Ground Truth', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(seg_t_overlay, 'Segmentation SegFormer-B4 Teacher Logits', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp_gt_overlay, 'Disparity FoundationStereo Pseudo Ground Truth', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp_t_overlay, 'Disparity FoundationStereo Teacher Logits', (20, 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        # 2x2 Grid: [Seg GT | Seg Teacher] \n [Disp GT | Disp Teacher]
        top_row = np.hstack((seg_gt_overlay, seg_t_overlay))
        bottom_row = np.hstack((disp_gt_overlay, disp_t_overlay))
        combined_frame = np.vstack((top_row, bottom_row))
        
        video_out.write(combined_frame)
        
    video_out.release()
# %%
