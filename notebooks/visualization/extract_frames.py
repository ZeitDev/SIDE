# %%
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import mlflow
from PIL import Image

# Ensure we can import from `utils` if running this notebook specifically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.helpers import upsample_logits, logits2disparity
from utils import helpers
from data.transforms import build_transforms

# Create a bright Red-Yellow-Green colormap LUT (0=Red, 127=Yellow, 255=Green)
_lut_x = np.linspace(0, 1, 256)
_lut_r = np.clip(2.0 - 2.0 * _lut_x, 0, 1) * 255
_lut_g = np.clip(2.0 * _lut_x, 0, 1) * 200
_lut_b = np.zeros_like(_lut_x)
BRIGHT_RYG_LUT = np.stack([_lut_b, _lut_g, _lut_r], axis=1).astype(np.uint8) # BGR format

max_disparity = 512.0
seg_color = [255, 255, 0] # BGR format
overlay_disparity = False

# %%
def extract_frame(base_path, dataset_name, frame_id, output_dir, model=None, transform=None, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)
    
    image_name = f'image{frame_id:03d}.png'
    pt_filename = f'image{frame_id:03d}.pt'
    
    # Paths
    dataset_path = os.path.join(base_path, dataset_name)
    left_path = os.path.join(dataset_path, 'input', 'left_images', image_name)
    right_path = os.path.join(dataset_path, 'input', 'right_images', image_name)
    seg_path = os.path.join(dataset_path, 'target', 'segmentation', image_name)
    disp_path = os.path.join(dataset_path, 'target', 'disparity', image_name)
    teach_disp_path = os.path.join(dataset_path, 'teacher', 'disparity_128_256_256', pt_filename)
    
    # Check if files exist
    if not os.path.exists(left_path):
        print(f"File not found: {left_path}")
        return
        
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    # Calculate center crop
    h, w = left_img.shape[:2]
    crop_h, crop_w = 1024, 1024
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    
    # Apply crop
    left_img = left_img[top:top+crop_h, left:left+crop_w]
    cv2.imwrite(os.path.join(output_dir, 'left_image.png'), left_img)
    
    if right_img is not None:
        right_img = right_img[top:top+crop_h, left:left+crop_w]
        cv2.imwrite(os.path.join(output_dir, 'right_image.png'), right_img)
    
    # Segmentation Target
    if os.path.exists(seg_path):
        seg_img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        seg_img = seg_img[top:top+crop_h, left:left+crop_w]
        seg_mask = (seg_img == 1)
        
        # Pure segmentation mask
        pure_seg = np.zeros_like(left_img)
        pure_seg[seg_mask] = seg_color
        cv2.imwrite(os.path.join(output_dir, 'segmentation_pure.png'), pure_seg)
        
        if seg_mask.any():
            color_mask = np.zeros_like(left_img)
            color_mask[seg_mask] = seg_color
            blended = cv2.addWeighted(left_img, 0.6, color_mask, 0.4, 0)
            seg_overlay = np.where(seg_mask[:, :, None], blended, left_img)
        else:
            seg_overlay = left_img.copy()
            
        cv2.imwrite(os.path.join(output_dir, 'segmentation_overlay.png'), seg_overlay)
        
    # Disparity Target
    if os.path.exists(disp_path):
        disp_img = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        disp_img = disp_img[top:top+crop_h, left:left+crop_w]
        valid_disp_mask = (disp_img > 0)
        
        if valid_disp_mask.any():
            valid_vals = disp_img[valid_disp_mask].astype(np.float32) / 128.0
            disp_norm_full = np.zeros(disp_img.shape, dtype=np.uint8)
            disp_norm_full[valid_disp_mask] = np.clip(valid_vals * (255.0 / max_disparity), 0, 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_norm_full, cv2.COLORMAP_MAGMA)
            
            if overlay_disparity:
                blended = cv2.addWeighted(left_img, 0.6, disp_color, 0.4, 0)
                disp_overlay = np.where(valid_disp_mask[:, :, None], blended, left_img)
            else:
                disp_overlay = np.where(valid_disp_mask[:, :, None], disp_color, np.zeros_like(left_img))
                
            pure_disp = np.zeros_like(left_img)
            pure_disp[valid_disp_mask] = disp_color[valid_disp_mask]
            cv2.imwrite(os.path.join(output_dir, 'disparity_pure.png'), pure_disp)
        else:
            disp_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)
            
        cv2.imwrite(os.path.join(output_dir, 'disparity_target.png'), disp_overlay)
        
    # Teacher Confidence Map
    if os.path.exists(teach_disp_path):
        teach_disp_pt = torch.load(teach_disp_path, map_location='cpu')
        # Compatibility handling
        if isinstance(teach_disp_pt, dict):
            pass 
        elif hasattr(teach_disp_pt, 'float'):
            teach_disp_pt = teach_disp_pt.float()
        
        if teach_disp_pt.dim() == 3: teach_disp_pt = teach_disp_pt.unsqueeze(0)
        
        # Calculate confidence
        teach_disp_logits_up = F.interpolate(teach_disp_pt, size=(crop_h, crop_w), mode='bilinear', align_corners=False)
        conf_pt = F.softmax(teach_disp_logits_up, dim=1).max(dim=1)[0]
        conf_np = conf_pt.squeeze().detach().cpu().numpy()
        
        conf_scaled = np.clip(conf_np * 255, 0, 255).astype(np.uint8)
        conf_color = BRIGHT_RYG_LUT[conf_scaled]
        
        # Pure confidence map
        cv2.imwrite(os.path.join(output_dir, 'teacher_confidence_pure.png'), conf_color)
        
        # Conf overlay
        conf_overlay = cv2.addWeighted(left_img, 0.6, conf_color, 0.4, 0)
        cv2.imwrite(os.path.join(output_dir, 'teacher_confidence_overlay.png'), conf_overlay)
        
    # --- MODEL PREDICTIONS ---
    if model is not None and transform is not None:
        print("Running model inference...")
        img_pil = np.array(Image.open(left_path).convert('RGB'))
        right_img_pil = np.array(Image.open(right_path).convert('RGB')) if os.path.exists(right_path) else None
        
        data_dict = {'image': img_pil}
        if right_img_pil is not None:
            data_dict['right_image'] = right_img_pil
            
        data_dict = transform(**data_dict)
        
        img_t = data_dict['image'].unsqueeze(0).to(device)
        right_img_t = data_dict['right_image'].unsqueeze(0).to(device) if 'right_image' in data_dict else None
        
        with torch.no_grad():
            output = model(img_t, right_img_t)
            
        if 'segmentation' in output:
            out_seg = output['segmentation']
            pred_seg_mask = out_seg.argmax(dim=1).squeeze().cpu().numpy()
            
            # Pure prediction segmentation
            pred_pure_seg = np.zeros_like(left_img)
            pred_seg_bool = (pred_seg_mask == 1)
            pred_pure_seg[pred_seg_bool] = seg_color
            cv2.imwrite(os.path.join(output_dir, 'prediction_segmentation_pure.png'), pred_pure_seg)
            
            if pred_seg_bool.any():
                color_mask = np.zeros_like(left_img)
                color_mask[pred_seg_bool] = seg_color
                blended = cv2.addWeighted(left_img, 0.6, color_mask, 0.4, 0)
                pred_seg_overlay = np.where(pred_seg_bool[:, :, None], blended, left_img)
            else:
                pred_seg_overlay = left_img.copy()
            cv2.imwrite(os.path.join(output_dir, 'prediction_segmentation_overlay.png'), pred_seg_overlay)
            
        if 'disparity' in output:
            out_disp = output['disparity']
            pred_disp = out_disp.squeeze().cpu().numpy() * 512.0
            
            valid_pred_disp = (pred_disp > 0)
            if valid_pred_disp.any():
                valid_vals_pred = pred_disp[valid_pred_disp].astype(np.float32)
                disp_norm_full_pred = np.zeros(pred_disp.shape, dtype=np.uint8)
                disp_norm_full_pred[valid_pred_disp] = np.clip(valid_vals_pred * (255.0 / max_disparity), 0, 255).astype(np.uint8)
                pred_disp_color = cv2.applyColorMap(disp_norm_full_pred, cv2.COLORMAP_MAGMA)
                
                pure_pred_disp = np.zeros_like(left_img)
                pure_pred_disp[valid_pred_disp] = pred_disp_color[valid_pred_disp]
                cv2.imwrite(os.path.join(output_dir, 'prediction_disparity_pure.png'), pure_pred_disp)
                
                if overlay_disparity:
                    blended = cv2.addWeighted(left_img, 0.6, pred_disp_color, 0.4, 0)
                    pred_disp_overlay = np.where(valid_pred_disp[:, :, None], blended, left_img)
                else:
                    pred_disp_overlay = np.where(valid_pred_disp[:, :, None], pred_disp_color, np.zeros_like(left_img))
            else:
                pred_disp_overlay = left_img.copy() if overlay_disparity else np.zeros_like(left_img)
                
            cv2.imwrite(os.path.join(output_dir, 'prediction_disparity_overlay.png'), pred_disp_overlay)
        
    print(f"Extraction for {dataset_name} frame {frame_id} completed successfully.")
    print(f"Check output directory: {output_dir}")

# %%
base_dataset_path = '/data/Zeitler/SIDED/EndoVis17/processed'
dataset_to_extract = 'val/instrument_dataset_5'
frame_number = 199
output_folder = f'/data/Zeitler/Visualization/extracted_frames/{dataset_to_extract.replace("/", "_")}_frame{frame_number}'

# Ensure MLflow is configured with correct tracking URI
from utils.setup import setup_environment
current_dir = os.getcwd()
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)
os.chdir(current_dir)

# Model setup
arch = 'convnext'
run = 'wMT-KD/260406:2036/train' # Update to run id of choice
task_mode = 'combined' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading model from {run}...")
os.chdir('/data/Zeitler/code/SIDE') # Make sure we're in root for MLflow/YAML
model_run_id = helpers.get_model_run_id(run)
model = mlflow.pytorch.load_model(f'runs:/{model_run_id}/best_model_{task_mode}').to(device)
model.eval()

# Load config to build identical transforms
config_name = run.split('/')[0]
with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
with open(os.path.join('configs', arch, config_name + '.yaml'), 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)
test_transform = build_transforms(config, mode='test')
os.chdir(current_dir) # Change back

extract_frame(
    base_path=base_dataset_path, 
    dataset_name=dataset_to_extract, 
    frame_id=frame_number, 
    output_dir=output_folder,
    model=model,
    transform=test_transform,
    device=device
)
