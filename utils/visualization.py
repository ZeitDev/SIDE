import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import Optional, Dict, Any

def _get_segmentation_raw_overlay(image: np.ndarray, target: np.ndarray, num_of_segmentation_classes: int, alpha: float = 0.5) -> np.ndarray:
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    overlay = image.copy()
    colors = plt.cm.get_cmap('hot')
    
    color_mask = np.zeros_like(image)
    for class_id in range(1, num_of_segmentation_classes):
        class_mask = target == class_id
        if np.any(class_mask):
            color = (np.array(colors(class_id / num_of_segmentation_classes))[:3] * 255).astype(np.uint8)
            color_mask[class_mask] = color

    overlay_region = target > 0
    
    if np.any(overlay_region):
        overlay[overlay_region] = cv2.addWeighted(
            src1=image[overlay_region],
            alpha=1 - alpha,
            src2=color_mask[overlay_region],
            beta=alpha,
            gamma=0
        )
        
    return overlay

def _get_segmentation_error_overlay(image: np.ndarray, target: np.ndarray, output: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    true_classifications = (target == output)
    false_classifications = (target != output)

    color_mask = np.zeros_like(image)
    color_mask[true_classifications] = [0, 255, 0]
    color_mask[false_classifications] = [255, 0, 0]

    overlay_region = true_classifications | false_classifications
    overlay = image.copy()
    
    if np.any(overlay_region):
        overlay[overlay_region] = cv2.addWeighted(
            src1=image[overlay_region],
            alpha=1 - alpha,
            src2=color_mask[overlay_region],
            beta=alpha,
            gamma=0
        )
        
    return overlay

def _scale_image(image: torch.Tensor) -> torch.Tensor:
    return (image - image.min()) / (image.max() - image.min())

def _get_disparity_raw_overlay(image: np.ndarray, disparity_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    overlay = image.copy()
    disparity_normalized = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min())
    disparity_colored = plt.cm.jet(disparity_normalized)[:, :, :3]
    disparity_colored = (disparity_colored * 255).astype(np.uint8)

    overlay_region = disparity_map > 0
    
    if np.any(overlay_region):
        overlay[overlay_region] = cv2.addWeighted(
            src1=image[overlay_region],
            alpha=1 - alpha,
            src2=disparity_colored[overlay_region],
            beta=alpha,
            gamma=0
        )
        
    return overlay

def _get_disparity_error_overlay(image: np.ndarray, target: np.ndarray, output: np.ndarray, max_disparity: float, alpha: float = 0.5) -> np.ndarray:
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    error_map = np.abs(target - output)
    
    # We want Red to indicate a specific error in PIXELS (e.g., 10px).
    # Since input maps are normalized [0, 1], we map the pixel threshold to this range.
    error_threshold_pixels = 3.0
    max_vis_error = error_threshold_pixels / max_disparity if max_disparity > 0 else 0.1
    
    error_normalized = np.clip(error_map / max_vis_error, 0, 1)
    
    # Use RdYlGn_r colormap: 0.0 is Green (Low Error), 1.0 is Red (High Error)
    error_colored = plt.cm.RdYlGn_r(error_normalized)[:, :, :3]
    error_colored = (error_colored * 255).astype(np.uint8)

    # Apply overlay to all valid pixels (target > 0), so correct pixels appear Green
    overlay_region = target != 0
    
    overlay = image.copy()
    if np.any(overlay_region):
        overlay[overlay_region] = cv2.addWeighted(
            src1=image[overlay_region],
            alpha=1 - alpha,
            src2=error_colored[overlay_region],
            beta=alpha,
            gamma=0
        )
        
    return overlay

def get_multitask_visuals(image: torch.Tensor, targets: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], num_of_segmentation_classes: Int = 2, epoch: Optional[int] = None, index: Optional[int] = None, max_disparity: float = 1.0) -> Figure:
    tasks = [t for t in ['segmentation', 'disparity'] if t in targets]
    n_rows = len(tasks)
    
    image_arr = _scale_image(image).numpy().transpose(1, 2, 0)
    h, w, _ = image_arr.shape
    aspect_ratio = w / h
    
    row_height = 4.0
    fig_width = row_height * aspect_ratio * 3
    fig_height = row_height * n_rows
    
    fig, ax = plt.subplots(n_rows, 3, figsize=(fig_width, fig_height), gridspec_kw={'wspace': 0.02, 'hspace': 0.15})
    
    if n_rows == 1: ax = np.expand_dims(ax, axis=0)

    fig.suptitle(f'Validation Overlay | Epoch {epoch} | Index {index}', fontsize=16)
    
    for i, task in enumerate(tasks):
        if task == 'segmentation':
            segmentation_target_array = targets['segmentation'].numpy().squeeze()
            segmentation_prediction_array = torch.argmax(outputs['segmentation'], dim=0).numpy()
            
            segmentation_target_overlay = _get_segmentation_raw_overlay(image_arr, segmentation_target_array, num_of_segmentation_classes)
            segmentation_prediction_overlay = _get_segmentation_raw_overlay(image_arr, segmentation_prediction_array, num_of_segmentation_classes)
            segmentation_accuracy_overlay = _get_segmentation_error_overlay(image_arr, segmentation_target_array, segmentation_prediction_array)
            
            ax[i, 0].imshow(segmentation_target_overlay)
            ax[i, 0].set_title('Segmentation Target')
            ax[i, 0].axis('off')
            
            ax[i, 1].imshow(segmentation_prediction_overlay)
            ax[i, 1].set_title('Segmentation Prediction')
            ax[i, 1].axis('off')
            
            ax[i, 2].imshow(segmentation_accuracy_overlay)
            ax[i, 2].set_title('Segmentation Error')
            ax[i, 2].axis('off')
            
        elif task == 'disparity':
            disparity_target_array = targets['disparity'].numpy().squeeze()
            disparity_prediction_array = outputs['disparity'].numpy().squeeze()
            
            disparity_target_overlay = _get_disparity_raw_overlay(image_arr, disparity_target_array)
            disparity_prediction_overlay = _get_disparity_raw_overlay(image_arr, disparity_prediction_array)
            disparity_error_overlay = _get_disparity_error_overlay(image_arr, disparity_target_array, disparity_prediction_array, max_disparity)
            
            ax[i, 0].imshow(disparity_target_overlay)
            ax[i, 0].set_title('Disparity Target')
            ax[i, 0].axis('off')
            
            ax[i, 1].imshow(disparity_prediction_overlay)
            ax[i, 1].set_title('Disparity Prediction')
            ax[i, 1].axis('off')
            
            ax[i, 2].imshow(disparity_error_overlay)
            ax[i, 2].set_title('Disparity 3 Pixel Error')
            ax[i, 2].axis('off')
    
    plt.tight_layout()
    
    return fig