import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import Optional

def _get_target_overlay(image: np.ndarray, target: np.ndarray, n_classes: int, alpha: float = 0.5) -> np.ndarray:
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    overlay = image.copy()
    colors = plt.cm.get_cmap('hot')
    
    color_mask = np.zeros_like(image)
    for class_id in range(1, n_classes):
        class_mask = target == class_id
        if np.any(class_mask):
            color = (np.array(colors(class_id / n_classes))[:3] * 255).astype(np.uint8)
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

def _get_output_overlay(image: np.ndarray, target: np.ndarray, output: np.ndarray, alpha: float = 0.5) -> np.ndarray:
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

def get_image_target_output_overlay(image: torch.Tensor, target: torch.Tensor, output: torch.Tensor, n_classes: int = 0, epoch: Optional[int] = None, index: Optional[int] = None) -> Figure:
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'Validation Overlay | Epoch {epoch} | Index {index}', fontsize=16)
    
    image_arr = _scale_image(image).numpy().transpose(1, 2, 0)
    target_arr = target.numpy().squeeze()
    output_arr = torch.argmax(output, dim=0).numpy()
    
    target_overlay = _get_target_overlay(image_arr, target_arr, n_classes)
    output_overlay = _get_output_overlay(image_arr, target_arr, output_arr)
    
    ax[0].imshow(image_arr)
    ax[0].set_title('Image')
    ax[0].axis('off')
    
    ax[1].imshow(target_arr, cmap='hot', vmin=0, vmax=n_classes)
    ax[1].set_title('Ground Truth Mask')
    ax[1].axis('off')
    
    ax[2].imshow(output_arr, cmap='hot', vmin=0, vmax=n_classes)
    ax[2].set_title('Predicted Mask')
    ax[2].axis('off')
    
    ax[3].imshow(target_overlay)
    ax[3].set_title('Ground Truth Overlay')
    ax[3].axis('off')
    
    ax[4].imshow(output_overlay)
    ax[4].set_title('Predicted Overlay')
    ax[4].axis('off')
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    return fig