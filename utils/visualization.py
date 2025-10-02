import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def _tensor_to_array(tensor):
    """Convert a PyTorch tensor to a NumPy array for visualization."""
    # Move tensor to CPU, convert to numpy, and change from (C, H, W) to (H, W, C)
    return tensor.cpu().numpy().transpose(1, 2, 0)

def _logits_to_binary(logits):
    """Convert model output logits to a binary mask."""
    # Apply sigmoid, threshold at 0.5, move to CPU, and remove channel dimension
    return (torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze()

def get_combined_overlay(image, gt_mask, pred_mask, alpha=0.5):
    """
    Generates an overlay combining ground truth and prediction masks on an image.
    - Green: False Negative (Ground Truth only)
    - Red: False Positive (Prediction only)
    - Yellow: True Positive (Overlap)
    """
    # Ensure image is in 8-bit format for color operations
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Ensure masks are boolean
    gt_mask_bool = gt_mask.astype(bool)
    pred_mask_bool = pred_mask.astype(bool)

    # Create a color mask
    color_mask = np.zeros_like(image)
    color_mask[gt_mask_bool & ~pred_mask_bool] = [0, 255, 0]  # Green for False Negatives
    color_mask[~gt_mask_bool & pred_mask_bool] = [255, 0, 0]  # Red for False Positives
    color_mask[gt_mask_bool & pred_mask_bool] = [255, 255, 0] # Yellow for True Positives

    # Create a boolean mask of all areas to be overlayed
    overlay_region = gt_mask_bool | pred_mask_bool
    
    # Blend the original image with the color mask
    overlay = image.copy()
    overlay[overlay_region] = cv2.addWeighted(
        src1=image[overlay_region],
        alpha=1 - alpha,
        src2=color_mask[overlay_region],
        beta=alpha,
        gamma=0
    )
    return overlay

def image_mask_overlay_figure(image, mask, output, epoch=None):
    """
    Create a matplotlib figure showing the image, ground truth, and a combined overlay.

    Args:
        image (torch.Tensor): The input image tensor.
        mask (torch.Tensor): The ground truth mask tensor.
        output (torch.Tensor): The model output tensor.
        epoch (int, optional): The current epoch number for title annotation.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    if epoch is not None:
        fig.suptitle(f'Validation Results | Epoch {epoch}', fontsize=16)
    
    # Convert tensors to numpy arrays for processing
    image_arr = _tensor_to_array(image)
    mask_arr = mask.cpu().numpy().squeeze()
    output_arr = _logits_to_binary(output)
    
    # Generate the combined overlay
    combined_overlay = get_combined_overlay(image_arr, mask_arr, output_arr)
    
    # Plot Image
    ax[0].imshow(image_arr)
    ax[0].set_title('Image')
    ax[0].axis('off')
    
    # Plot Ground Truth Mask
    ax[1].imshow(mask_arr, cmap='gray')
    ax[1].set_title('Ground Truth Mask')
    ax[1].axis('off')
    
    # Plot Combined Overlay
    ax[2].imshow(combined_overlay)
    ax[2].set_title('Combined Overlay (TP/FP/FN)')
    ax[2].axis('off')
    
    plt.tight_layout()
    
    return fig