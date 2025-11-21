import torch
import math
from metrics.segmentation import IoU, Dice

def test_metric_reset():
    iou_metric = IoU(n_classes=2)
    
    output = torch.tensor([[[[0.1, 0.9], [0.8, 0.2]], [[0.9, 0.1], [0.2, 0.8]]]])
    target = torch.tensor([[[1, 0], [0, 1]]])
    iou_metric.update(output, target)
    
    assert not torch.all(iou_metric.confusion_matrix == 0)
    
    iou_metric.reset()
    
    assert torch.all(iou_metric.confusion_matrix == 0)

def test_iou_metric():
    iou_metric = IoU(n_classes=2)
    iou_metric.reset()

    # Predictions (logits)
    # Shape: (N, C, H, W) -> (1, 2, 2, 3)
    output = torch.tensor(
        [
            [
                [ # Class 0 logits
                    [1.0, 1.0, 1.0], # Row 0 => Winner class 0 => [0, 0, 0]
                    [0.0, 0.0, 0.0]  # Row 1
                ], 
                [ # Class 1 logits
                    [0.0, 0.0, 0.0], # Row 0
                    [1.0, 1.0, 1.0]  # Row 1 => Winner class 1 => [1, 1, 1]
                ]
            ]
        ]
    )
    # Output Argmax: [0, 0, 0]
    #                [1, 1, 1]

    # Ground truth
    # Shape: (N, H, W) -> (1, 2, 3)
    target = torch.tensor(
        [
            [
                [0, 0, 1],
                [1, 1, 0]
            ]
        ]
    )

    iou_metric.update(output, target)
    results = iou_metric.compute()

    # Prediction:
    # [0, 0, 0]
    # [1, 1, 1]
    # Ground Truth:
    # [0, 0, 1]
    # [1, 1, 0]
    #
    # Confusion Matrix:
    #            Pred 0  | Pred 1
    # Target 0:  2 (TP0) | 1 (FN)
    # Target 1:  1 (FP)  | 2 (TP1)
    #
    # Class 0: TP0=2, FP=1, FN=1 -> IoU = TP / (TP + FP + FN) = 2 / (2+1+1) = 2/4 = 0.5
    # Class 1: TP1=2, FP=1, FN=1 -> IoU = TP / (TP + FP + FN) = 2 / (2+1+1) = 2/4 = 0.5
    # mIoU = (0.5 + 0.5) / 2 = 0.5

    assert math.isclose(results[0], 0.5)
    assert math.isclose(results[1], 0.5)
    assert math.isclose(results['mIoU'], 0.5)
    
def test_dice_metric():
    dice_metric = Dice(n_classes=2)
    dice_metric.reset()

    # Predictions (logits)
    output = torch.tensor(
        [
            [
                [ # Class 0 logits
                    [1.0, 1.0], # Row 0 => Winner class 0 => [0, 0]
                    [0.0, 0.0]  # Row 1
                ], 
                [ # Class 1 logits
                    [0.0, 0.0], # Row 0
                    [1.0, 1.0]  # Row 1 => Winner class 1 => [1, 1]
                ]
            ]
        ]
    )

    # Ground truth
    target = torch.tensor(
        [
            [
                [1, 0],
                [1, 0]
            ]
        ]
    )

    dice_metric.update(output, target)
    results = dice_metric.compute()

    # Prediction:
    # [0, 0]
    # [1, 1]
    # Ground Truth:
    # [1, 0]
    # [1, 0]
    #
    # Confusion Matrix:
    #            Pred 0  | Pred 1
    # Target 0:  1 (TP0) | 1 (FN)
    # Target 1:  1 (FP)  | 1 (TP1)
    #
    # Class 0: TP0=1, FP=1, FN=1 -> Dice = (2*TP) / (2*TP + FP + FN) = 2 / (2 + 1 + 1) = 2/4 = 0.5
    # Class 1: TP1=1, FP=1, FN=1 -> Dice = (2*TP) / (2*TP + FP + FN) = 2 / (2 + 1 + 1) = 2/4 = 0.5
    # mDICE = (0.6667 + 0.6667) / 2 = 0.6667

    assert math.isclose(results[0], 0.5)
    assert math.isclose(results[1], 0.5)
    assert math.isclose(results['mDICE'], 0.5)