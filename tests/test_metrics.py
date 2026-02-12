import math
import torch
from metrics.segmentation import IoU, Dice
from metrics.disparity import EPE, Bad3, MAE

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

    # Target
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
    # Target:
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
    assert math.isclose(results['mean'], 0.5)
    assert math.isclose(results['std'], 0.0)
    
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

    # Target
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
    # Target:
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
    assert math.isclose(results['mean'], 0.5)
    assert math.isclose(results['std'], 0.0)
    
def test_missing_class():
    iou_metric = IoU(n_classes=3)
    iou_metric.reset()
    dice_metric = Dice(n_classes=3)
    dice_metric.reset()
    
    # Prediction (logits)
    # Shape: (N, C, H, W) -> (1, 3, 2, 2)
    output = torch.tensor(
        [
            [
                [ # Class 0 logits
                    [1.0, 1.0], # Row 0 => Winner class 0 => [0, 0]
                    [0.0, 0.0]  # Row 1
                ], 
                [ # Class 1 logits
                    [0.0, 0.0], # Row 0
                    [1.0, 0.0]  # Row 1 => Winner class 1 => [1, ~]
                ],
                [ # Class 2 logits
                    [0.0, 0.0], # Row 0
                    [0.0, 1.0]  # Row 1 => Winner class 2 => [~, 2]
                ]
            ]
        ]
    )
    
    # Target
    target = torch.tensor(
        [
            [
                [0, 0],
                [1, 1]
            ]
        ]
    )
    
    iou_metric.update(output, target)
    iou_results = iou_metric.compute()
    dice_metric.update(output, target)
    dice_results = dice_metric.compute()
    
    # Prediction:
    # [0, 0]
    # [1, 2]
    # Target:
    # [0, 0]
    # [1, 1]
    #
    # Confusion Matrix:
    #            Pred 0  | Pred 1 | Pred 2
    # Target 0:  2 (TP0) | 0 (FN1) | 0
    # Target 1:  0       | 1 (TP1) | 1 (FN2)
    # Target 2:  0       | 0      | 0
    # IoU:
    # Class 0: TP0=2, FP=0, FN=0 -> IoU = TP / (TP + FP + FN) = 2 / (2 + 0 + 0) = 2/2 = 1.0
    # Class 1: TP1=1, FP=0, FN1=1 -> IoU = TP / (TP + FP + FN) = 1 / (1 + 0 + 1) = 1/2 = 0.5
    # Class 2: TP2=0, FP=1, FN2=0 -> IoU = TP / (TP + FP + FN) = 0 / (0 + 1 + 0) = 0/1 = 0.0
    # mIoU = (1.0 + 0.5) / 2 = 0.75 (0.0 of class 2 ignored)
    # DICE:
    # Class 0: TP0=2, FP=0, FN=0 -> Dice = (2*TP) / (2*TP + FP + FN) = 4 / (4 + 0 + 0) = 4/4 = 1.0
    # Class 1: TP1=1, FP=0, FN1=1 -> Dice = (2*TP) / (2*TP + FP + FN) = 2 / (2 + 0 + 1) = 2/3 = 0.6667
    # Class 2: TP2=0, FP=1, FN2=0 -> Dice = (2*TP) / (2*TP + FP + FN) = 0 / (0 + 1 + 0) = 0/1 = 0.0
    # mDICE = (1.0 + 0.6667) / 2 = 0.83335 (0.0 of class 2 ignored)
    
    assert math.isclose(iou_results[0], 1.0)
    assert math.isclose(iou_results[1], 0.5)
    assert math.isclose(iou_results[2], 0.0)
    assert math.isclose(iou_results['mean'], 0.75)
    
    assert math.isclose(dice_results[0], 1.0)
    assert math.isclose(dice_results[1], 2/3, rel_tol=1e-4)
    assert math.isclose(dice_results[2], 0.0)
    assert math.isclose(dice_results['mean'], (1.0 + 2/3) / 2, rel_tol=1e-4)

def test_disparity_metrics():
    epe_metric = EPE()
    bad3_metric = Bad3()
    mae_metric = MAE()
    
    epe_metric.reset()
    bad3_metric.reset()
    mae_metric.reset()
    
    # Predictions (normalized 0-1)
    # Shape: (N, C, H, W) -> (1, 1, 2, 2)
    output = torch.tensor(
        [
            [
                [
                    [0.5, 0.25], 
                    [0.125, 0.0]
                ]
            ]
        ]
    )
    
    # Target (normalized 0-1)
    target = torch.tensor(
        [
            [
                [
                    [0.5, 0.5], 
                    [0.25, 0.0]
                ]
            ]
        ]
    )
                             
    # Instrinsic Parameters
    baseline = torch.tensor(0.5)       
    focal_length = torch.tensor(1024.0)
    # baseline * focal_length = 512
    
    epe_metric.update(output, target, baseline, focal_length)
    bad3_metric.update(output, target, baseline, focal_length)
    mae_metric.update(output, target, baseline, focal_length)
    
    epe_result = epe_metric.compute()
    bad3_result = bad3_metric.compute()
    mae_result = mae_metric.compute()
    
    # Calculations (max_disparity = 512):
    # Prediction (px):
    # [256.0, 128.0]
    # [64.0, 0.0]
    # Target (px):
    # [256.0, 256.0]
    # [128.0, 0.0 (ignored)]
    #
    # Valid Mask (GT > 0):
    # [1, 1]
    # [1, 0] -> N_valid = 3
    #
    # EPE (End Point Error):
    # Diff = |Pred - GT|
    # [|256-256|=0.0, |128-256|=128.0]
    # [|64-128|=64.0, 0.0]
    # Sum = 0.0 + 128.0 + 64.0 = 192.0
    # EPE = Sum / N_valid = 192.0 / 3 = 64.0
    #
    # Bad3 (> 3px Error):
    # mask = Diff > 3.0
    # [0.0>3 (0), 128.0>3 (1)]
    # [64.0>3 (1), 0.0]
    # Sum Bad = 0 + 1 + 1 = 2
    # Bad3 = Sum Bad / N_valid = 2 / 3 = 0.6666
    #
    # Depth MAE (Mean Absolute Error):
    # Depth = (Baseline * Focal Length) / Disparity = 512 / Disparity
    # Predicted Depth:
    # [512/256=2.0, 512/128=4.0]
    # [512/64=8.0, 0.0]
    # Target Depth:
    # [512/256=2.0, 512/256=2.0]
    # [512/128=4.0, 0.0)]
    # Depth Diff = |Pred - Target|
    # [|2-2|=0.0, |4-2|=2.0]
    # [|8-4|=4.0, 0.0]
    # Sum = 0.0 + 2.0 + 4.0 = 6.0
    # MAE = Sum / N_valid = 6.0 / 3 = 2.0
    
    assert math.isclose(epe_result['EPE_pixel'], 64.0)
    assert math.isclose(bad3_result['Bad3_rate'], 2/3, rel_tol=1e-4)
    assert math.isclose(mae_result['MAE_mm'], 2.0)
