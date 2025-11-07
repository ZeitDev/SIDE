import torch
from typing import Dict, Optional

class SegmentationMetric:
    def __init__(self, num_classes: int, device: torch.device, ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        # ? output, target = output.detach(), target.detach() - not needed?
        outputs = torch.argmax(output, dim=1)
        
        outputs = outputs.reshape(-1)
        target = target.reshape(-1)
        
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            outputs = outputs[mask]
            target = target[mask]
            
        _confusion_matrix = torch.bincount(
            self.num_classes * target + outputs,
            minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += _confusion_matrix

    def compute(self) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        self.confusion_matrix.fill_(0)


class IoU(SegmentationMetric):
    def compute(self) -> Dict[str, float]:
        tp = torch.diag(self.confusion_matrix).float()
        fp = (self.confusion_matrix.sum(dim=0) - tp).float()
        fn = (self.confusion_matrix.sum(dim=1) - tp).float()
        
        iou_per_class = tp / (tp + fp + fn)
        
        results = {}
        for i, iou in enumerate(iou_per_class): results[i] = iou.item()
            
        valid_iou = iou_per_class[~torch.isnan(iou_per_class)]
        mean_iou = valid_iou.mean().item()
        results['mIoU'] = mean_iou
        
        return results

class Dice(SegmentationMetric):
    def compute(self) -> Dict[str, float]:
        tp = torch.diag(self.confusion_matrix).float()
        fp = (self.confusion_matrix.sum(dim=0) - tp).float()
        fn = (self.confusion_matrix.sum(dim=1) - tp).float()
        
        dice_per_class = (2 * tp) / (2 * tp + fp + fn)

        results = {}
        for i, dice in enumerate(dice_per_class): results[i] = dice.item()

        valid_dice = dice_per_class[~torch.isnan(dice_per_class)]
        mean_dice = valid_dice.mean().item()
        results['mDICE'] = mean_dice
            
        return results