import torch.nn as nn
import torch.nn.functional as F

class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, ignore_value=0, reduction='mean', beta=1.0):
        super().__init__()
        self.ignore_value = ignore_value
        self.reduction = reduction
        self.beta = beta
        
    def forward(self, prediction, target):
        valid_mask = target != self.ignore_value
        loss = F.smooth_l1_loss(prediction, target, reduction='none', beta=self.beta)
        loss = loss * valid_mask
        
        if self.reduction == 'mean':
            valid_count = valid_mask.sum()
            return loss.sum() / valid_count
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss