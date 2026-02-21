import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import soft_argmin

class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, ignore_value=0, reduction='mean', beta=1.0):
        super().__init__()
        self.ignore_value = ignore_value
        self.reduction = reduction
        self.beta = beta
        
    def forward(self, output_logits, targets):
        predictions = soft_argmin(output_logits)
        
        valid_mask = (targets != self.ignore_value).float()
        loss = F.smooth_l1_loss(predictions, targets, reduction='none', beta=self.beta)
        loss = loss * valid_mask
        
        if self.reduction == 'mean':
            valid_count = valid_mask.sum()
            return loss.sum() / valid_count
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class PixelWiseKLDivLoss(nn.Module):
    def __init__(self, temperature=1.0, scale=1.0):
        super().__init__()
        self.temperature = temperature
        self.scale = scale
        self.criterion = nn.KLDivLoss(reduction='sum', log_target=False)
        
    def forward(self, student_logits, teacher_logits):
        if self.scale != 1.0: student_logits = F.avg_pool2d(student_logits, kernel_size=int(1/self.scale), stride=int(1/self.scale))
        
        b, d, h, w = student_logits.shape
        
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature
        
        student_log_probabilities = F.log_softmax(student_logits, dim=1)
        with torch.no_grad(): teacher_probabilities = F.softmax(teacher_logits, dim=1)
        
        loss = self.criterion(student_log_probabilities, teacher_probabilities)
        loss = (loss / (b * h * w)) * (self.temperature ** 2)
        
        return loss