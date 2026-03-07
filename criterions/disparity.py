import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import logits2disparity

class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, max_disparity: float = 512, ignore_value: float = 0, reduction: str = 'mean', beta: float = 1.0):
        super().__init__()
        self.max_disparity = max_disparity
        self.ignore_value = ignore_value
        self.reduction = reduction
        self.beta = beta

    def forward(self, outputs, targets):
        outputs_px = outputs * self.max_disparity
        targets_px = targets * self.max_disparity
        
        valid_mask = (targets_px != self.ignore_value).float()
        loss = F.smooth_l1_loss(outputs_px, targets_px, reduction='none', beta=self.beta)
        loss = loss * valid_mask
        
        if self.reduction == 'mean':
            valid_count = valid_mask.sum()
            return loss.sum() / valid_count
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class PixelWiseKLDivLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction='none', log_target=False)
        
    def forward(self, student_logits, teacher_logits, targets):
        B, D, H, W = student_logits.shape
        teacher_logits = teacher_logits.detach()
        
        targets = F.interpolate(targets, size=(H, W), mode='nearest-exact') # Sample down to 1/4 resolution
        valid = targets > 0 # removes occlusion from left to right and later maybe instruments as well, as they dont perform that good on depth?
        
        raw_teacher_probabilities = F.softmax(teacher_logits, dim=1)
        teacher_confidence = raw_teacher_probabilities.max(dim=1, keepdim=True)[0]
        
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature  
        
        # student = logits2disparity(student_logits, size=targets.shape[2:])
        # teacher = logits2disparity(teacher_logits, size=targets.shape[2:])
        
        student_log_probabilities = F.log_softmax(student_logits, dim=1)
        teacher_probabilities = F.softmax(teacher_logits, dim=1)
        
        pixel_loss = self.criterion(student_log_probabilities, teacher_probabilities)
        weighted_pixel_loss = pixel_loss * teacher_confidence
        valid_pixel_loss = weighted_pixel_loss * valid.float()
        
        loss = (valid_pixel_loss.sum() / valid.sum()) * (self.temperature ** 2)
        
        return loss