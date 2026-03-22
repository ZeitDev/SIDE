import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import logits2disparity

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        valid_mask = targets > 0
        
        if valid_mask.sum() == 0:
            return (outputs * 0.0).sum() 
        
        valid_outputs = outputs[valid_mask]
        valid_targets = targets[valid_mask]
        
        loss = F.l1_loss(valid_outputs, valid_targets, reduction='mean')
        
        return loss

class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, max_disparity: float = 512, beta: float = 1.0):
        super().__init__()
        self.max_disparity = max_disparity
        self.beta = beta

    def forward(self, outputs, targets):      
        valid_mask = (targets > 0).float()
        
        if valid_mask.sum() == 0:
            return (outputs * 0.0).sum() 
        
        loss = F.smooth_l1_loss(outputs, targets, reduction='none', beta=self.beta)
        loss = loss * valid_mask
        
        valid_count = valid_mask.sum()
        loss = loss.sum() / valid_count
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
        
        if valid.sum() == 0:
            return (student_logits * 0.0).sum()
        
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