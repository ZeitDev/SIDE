import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

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
    def __init__(self, temperature_start: float = 1.0, temperature_end: float = 4.0, total_epochs: int = 100, steps_per_epoch: int = 100):
        super().__init__()
        self.current_step = 0
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.total_epoch_steps = total_epochs * steps_per_epoch
        self.criterion = nn.KLDivLoss(reduction='none', log_target=False)
    
    def get_current_temperature(self):
        if self.current_step >= self.total_epoch_steps: return self.temperature_end
        
        progress = self.current_step / self.total_epoch_steps
        
        temperature = self.temperature_start + (self.temperature_end - self.temperature_start) * progress
        return temperature
    
    def forward(self, student_logits, teacher_logits, targets, teacher_confidence):
        T = self.get_current_temperature()
        
        B, D, H, W = student_logits.shape
        teacher_logits = teacher_logits.detach()
        
        targets = F.interpolate(targets, size=(H, W), mode='nearest-exact') # Sample down to 1/4 resolution
        valid = targets > 0 # removes occlusion from left border and later maybe instruments as well, as they dont perform that good on depth?
        
        if valid.sum() == 0:
            return (student_logits * 0.0).sum()
        
        student_logits = student_logits / T
        teacher_logits = teacher_logits / T
        
        student_log_probabilities = F.log_softmax(student_logits, dim=1)
        teacher_probabilities = F.softmax(teacher_logits, dim=1)
        
        pixel_loss = self.criterion(student_log_probabilities, teacher_probabilities)
        weighted_pixel_loss = pixel_loss * teacher_confidence
        valid_pixel_loss = weighted_pixel_loss * valid.float()
        
        loss = (valid_pixel_loss.sum() / valid.sum()) * (T ** 2)
        
        if self.training: self.current_step += 1
        
        return loss
    
class EntropyConfidence(nn.Module):
    def __init__(self, c_min=0.3900, c_max=0.8981):
        super().__init__()
        self.c_min = c_min
        self.c_max = c_max
        self.denominator = max(self.c_max - self.c_min, 1e-6)

    def forward(self, logits, prob_dim=1):
        probs = F.softmax(logits, dim=prob_dim)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=prob_dim, keepdim=True)
        
        num_bins = logits.shape[prob_dim]
        max_entropy = math.log2(num_bins)
        base_confidence = 1.0 - (entropy / max_entropy)
        
        scaled_conf = (base_confidence - self.c_min) / self.denominator
        return torch.clamp(scaled_conf, min=0.0, max=1.0)