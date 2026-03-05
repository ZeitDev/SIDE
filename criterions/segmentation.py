import torch.nn as nn
import torch.nn.functional as F

from utils.helpers import upsample_logits

class ChannelWiseKLDivLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction='sum', log_target=False)

    def forward(self, student_logits, teacher_logits):
        B, C, H, W = student_logits.shape
        
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature
        
        teacher_logits = upsample_logits(teacher_logits, size=(H, W))
        
        student_logits = student_logits.view(B, C, -1)
        teacher_logits = teacher_logits.view(B, C, -1)
        
        student_log_probabilities = F.log_softmax(student_logits, dim=2)
        teacher_probabilities = F.softmax(teacher_logits.detach(), dim=2)
            
        loss = self.criterion(student_log_probabilities, teacher_probabilities)

        loss = (loss / (B * C)) * (self.temperature ** 2)

        return loss