import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseKLDivLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature        
        self.criterion = nn.KLDivLoss(reduction='sum', log_target=False)

    def forward(self, student_logits, teacher_logits):
        b, c, h, w = student_logits.shape
        
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature
        
        student_logits = student_logits.view(b, c, -1)
        teacher_logits = teacher_logits.view(b, c, -1)
        
        student_log_probabilities = F.log_softmax(student_logits, dim=2)
        with torch.no_grad(): teacher_probabilities = F.softmax(teacher_logits, dim=2)
            
        loss = self.criterion(student_log_probabilities, teacher_probabilities)

        loss = (loss / (b * c)) * (self.temperature ** 2)

        return loss