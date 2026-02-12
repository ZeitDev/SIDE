import torch.nn as nn
import torch.nn.functional as F

class KLDivLoss(nn.Module):
    def __init__(self, temperature=4.0, reduction='batchmean'):
        super().__init__()
        self.temperature = temperature        
        self.criterion = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, student_logits, teacher_logits):
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature
        
        student_log_probabilities = F.log_softmax(student_logits, dim=1)
        teacher_log_probabilities = F.log_softmax(teacher_logits, dim=1)
            
        loss = self.criterion(student_log_probabilities, teacher_log_probabilities)

        h, w = student_logits.shape[-2], student_logits.shape[-1]
        loss_mean = loss / (h * w)

        return loss_mean