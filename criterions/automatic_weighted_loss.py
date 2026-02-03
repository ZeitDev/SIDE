import torch
import torch.nn as nn

'''
Uses the homoscedastic uncertainty weighting method by Kendall (2018)
See page 5, bottom left formula
'''

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, criterions, freeze=False):
        super().__init__()
        self.criterions = criterions
        self.logarithmic_variances = nn.ParameterDict()
        
        for task in criterions.keys():
            param = nn.Parameter(torch.zeros(1))
            if freeze: param.requires_grad = False
            self.logarithmic_variances[task] = param

    def forward(self, outputs, targets):
        total_loss = 0.0
        raw_task_losses = {}
        
        for task, task_output in outputs.items():
            if task in self.criterions and task in targets:
                criterion = self.criterions[task]
                raw_task_loss = criterion(task_output, targets[task])
                raw_task_losses[task] = raw_task_loss.item()                
                
                # Uncertainty Formula from paper:
                # L_total = sum( (1 / 2*sigma_i^2) * L_i + log(sigma_i) )
                # L_i: Raw loss for task i
                # sigma_i: task uncertainty / noise 
                # logarithmic_variance = s [paper] = log(sigma_i^2): trainable, more numerically stable, avoids division by zero than only using sigma_i^2
                # 0.5 * logarithmic_variance = log(sigma_i) [paper]: power rule of logarithms
                # precision = 1/(sigma_i^2) [paper] = exp(-s): confidence in task i, exp conversion to ensure positivity and avoid division by zero
                # ! Important: Paper uses 1 * precision for classification tasks, and 0.5 * precision for regression tasks. Here we use 0.5 for all tasks for simplicity and because precision is trainable anyways.
                # ! Important: Weight Decay should not be applied as optimizer wants to force logarithmic_variance to zero otherwise.
                logarithmic_variance = self.logarithmic_variances[task]
                precision = torch.exp(-logarithmic_variance)
                total_loss += (0.5 * precision * raw_task_loss) + (0.5 * logarithmic_variance)
                
        return total_loss, raw_task_losses