import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any

from utils.helpers import load

class LossComposer(nn.Module):
    """
    Intra: Combines within a task GT and KD loss
    Inter: Combines across tasks into a single global loss
    """
    def __init__(self, config, criterions, tasks):
        super().__init__()
        self.config = config
        self.criterions = criterions
        self.tasks = tasks

        self.inter = load(
            config['training']['weighting']['inter']['name'],
            keys=self.tasks,
            params=config['training']['weighting']['inter']['params']
        )
        
        self.intras = nn.ModuleDict()
        for task in self.tasks:
            self.intras[task] = load(
                config['training']['weighting'][f'intra_{task}']['name'],
                keys=['target', 'distillation'],
                params=config['training']['weighting'][f'intra_{task}']['params']
            )
        
    def _compute_raw_losses(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw_losses = {}
        for task, task_output in outputs.items():
            if task not in self.criterions or task not in targets: continue # skip keys like baseline, focal_length

            criterion = self.criterions[task]
            if 'disparity_distillation' in task:
                intercept_features = outputs['disparity_intercept_features']
                true_targets = targets['disparity']
                raw_loss = criterion(intercept_features, targets[task], true_targets) # custom kd loss needs 3 args
            else:
                raw_loss = criterion(task_output, targets[task]) # standard losses need 2 args

            raw_losses[task] = raw_loss
            
        return raw_losses
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        raw_losses = self._compute_raw_losses(outputs, targets)

        intra_losses = {}
        intra_loss_weights = {}
        for task in self.tasks:
            if f'{task}_distillation' in raw_losses:
                intra_losses[task], intra_loss_weights[task] = self.intras[task].combine(
                    {'target': raw_losses[task], 'distillation': raw_losses[f'{task}_distillation']}
                )
            else:
                intra_losses[task] = raw_losses[task]
                intra_loss_weights[task] = {'target': 1.0, 'distillation': 0.0}

        if len(self.tasks) > 1:
            inter_loss, inter_loss_weights = self.inter.combine(intra_losses)
        else:
            inter_loss = intra_losses[self.tasks[0]]
            inter_loss_weights = {self.tasks[0]: 1.0}
            
        raw_task_losses = {k: float(v.detach().item()) for k, v in raw_losses.items()}
        
        return inter_loss, inter_loss_weights, intra_losses, intra_loss_weights, raw_task_losses
    
    def step_weighting(self, metrics: Optional[Dict[str, Any]] = None):
        if len(self.tasks) > 1 and hasattr(self.inter, 'step'):
            self.inter.step(metrics)
            
        for intra in self.intras.values():
            if hasattr(intra, 'step'):
                intra.step(metrics)
            