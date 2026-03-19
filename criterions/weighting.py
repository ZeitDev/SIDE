import math
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

class BaseWeighting(nn.Module):
    def __init__(self, keys: List[str], params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.keys = keys
        self.params = params
        
    def combine(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Combines N losses into a single scalar loss."""
        raise NotImplementedError
    
    def step(self, metrics: Dict[str, Any]):
        """Hook to be called at the end of an epoch or validation pass."""
        pass
    
class Unweighted(BaseWeighting):
    def combine(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        combined_loss = sum(losses.values())
        weights = {key: 1.0 for key in losses.keys()}
        
        return combined_loss, weights
    
class Uncertainty(BaseWeighting):
    """
    # Uncertainty Formula from paper:
    # L_total = sum( (1 / 2*sigma_i^2) * L_i + log(sigma_i) )
    # L_i: Raw loss for task i
    # sigma_i: task uncertainty / noise 
    # logarithmic_variance = s [paper] = log(sigma_i^2): trainable, more numerically stable, avoids division by zero than only using sigma_i^2
    # 0.5 * logarithmic_variance = log(sigma_i) [paper]: power rule of logarithms
    # precision = 1/(sigma_i^2) [paper] = exp(-s): confidence in task i, exp conversion to ensure positivity and avoid division by zero
    # ! precision==confidence: automatic weight in dashboard
    # ! Important: Paper uses 1 * precision for classification tasks, and 0.5 * precision for regression tasks. Here we use 0.5 for all tasks for simplicity and because precision is trainable anyways.
    # ! Important: Weight Decay should not be applied as optimizer wants to force logarithmic_variance to zero otherwise.
    """
    def __init__(self, keys: List[str], params: Optional[Dict[str, Any]] = None):
        super().__init__(keys, params)
        self.logarithmic_variances = nn.ParameterDict({key: nn.Parameter(torch.zeros(1)) for key in keys})

    def combine(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        combined_loss = 0
        weights = {}
        
        for key, loss in losses.items():
            logarithmic_variance = self.logarithmic_variances[key]
            precision = torch.exp(-logarithmic_variance)
            combined_loss += (0.5 * precision * loss) + (0.5 * logarithmic_variance)
            weights[key] = precision.item()
        
        return combined_loss, weights
    
class CompetenceDecay(BaseWeighting):
    def __init__(self, keys: List[str], params: Optional[Dict[str, Any]] = None):
        super().__init__(keys, params)
        
        self.alpha = self.params['alpha'] # How fast the competence decay reacts to changes in student performance, higher alpha means faster reaction
        self.teacher_performance = self.params['teacher_performance']
        self.inverse = self.params['inverse']
        self.metric_key = self.params['metric_key']
        self.metric_is_score = self.params['metric_is_score']
        
        self.weight_distillation = 1.0 if not self.inverse else 0.0
        self.weight_target = 0.0 if not self.inverse else 1.0
        
        self.eps = 1e-8
        self.student_error_ema = None
        
    def combine(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        combined_loss = (self.weight_target * losses['target']) + (self.weight_distillation * losses['distillation'])
        weights = {'target': self.weight_target, 'distillation': self.weight_distillation}
        
        return combined_loss, weights
    
    def step(self, metrics: Dict[str, Any]):
        student_performance = metrics[self.metric_key]
        teacher_performance = self.teacher_performance
        
        if self.metric_is_score:
            student_error = 1 - student_performance
            teacher_error = 1 - teacher_performance
        else:
            student_error = student_performance
            teacher_error = teacher_performance
            
        if self.student_error_ema is None: self.student_error_ema = student_error
        else: self.student_error_ema = (self.alpha * student_error) + ((1.0 - self.alpha) * self.student_error_ema)
        self.student_error_ema = max(self.student_error_ema, self.eps)    
        
        gap = (self.student_error_ema - teacher_error) / self.student_error_ema
        self.weight_distillation = max(0.0, gap) if not self.inverse else min(1.0, 1 - gap)
        self.weight_target = 1.0 - self.weight_distillation      
        
        
class DynamicTaskPriority(BaseWeighting):
    def __init__(self, keys: List[str], params: Optional[Dict[str, Any]] = None):
        super().__init__(keys, params)
        
        self.alpha = self.params['alpha'] # Moving average reacts faster to changes with higher alpha
        self.gamma = self.params['gamma'] # Attenuation factor, weight^gamma, penalizes easy task more
        self.metric_keys = {key: self.params[f'metric_key_{key}'] for key in keys}
        self.metric_is_score = {key: self.params[f'metric_is_score_{key}'] for key in keys}
        self.eps = 1e-8
        self.kappa_ema = {key: 0.5 for key in keys}
    
    def step(self, metrics: Dict[str, Any]):
        for key in self.keys:
            if self.metric_is_score[key]: current_kappa = metrics[self.metric_keys[key]]
            else: current_kappa = 1 - metrics[self.metric_keys[key]]
            
            current_kappa = max(0.0, min(1.0, current_kappa))
            
            self.kappa_ema[key] = (self.alpha * current_kappa) + ((1.0 - self.alpha) * self.kappa_ema[key])
        
    def combine(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        combined_loss = 0
        raw_weights = {}
        weights = {}
        
        for key in self.keys:
            kappa_bar = max(self.kappa_ema[key], self.eps)
            kappa_bar = min(kappa_bar, 1.0 - self.eps)
            raw_weights[key] = -1.0 * (1.0 - kappa_bar) ** self.gamma * math.log(kappa_bar)
            
        sum_weights = sum(raw_weights.values())
        for key in self.keys:
            if sum_weights < self.eps: norm_weight = 1.0
            else: norm_weight = (raw_weights[key] / sum_weights) * len(self.keys)
            
            weights[key] = norm_weight
            combined_loss += norm_weight * losses[key]
        
        return combined_loss, weights