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
    
    def step(self, metrics: Optional[Dict[str, Any]] = None):
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