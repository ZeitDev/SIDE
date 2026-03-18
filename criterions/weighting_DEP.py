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
                if 'disparity_distillation' in task:
                    intercept_features = outputs['disparity_intercept_features']
                    true_targets = targets['disparity']
                    raw_task_loss = criterion(intercept_features, targets[task], true_targets)
                else:
                    raw_task_loss = criterion(task_output, targets[task])
                raw_task_losses[task] = raw_task_loss.item()
                
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
                logarithmic_variance = self.logarithmic_variances[task]
                precision = torch.exp(-logarithmic_variance)
                total_loss += (0.5 * precision * raw_task_loss) + (0.5 * logarithmic_variance)
                
        return total_loss, raw_task_losses
    
class UnweightedSumLoss(nn.Module):
    def __init__(self, criterions):
        super().__init__()
        self.criterions = criterions

    def forward(self, outputs, targets):
        total_loss = 0.0
        raw_task_losses = {}
        
        for task, task_output in outputs.items():
            if task in self.criterions and task in targets:
                criterion = self.criterions[task]
                
                # Keep your exact same custom logic for the teacher
                if 'disparity_distillation' in task:
                    intercept_features = outputs['disparity_intercept_features']
                    true_targets = targets['disparity']
                    raw_task_loss = criterion(intercept_features, targets[task], true_targets)
                else:
                    raw_task_loss = criterion(task_output, targets[task])
                    
                raw_task_losses[task] = raw_task_loss.item()
                
                # No weights! Just standard addition
                total_loss += raw_task_loss 
                
        return total_loss, raw_task_losses
    
    
class FixedGTKDIntra(nn.Module):
    def __init__(self, tasks: List[str], params: Optional[Dict[str, Any]] = None):
        super().__init__()
        params = params or {}
        alpha_cfg = params.get("alpha_gt", {})
        self.alpha_gt = {task: float(alpha_cfg.get(task, 1.0)) for task in tasks}

    def combine_task(
        self,
        task: str,
        gt_loss: Optional[torch.Tensor],
        kd_loss: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if gt_loss is None and kd_loss is None:
            return None
        if kd_loss is None:
            return gt_loss
        if gt_loss is None:
            return kd_loss

        a = self.alpha_gt.get(task, 1.0)
        return (a * gt_loss) + ((1.0 - a) * kd_loss)


class UniformInter(nn.Module):
    def __init__(self, tasks: List[str], params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.tasks = tasks
        params = params or {}
        init_weights = params.get("weights", {})
        self.weights = {task: float(init_weights.get(task, 1.0)) for task in tasks}

    def combine(self, task_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not task_losses:
            raise ValueError("No task losses available to combine.")

        total = None
        used = {}
        for task, loss in task_losses.items():
            w = self.weights.get(task, 1.0)
            used[task] = w
            total = loss * w if total is None else total + (loss * w)

        assert total is not None
        return total, used

    def update_from_validation(self, val_metrics: Dict[str, float]) -> None:
        return


class UncertaintyInter(nn.Module):
    def __init__(self, tasks: List[str], params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.tasks = tasks
        self.log_vars = nn.ParameterDict({task: nn.Parameter(torch.zeros(1)) for task in tasks})

    def combine(self, task_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not task_losses:
            raise ValueError("No task losses available to combine.")

        total = None
        weights = {}
        for task, loss in task_losses.items():
            s = self.log_vars[task]
            precision = torch.exp(-s)
            weights[task] = float(precision.detach().item())
            weighted = (0.5 * precision * loss) + (0.5 * s)
            total = weighted if total is None else total + weighted

        assert total is not None
        return total, weights

    def update_from_validation(self, val_metrics: Dict[str, float]) -> None:
        return


class ValidationAdaptiveInter(nn.Module):
    def __init__(self, tasks: List[str], params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.tasks = tasks
        params = params or {}

        self.metric_map = params.get("metric_map", {})
        self.mode = params.get("mode", {})  # "min" or "max" per task
        self.ema = float(params.get("ema", 0.9))
        self.step = float(params.get("step", 0.2))
        self.min_w = float(params.get("min_w", 0.1))
        self.max_w = float(params.get("max_w", 10.0))

        init_weights_cfg = params.get("init_weights", {})
        init = torch.tensor([float(init_weights_cfg.get(t, 1.0)) for t in tasks], dtype=torch.float32)
        self.register_buffer("weights", init)
        self.best_scores: Dict[str, Optional[float]] = {t: None for t in tasks}

    def _norm_weights(self) -> torch.Tensor:
        w = torch.clamp(self.weights, min=self.min_w, max=self.max_w)
        s = torch.sum(w)
        if float(s.item()) == 0.0:
            return torch.ones_like(w) / len(w)
        return w / s

    def combine(self, task_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not task_losses:
            raise ValueError("No task losses available to combine.")

        norm_w = self._norm_weights()
        total = None
        used = {}
        for i, task in enumerate(self.tasks):
            if task not in task_losses:
                continue
            w = norm_w[i]
            used[task] = float(w.detach().item())
            weighted = task_losses[task] * w
            total = weighted if total is None else total + weighted

        if total is None:
            raise ValueError("No active task had a loss to combine.")
        return total, used

    def update_from_validation(self, val_metrics: Dict[str, float]) -> None:
        new_w = self.weights.clone()
        for i, task in enumerate(self.tasks):
            metric_key = self.metric_map.get(task)
            if metric_key is None or metric_key not in val_metrics:
                continue

            curr = float(val_metrics[metric_key])
            best = self.best_scores[task]
            mode = self.mode.get(task, "min")

            if best is None:
                self.best_scores[task] = curr
                continue

            improved = curr < best if mode == "min" else curr > best
            if improved:
                self.best_scores[task] = curr
                target = torch.clamp(new_w[i] * (1.0 - self.step), min=self.min_w, max=self.max_w)
            else:
                target = torch.clamp(new_w[i] * (1.0 + self.step), min=self.min_w, max=self.max_w)

            new_w[i] = (self.ema * new_w[i]) + ((1.0 - self.ema) * target)

        self.weights.copy_(torch.clamp(new_w, min=self.min_w, max=self.max_w))


class LossComposer(nn.Module):
    def __init__(
        self,
        criterions: Dict[str, nn.Module],
        tasks: List[str],
        weighting_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.criterions = criterions
        self.tasks = tasks
        weighting_config = weighting_config or {}

        intra_cfg = weighting_config.get("intra_task", {})
        inter_cfg = weighting_config.get("inter_task", {})

        intra_name = intra_cfg.get("name", "fixed_gt_kd")
        intra_params = intra_cfg.get("params", {})
        if intra_name == "fixed_gt_kd":
            self.intra = FixedGTKDIntra(tasks=tasks, params=intra_params)
        else:
            raise ValueError(f"Unsupported intra_task strategy: {intra_name}")

        inter_name = inter_cfg.get("name", "uniform")
        inter_params = inter_cfg.get("params", {})
        if inter_name == "uniform":
            self.inter = UniformInter(tasks=tasks, params=inter_params)
        elif inter_name == "uncertainty":
            self.inter = UncertaintyInter(tasks=tasks, params=inter_params)
        elif inter_name == "validation_adaptive":
            self.inter = ValidationAdaptiveInter(tasks=tasks, params=inter_params)
        else:
            raise ValueError(f"Unsupported inter_task strategy: {inter_name}")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        raw_losses = _compute_raw_losses(self.criterions, outputs, targets)

        task_losses: Dict[str, torch.Tensor] = {}
        for task in self.tasks:
            gt = raw_losses.get(task)
            kd = raw_losses.get(f"{task}_teacher")
            task_loss = self.intra.combine_task(task=task, gt_loss=gt, kd_loss=kd)
            if task_loss is not None:
                task_losses[task] = task_loss

        total_loss, task_weights = self.inter.combine(task_losses)
        raw_task_losses = {k: float(v.detach().item()) for k, v in raw_losses.items()}
        return total_loss, raw_task_losses, task_weights

    def extra_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def update_from_validation(self, val_metrics: Dict[str, float]) -> None:
        if hasattr(self.inter, "update_from_validation"):
            self.inter.update_from_validation(val_metrics)


def build_loss_composer(
    criterions: Dict[str, nn.Module],
    tasks: List[str],
    weighting_config: Optional[Dict[str, Any]] = None
) -> LossComposer:
    return LossComposer(criterions=criterions, tasks=tasks, weighting_config=weighting_config)