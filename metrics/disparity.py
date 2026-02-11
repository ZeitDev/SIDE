import torch
from typing import Dict

class DisparityMetric:
    def __init__(self, max_disparity: float = 512, device: torch.device = torch.device('cpu')):
        self.device = device
        self.max_disparity = max_disparity
        
        self.total_error = torch.tensor(0.0, device=self.device)
        self.total_valid_pixels = torch.tensor(0.0, device=self.device)

    def update(self, output: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor, focal_length: torch.Tensor) -> None:
        """
        output: Predicted disparity in [px].
        target: Ground truth disparity in [px].
        baseline: Stereo baseline. The unit used here millimeters defines the unit of the calculated depth.
        focal_length: Focal length in [px].
        """
        with torch.no_grad():
            output = output * self.max_disparity
            target = target * self.max_disparity
            
            valid_mask = target != 0

            batch_error_sum = self.get_batch_error_sum(output, target, valid_mask, baseline, focal_length)

            self.total_error += batch_error_sum
            self.total_valid_pixels += valid_mask.sum()

    def compute(self) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        self.total_error.fill_(0)
        self.total_valid_pixels.fill_(0)

    def get_batch_error_sum(self, output: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, baseline: torch.Tensor, focal_length: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EPE(DisparityMetric):
    def get_batch_error_sum(self, output: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, baseline: torch.Tensor, focal_length: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(output - target)
        return diff[valid_mask].sum()

    def compute(self) -> Dict[str, float]:
        return {'EPE_pixel': (self.total_error / self.total_valid_pixels).item()}


class Bad3(DisparityMetric):
    def get_batch_error_sum(self, output: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, baseline: torch.Tensor, focal_length: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(output - target)
        bad_pixels = (diff > 3) & valid_mask
        return bad_pixels.float().sum()

    def compute(self) -> Dict[str, float]:
        return {'Bad3_rate': (self.total_error / self.total_valid_pixels).item()}


class MAE(DisparityMetric):
    def get_batch_error_sum(self, output: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, baseline: torch.Tensor, focal_length: torch.Tensor) -> torch.Tensor:
        depth_output = (focal_length * baseline) / output
        depth_target = (focal_length * baseline) / target

        abs_diff = torch.abs(depth_output - depth_target)
        
        return abs_diff[valid_mask].sum() 

    def compute(self) -> Dict[str, float]:
        return {'MAE_mm': (self.total_error / self.total_valid_pixels).item()}