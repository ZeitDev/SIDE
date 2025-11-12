import os
import random
import logging
import numpy as np
from typing import cast, Any, Dict, Optional

import torch
import mlflow
import matplotlib.pyplot as plt

from utils import visualization
from metrics.segmentation import IoU, Dice

from utils.logger import CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))

class BaseProcessor:
    def __init__(self, config: Dict[str, Any]):
        logger.header(f'Initializing {self.__class__.__name__}')
        self.config = config
        self.metrics: Dict[str, Any] = {}
        self.segmentation_class_mappings: Optional[Dict[int, str]] = None
        self.n_logged_images = 0
        self._setup()

    def _setup(self) -> None:
        logger.subheader('Setup')
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        logger.info(f'Restricting to GPU {os.environ.get("CUDA_VISIBLE_DEVICES")}')
        
        cpu_cores = list(range(24))
        os.sched_setaffinity(os.getpid(), cpu_cores)
        logger.info(f'Set CPU affinity to cores: {cpu_cores}')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        torch.cuda.empty_cache()
        seed = 42
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        logger.info(f'Set reproducibility seed to {seed}')
   
    def _init_metrics(self) -> None:
        logger.subheader('Initializing Metrics')
        self.metrics = {}
        tasks_config = self.config['training']['tasks']
        
        if tasks_config['segmentation']['enabled']:
            self.metrics['segmentation'] = {}
            
            num_classes = tasks_config['segmentation']['decoder']['params']['num_classes']
            self.metrics['segmentation']['IoU'] = IoU(num_classes=num_classes, device=self.device)
            self.metrics['segmentation']['DICE'] = Dice(num_classes=num_classes, device=self.device)
            logger.info(f'Initialized IoU and DICE metrics for segmentation with {num_classes} classes.')
            
        if tasks_config['disparity']['enabled']:
            pass # TODO: implement disparity metrics

    def _compute_metrics(self) -> Dict[str, float]:
        computed_metrics = {}
        for task, task_metrics in self.metrics.items():
            for metric_name, metric in task_metrics.items():
                metric_results = metric.compute()
                
                for key, value in metric_results.items():
                    if isinstance(key, int) and self.segmentation_class_mappings:
                        class_name = self.segmentation_class_mappings[key].replace(' ', '_').lower()
                        computed_metrics[f'performance/{task}/{metric_name}/{key}::{class_name}'] = value
                    else:
                        computed_metrics[f'performance/{task}/{metric_name}/{key}'] = value
        return computed_metrics
    
    def _log_visuals(self, epoch: Any, images: torch.Tensor, targets: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> None:
        log_n_images = self.config['logging']['n_validation_images']
        if log_n_images > 0:
            n_logged_batch = 0
            for i in range(log_n_images):
                if n_logged_batch < len(images) and self.n_logged_images < log_n_images:
                    self.n_logged_images += 1
                    n_logged_batch += 1
                    
                    if self.config['training']['tasks']['segmentation']['enabled']:
                        num_classes = len(self.segmentation_class_mappings) if self.segmentation_class_mappings else 0
                        figure = visualization.get_image_target_output_overlay(image=images[i].cpu().detach(), target=targets['segmentation'][i].cpu().detach(), output=outputs['segmentation'][i].cpu().detach(), num_classes=num_classes, epoch=epoch, index=i)
                        
                        if not self.config['logging']['notebook_mode']: mlflow.log_figure(figure, artifact_file=f'validation_overlays/epoch_{epoch}/index_{i}.png')
                        else: plt.show(figure)
                        plt.close(figure)