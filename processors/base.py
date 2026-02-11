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
from metrics.disparity import EPE, Bad3, MAE

from utils.logger import CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))

class BaseProcessor:
    def __init__(self, config: Dict[str, Any]):
        logger.header(f'Initializing {self.__class__.__name__}')
        self.config = config
        self.tasks = []
        self.n_classes = {}
        self.metrics: Dict[str, Any] = {}
        self.segmentation_class_mappings: Optional[Dict[int, str]] = None
        self.n_logged_images = 0
        self._setup()

    def _setup(self) -> None:
        logger.subheader('Setup')
        
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
            
            self.metrics['segmentation']['IoU_score'] = IoU(n_classes=self.n_classes['segmentation'], device=self.device)
            self.metrics['segmentation']['DICE_score'] = Dice(n_classes=self.n_classes['segmentation'], device=self.device)
            logger.info(f'Initialized IoU and DICE metrics for segmentation with {self.n_classes['segmentation']} classes.')
            
        if tasks_config['disparity']['enabled']:
            max_disparity = self.config['data']['max_disparity']
            self.metrics['disparity'] = {}
            self.metrics['disparity']['EPE_pixel'] = EPE(max_disparity=max_disparity, device=self.device)
            self.metrics['disparity']['Bad3_rate'] = Bad3(max_disparity=max_disparity, device=self.device)
            self.metrics['disparity']['MAE_mm'] = MAE(max_disparity=max_disparity, device=self.device)
            logger.info('Initialized EPE, Bad3, and MAE metrics for disparity.')

    def _compute_metrics(self, mode: str = 'validation') -> Dict[str, float]:
        computed_metrics = {}
        for task, task_metrics in self.metrics.items():
            for metric_name, metric in task_metrics.items():
                metric_results = metric.compute()
                
                for key, value in metric_results.items():
                    if isinstance(key, int) and self.segmentation_class_mappings:
                        class_name = self.segmentation_class_mappings[key].replace(' ', '_').lower()
                        computed_metrics[f'performance/{mode}/{task}/{metric_name}/{key}::{class_name}'] = value
                    elif metric_name == key:
                        computed_metrics[f'performance/{mode}/{task}/{metric_name}'] = value
                    else:
                        computed_metrics[f'performance/{mode}/{task}/{metric_name}/{key}'] = value
        return computed_metrics
    
    def _log_visuals(self, epoch: Any, images: torch.Tensor, targets: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> None:
        log_n_images = self.config['logging']['n_validation_images']
        max_epochs = self.config['training']['epochs']
        max_indices = self.config['logging']['n_validation_images']
        epoch_padding = len(str(max_epochs))
        index_padding = len(str(max_indices))
        
        if log_n_images > 0:
            for i in range(len(images)):
                if self.n_logged_images >= log_n_images: break
                self.n_logged_images += 1
                
                sample_targets = {k: v[i].cpu().detach() for k, v in targets.items()}
                sample_outputs = {k: v[i].cpu().detach() for k, v in outputs.items()}
                sample_image = images[i].cpu().detach()
                
                figure = visualization.get_multitask_visuals(
                    image=sample_image,
                    targets=sample_targets,
                    outputs=sample_outputs,
                    n_classes=self.n_classes,
                    epoch=epoch,
                    index=i,
                    max_disparity=self.config['data']['max_disparity']
                )
                
                if not self.config['logging']['notebook_mode']: 
                    mlflow.log_figure(figure, artifact_file=f'validation_overlays/epoch_{epoch:0{epoch_padding}}/index_{i:0{index_padding}}.png')
                else: 
                    plt.show(figure)
                plt.close(figure)