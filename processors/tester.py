import logging
from tqdm import tqdm
from typing import cast, Any, Dict

import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader 

from utils import helpers
from utils.helpers import load
from processors.base import BaseProcessor
from data.transforms import build_transforms

from utils.logger import CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))

class Tester(BaseProcessor):
    def __init__(self, config: Dict[str, Any], run_id: str):
        super().__init__(config)
        self.run_id = run_id
        self._load_data()
        self._init_metrics()

    def _load_data(self) -> None:
        logger.subheader('Load Data')

        data_config = self.config['data']
        dataset_class = load(data_config['dataset'])
        
        self.tasks = [task for task, task_config in self.config['training']['tasks'].items() if task_config['enabled']]
        
        test_transforms = build_transforms(self.config, mode='test')
        
        dataset_test = dataset_class(
            mode='test',
            config=self.config,
            transforms=test_transforms,
        )
        self.dataloader_test = DataLoader(
            dataset_test,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            persistent_workers=False
        )
        helpers.check_dataleakage('test', dataset_test)
        logger.info(f'Loaded test dataset: {data_config["dataset"]} with {len(dataset_test)} samples.')
        
        if 'segmentation' in self.tasks:
            self.segmentation_class_mappings = dataset_test.segmentation_class_mappings
            logger.info(f'Class Mappings for Segmentation Task: {self.segmentation_class_mappings}')
            
    def _load_models(self, task_mode: str) -> None:
        logger.subheader('Loading Model')
        model_path = f'runs:/{self.run_id}/best_model_{task_mode}'
        self.model = mlflow.pytorch.load_model(model_path, map_location=self.device)
        self.model.eval()
        logger.info(f'Loaded best model from {model_path}')
 
    def test(self) -> Dict[str, float]:
        logger.header('Starting Testing')
        
        if 'segmentation' in self.tasks: task_modes = ['segmentation']
        if 'disparity' in self.tasks: task_modes = ['disparity']
        if 'segmentation' in self.tasks and 'disparity' in self.tasks: task_modes = ['segmentation', 'disparity', 'combined']
        
        test_metrics = {}
        for task_mode in task_modes:
            self._load_models(task_mode=task_mode)
            
            for task_metrics in self.metrics.values():
                for metric in task_metrics.values():
                    metric.reset()
            
            batch_tqdm = tqdm(self.dataloader_test, desc='Testing', position=1, leave=False)
            with torch.no_grad():
                for data in batch_tqdm:
                    left_images = data['image'].to(self.device)
                    right_images = data['right_image'].to(self.device) if 'right_image' in data else None
                    targets = {task: data[task].to(self.device) for task in self.tasks}
                    
                    outputs = self.model(left_images, right_images)
                    
                    if 'segmentation' in outputs:
                        self._log_visuals(epoch='Test', images=left_images, targets=targets, outputs=outputs)
                        for metric in self.metrics['segmentation'].values():
                            metric.update(outputs['segmentation'], targets['segmentation'])
                    if 'disparity' in outputs:
                        baseline, focal_length = data['baseline'].to(self.device), data['focal_length'].to(self.device)
                        for metric in self.metrics['disparity'].values():
                            metric.update(outputs['disparity'], targets['disparity'], baseline, focal_length)

            logger.subheader('Test Results')
            test_metrics = self._compute_metrics(mode='testing')
            for key, value in test_metrics.items(): mlflow.log_metric(f'best_{task_mode}/{key}', value)
            
            for metric_key, metric_value in test_metrics.items():
                logger.info(f'{metric_key}: {metric_value:.4f}')
            
            del self.model
            
