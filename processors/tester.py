import logging
from tqdm import tqdm
from typing import cast, Any, Dict

import torch
import mlflow
from torch.utils.data import DataLoader 

from utils import helpers
from utils.helpers import load
from processors.base import BaseProcessor

from utils.logger import CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))

class Tester(BaseProcessor):
    def __init__(self, config: Dict[str, Any], run_id: str):
        super().__init__(config)
        self.run_id = run_id
        self._load_data()
        self._load_models()
        self._init_metrics()

    def _load_data(self) -> None:
        logger.subheader('Load Data')

        data_config = self.config['data']
        dataset_class = load(data_config['dataset'])
        
        test_transforms = helpers.build_transforms(data_config['transforms']['test'])
        
        dataset_test = dataset_class(
            mode='test',
            transforms=test_transforms,
            tasks=self.config['training']['tasks']
        )
        self.dataloader_test = DataLoader(
            dataset_test,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'])
            
        logger.info(f'Loaded test dataset: {data_config["dataset"]} with {len(dataset_test)} samples.')
        
        if self.config['training']['tasks']['segmentation']['enabled'] and hasattr(dataset_test, 'class_mappings'):
            self.segmentation_class_mappings = dataset_test.class_mappings
            num_classes = len(self.segmentation_class_mappings) # type: ignore
            self.config['training']['tasks']['segmentation']['decoder']['params']['num_classes'] = num_classes
            self.config['training']['tasks']['segmentation']['knowledge_distillation']['decoder']['params']['num_classes'] = num_classes
            
            logger.info(f'Class Mappings for Segmentation Task: {self.segmentation_class_mappings}')
            
    def _load_models(self) -> None:
        logger.subheader('Loading Models')
        self.models = {}
        
        for task_name, task_config in self.config['training']['tasks'].items():
            if task_config['enabled']:
                model_path = f'runs:/{self.run_id}/model_{task_name}'
                self.models[task_name] = mlflow.pytorch.load_model(model_path, map_location=self.device) # type: ignore
                self.models[task_name].eval()
                logger.info(f'Loaded model for task {task_name} from {model_path}')
 
    def test(self) -> Dict[str, float]:
        logger.header('Starting Testing')
        
        for task_metrics in self.metrics.values():
            for metric in task_metrics.values():
                metric.reset()
        
        batch_tqdm = tqdm(self.dataloader_test, desc='Testing', position=1, leave=False)
        with torch.no_grad():
            for images, targets in batch_tqdm:
                images = images.to(self.device)
                targets = {key: value.to(self.device) for key, value in targets.items()}
                
                outputs = {}
                for task_name, model in self.models.items():
                    outputs[task_name] = model(images)
                    
                    # * TEMP for debugging
                    # seg_targets = targets.get('segmentation', None)
                    # seg_outputs = torch.argmax(outputs['segmentation'], dim=1, keepdim=True)
                    # * TEMP for debugging
                    
                    if task_name in self.metrics:
                        for metric in self.metrics[task_name].values():
                            metric.update(outputs[task_name], targets[task_name])
                            
                self._log_visuals(epoch='Test', images=images, targets=targets, outputs=outputs)

        logger.subheader('Test Results')
        test_metrics = self._compute_metrics()
        for metric_key, metric_value in test_metrics.items():
            logger.info(f'{metric_key}: {metric_value:.4f}')
            
        return test_metrics    