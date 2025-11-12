import os
import logging
import numpy as np
from tqdm import tqdm
from typing import cast, Any, List, Dict, Optional

import torch
import mlflow
from torch.utils.data import DataLoader 

from utils import helpers
from utils.helpers import load
from utils.models import Decombiner
from processors.base import BaseProcessor

from utils.logger import CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))

class Trainer(BaseProcessor):
    def __init__(self, config: Dict[str, Any], train_subsets: List[str], val_subsets: Optional[List[str]] = None):
        super().__init__(config)
        self.train_subsets = train_subsets
        self.val_subsets = val_subsets
        self._load_data()
        self._load_model()
        self._load_teachers()
        self._load_components()
        self._init_metrics()
        
    def _load_data(self) -> None:
        logger.subheader('Load Data')

        data_config = self.config['data']
        dataset_class = load(data_config['dataset'])
        
        train_transforms = helpers.build_transforms(data_config['transforms']['train'])
        val_transforms = helpers.build_transforms(data_config['transforms']['test'])
        
        dataset_train = dataset_class(
            mode='train',
            transforms=train_transforms,
            tasks=self.config['training']['tasks'],
            subset_names=self.train_subsets
        )
        self.dataloader_train = DataLoader(
            dataset_train,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        dataset_val = dataset_class(
            mode='train',
            transforms=val_transforms,
            tasks=self.config['training']['tasks'],
            subset_names=self.val_subsets
        ) if self.val_subsets else None
        if dataset_val:
            self.dataloader_val = DataLoader(
                dataset_val,
                batch_size=data_config['batch_size'],
                shuffle=False,
                num_workers=data_config['num_workers'],
                pin_memory=data_config['pin_memory']
            )
            
        input_example_image, _ = dataset_train[0] 
        self.input_example = input_example_image.unsqueeze(0)
            
        logger.info(f'Loaded datasets: {data_config["dataset"]} with batch size {data_config["batch_size"]}, num_workers {data_config["num_workers"]}, pin_memory {data_config["pin_memory"]}')
        dataset_val_length = len(dataset_val) if dataset_val else 0
        logger.info(f'Num of Samples - Training: {len(dataset_train)}, Validation: {dataset_val_length}')
        
        if self.config['training']['tasks']['segmentation']['enabled']:
            self.segmentation_class_mappings = dataset_train.class_mappings
            num_classes = len(self.segmentation_class_mappings) # type: ignore
            self.config['training']['tasks']['segmentation']['decoder']['params']['num_classes'] = num_classes
            self.config['training']['tasks']['segmentation']['knowledge_distillation']['decoder']['params']['num_classes'] = num_classes
        
            logger.info(f'Class Mappings for Segmentation Task: {self.segmentation_class_mappings}')
        
    def _load_model(self) -> None:
        logger.subheader('Loading Model')
        
        encoder_config = self.config['training']['encoder']
        EncoderClass = load(encoder_config['name'])
        encoder = EncoderClass(**encoder_config['params'])
        logger.info(f'Loaded encoder: {encoder_config["name"]} with params {encoder_config["params"]}')
        
        decoders = {}
        tasks_config = self.config['training']['tasks']
        for task, task_config in tasks_config.items():
            if task_config['enabled']:
                decoder_config = task_config['decoder']
                DecoderClass = load(decoder_config['name'])
                decoders[task] = DecoderClass(**decoder_config['params'])
                logger.info(f'Loaded decoder for task {task}: {decoder_config["name"]} with params {decoder_config["params"]}')
            
        self.model = load(
            'utils.models.Combiner', 
            encoder=encoder, 
            decoders=decoders
        ).to(self.device)
        
        if self.config['training']['checkpoint']:
            logger.info(f'Resuming checkpoint {self.config["training"]["checkpoint"]}')
            state_dict = torch.load(self.config['training']['checkpoint'], map_location=self.device)
            self.model.load_state_dict(state_dict['model_state_dict'])
        elif self.config['training']['finetune']:
            logger.info(f'Finetuning {self.config["training"]["finetune"]} with frozen encoder')
            state_dict = torch.load(self.config['training']['finetune'], map_location=self.device)
            self.model.load_state_dict(state_dict['model_state_dict'])
            for param in self.model.encoder.parameters():
                param.requires_grad = False
                    
    def _load_teachers(self) -> None:
        self.kd_models = {}
        self.kd_criterions = {}
        for task, task_config in self.config['training']['tasks'].items():
            kd_config = task_config['knowledge_distillation']
            if kd_config['enabled']:
                logger.subheader(f'Loading Teachers for {task}')
                self.kd_models[task] = []
            
                for state_path in kd_config['states']:
                    teacher_encoder = load(kd_config['encoder']['name'], **kd_config['encoder']['params'])
                    logger.info(f'Loaded teacher encoder: {kd_config["encoder"]["name"]} with params {kd_config["encoder"]["params"]}')
                    
                    teacher_decoder = load(kd_config['decoder']['name'], **kd_config['decoder']['params'])
                    logger.info(f'Loaded teacher decoder: {kd_config["decoder"]["name"]} with params {kd_config["decoder"]["params"]}')
                    
                    teacher_model = load(
                        'utils.models.Combiner',
                        encoder=teacher_encoder,
                        decoders={task: teacher_decoder}
                    ).to(self.device)
                    
                    state_dict = torch.load(state_path, map_location=self.device)
                    teacher_model.load_state_dict(state_dict['model_state_dict'])
                    teacher_model.eval()
                    self.kd_models[task].append(teacher_model)
                    logger.info(f'Loaded teacher {state_path}')
                    
                self.kd_criterions[task] = load(
                    kd_config['criterion']['name'],
                    **kd_config['criterion']['params'])
                logger.info(f'Loaded KD criterion for task {task}: {kd_config["criterion"]["name"]} with params {kd_config["criterion"]["params"]}')
        
    def _load_components(self) -> None:
        logger.subheader('Loading Components')

        self.criterions = {}
        tasks_config = self.config['training']['tasks']
        for task, task_config in tasks_config.items():
            if task_config['enabled']:
                criterion_config = task_config['criterion']
                self.criterions[task] = load(
                    criterion_config['name'], 
                    **criterion_config['params'])
                logger.info(f'Criterion for task {task}: {criterion_config["name"]} with params {criterion_config["params"]}')


        optimizer_config = self.config['training']['optimizer']
        optimizer_class = load(optimizer_config['name'])
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **optimizer_config['params'])
        logger.info(f'Optimizer: {optimizer_config["name"]} with params {optimizer_config["params"]}')
        
        scheduler_config = self.config['training']['scheduler']
        scheduler_class = load(scheduler_config['name'])
        self.scheduler = scheduler_class(
            self.optimizer,
            **scheduler_config['params'])
        logger.info(f'Scheduler: {scheduler_config["name"]} with params {scheduler_config["params"]}')
    
    def _save_model(self):
        logger.info(f'Saving best model to mlflow')        
        
        state_dict = torch.load(os.path.join('cache', 'model_state.pth'))
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.model.to('cpu')
        for task_name, task_config in self.config['training']['tasks'].items():
            if task_config['enabled']:
                task_model = Decombiner(self.model, task_name)
                artifact_path = f'model_{task_name}'
                mlflow.pytorch.log_model( # type: ignore
                    pytorch_model=task_model,
                    name=artifact_path,
                    input_example=self.input_example.cpu().numpy().astype(np.float32)
                )
        self.model.to(self.device)

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        
        batch_tqdm = tqdm(self.dataloader_train, desc='Training', position=1, leave=False)
        for images, targets in batch_tqdm:
            images = images.to(self.device)
            targets = {task: targets_task.to(self.device) for task, targets_task in targets.items()}


            with torch.set_grad_enabled(True):
                total_loss_batch = torch.tensor(0.0, device=self.device)

                outputs = self.model(images)
                
                # * TEMP for debugging
                # seg_targets = targets.get('segmentation', None)
                # seg_outputs = torch.argmax(outputs['segmentation'], dim=1, keepdim=True)
                # * TEMP for debugging
                
                for task, outputs_task in outputs.items():
                    loss = self.criterions[task](outputs_task, targets[task])
                    weight = self.config['training']['tasks'][task]['criterion']['weight']
                    total_loss_batch += (weight * loss)
                    
                if self.kd_models:
                    for task, kd_teachers in self.kd_models.items():
                        student_output = outputs[task]
                        with torch.no_grad():
                            teacher_outputs = [teacher(images)[task] for teacher in kd_teachers]
                            mean_teacher_output = torch.mean(torch.stack(teacher_outputs), dim=0)
                        
                        kd_loss = self.kd_criterions[task](student_output, mean_teacher_output)
                        kd_weight = self.config['training']['tasks'][task]['knowledge_distillation']['criterion']['weight']
                        total_loss_batch += kd_weight * kd_loss
                        
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                batch_tqdm.set_postfix({'batch_loss': f'{total_loss_batch.item():.4f}'})
        
        return {'optimization/loss_training': total_loss / len(self.dataloader_train)}
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        
        for task_metrics in self.metrics.values():
            for metric in task_metrics.values():
                metric.reset()
        
        batch_tqdm = tqdm(self.dataloader_val, desc='Validation', position=1, leave=False)
        for images, targets in batch_tqdm:
            images = images.to(self.device)
            targets = {task: targets_task.to(self.device) for task, targets_task in targets.items()}
            
            with torch.no_grad():
                total_loss_batch = torch.tensor(0.0, device=self.device)
                
                outputs = self.model(images)
                
                # * TEMP for debugging
                # seg_targets = targets.get('segmentation', None)
                # seg_outputs = torch.argmax(outputs['segmentation'], dim=1, keepdim=True)
                # * TEMP for debugging
                
                for task, outputs_task in outputs.items():
                    loss = self.criterions[task](outputs_task, targets[task])
                    weight = self.config['training']['tasks'][task]['criterion']['weight']
                    total_loss_batch += (weight * loss)
                    
                    for metric in self.metrics[task].values():
                        metric.update(outputs_task, targets[task])
                
                self._log_visuals(epoch=epoch, images=images, targets=targets, outputs=outputs)
                
                total_loss += total_loss_batch.item()
                batch_tqdm.set_postfix({'batch_loss': f'{total_loss_batch.item():.4f}'})
        
        epoch_metrics = self._compute_metrics()
        epoch_metrics['optimization/loss_validation'] = total_loss / len(self.dataloader_val)
                
        return epoch_metrics
    
    def train(self) -> Dict[str, float]:
        logger.header('Starting Training Loop')
        
        best_val_epoch = -1
        best_val_loss = float('inf')
        best_val_epoch_metrics = {}
        
        epochs = self.config['training']['epochs']
        epochs_tqdm = tqdm(range(epochs), desc='Epochs', position=0, leave=True)
        for epoch in epochs_tqdm:
            self.n_logged_images = 0
            
            train_epoch_metrics = self._train_epoch()
            mlflow.log_metrics(train_epoch_metrics, step=epoch)
            
            val_epoch_metrics = self._validate_epoch(epoch=epoch)
            mlflow.log_metrics(val_epoch_metrics, step=epoch)
            
            val_loss = val_epoch_metrics['optimization/loss_validation']
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]['lr']
            mlflow.log_metric('optimization/learning_rate', lr, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_epoch = epoch
                best_val_loss = val_loss
                best_val_epoch_metrics = val_epoch_metrics
                
                torch.save({'model_state_dict': self.model.state_dict()}, os.path.join('cache', 'model_state.pth'))
                mlflow.log_metric('optimization/loss_validation_best', best_val_loss, step=epoch)
                
            epochs_tqdm.set_postfix({
                'lr': f'{lr:.2e}',
                'train_loss': f'{train_epoch_metrics['optimization/loss_training']:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'best_val_epoch': best_val_epoch,
                'best_val_loss': f'{best_val_loss:.4f}'
            })
            
        self._save_model()
            
        return best_val_epoch_metrics
    
    def train_without_validation(self) -> None:
        logger.header('Starting Training Loop without Validation')
        
        epochs = self.config['training']['epochs']
        epochs_tqdm = tqdm(range(epochs), desc='Epochs', position=0, leave=True)
        for epoch in epochs_tqdm:
            train_epoch_metrics = self._train_epoch()
            mlflow.log_metrics(train_epoch_metrics, step=epoch)
            
            train_loss = train_epoch_metrics['optimization/loss_training']
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(train_loss)
            else:
                self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]['lr']
            mlflow.log_metric('optimization/learning_rate', lr, step=epoch)
            
            epochs_tqdm.set_postfix({
                'lr': f'{lr:.2e}',
                'train_loss': f'{train_epoch_metrics['optimization/loss_training']:.4f}',
            })
        
        torch.save({'model_state_dict': self.model.state_dict()}, os.path.join('cache', 'model_state.pth'))
        self._save_model()
 