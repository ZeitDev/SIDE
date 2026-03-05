import os
import logging
from tqdm import tqdm
from typing import cast, Any, List, Dict, Optional

import torch
import mlflow
from torch.utils.data import DataLoader 
from mlflow.models.signature import infer_signature

from utils import helpers
from utils.helpers import load, log_vram, get_model_run_id
from models.manager import AttachHead
from processors.base import BaseProcessor
from data.transforms import build_transforms
from criterions.automatic_weighted_loss import AutomaticWeightedLoss

from utils.logger import CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))


class MetricsTracker:
    def __init__(self, criterion_keys: List[str], log_interval: int, dataloader_len: int):
        self.log_interval = log_interval
        self.dataloader_len = dataloader_len
        self.criterion_keys = criterion_keys
        self.global_step = 0
        
        self._reset()
    
    def _reset(self):
        self.running_loss_weighted = 0.0
        self.running_raw_task_losses = {task: 0.0 for task in self.criterion_keys}
        self.running_task_weights = {task: 0.0 for task in self.criterion_keys}
        self.loss_for_scheduler = 0.0
    
    def update(self, loss: float, raw_task_losses: Dict[str, float], task_weights: Dict[str, float]):
        self.running_loss_weighted += loss
        
        for task, raw_loss in raw_task_losses.items():
            self.running_raw_task_losses[task] += raw_loss
            
        for task, weight in task_weights.items():
            self.running_task_weights[task] += weight
            
        self.global_step += 1
    
    def should_log(self) -> bool:
        return self.global_step % self.log_interval == 0
    
    def get_metrics(self, epoch: int, batch_idx: int, lr: float) -> Dict[str, float]:
        metrics = {}
        metrics['epoch_train'] = epoch + (batch_idx / self.dataloader_len)
        metrics['epoch_validation'] = epoch
        metrics['optimization/training/loss/auto_weighted_sum'] = self.running_loss_weighted / self.log_interval
        
        for task in self.criterion_keys:
            metrics[f'optimization/training/loss/raw/{task}'] = self.running_raw_task_losses[task] / self.log_interval
            metrics[f'optimization/training/loss/auto_weights/{task}'] = self.running_task_weights[task] / self.log_interval
        metrics['optimization/training/learning_rate'] = lr
        
        self.loss_for_scheduler = self.running_loss_weighted / self.log_interval
        self._reset()
        
        return metrics


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
        log_vram('Trainer Initialized')
        
    def _load_data(self) -> None:
        logger.subheader('Load Data')

        data_config = self.config['data']
        dataset_class = load(data_config['dataset'])
        
        self.tasks = [task for task, task_config in self.config['training']['tasks'].items() if task_config['enabled']]
        
        train_transforms = build_transforms(self.config, mode='train')
        val_transforms = build_transforms(self.config, mode='test')
        
        dataset_train = dataset_class(
            mode='train',
            config=self.config,
            transforms=train_transforms,
            subset_names=self.train_subsets
        )
        self.dataloader_train = DataLoader(
            dataset_train,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            persistent_workers=False
        )
        helpers.check_dataleakage('train', dataset_train)
        
        dataset_val = dataset_class(
            mode='train',
            config=self.config,
            transforms=val_transforms,
            subset_names=self.val_subsets
        ) if self.val_subsets else None
        if dataset_val:
            self.dataloader_val = DataLoader(
                dataset_val,
                batch_size=data_config['batch_size'],
                shuffle=False,
                num_workers=data_config['num_workers'],
                pin_memory=data_config['pin_memory'],
                persistent_workers=False
            )
            helpers.check_dataleakage('train', dataset_val)
            
        self.signature_input_example = dataset_train[0]['image'].unsqueeze(0)
        if 'disparity' in self.tasks: self.signature_input_example_right = dataset_train[0]['right_image'].unsqueeze(0)
            
        logger.info(f'Loaded datasets: {data_config["dataset"]} with batch size {data_config["batch_size"]}, num_workers {data_config["num_workers"]}, pin_memory {data_config["pin_memory"]}')
        dataset_val_length = len(dataset_val) if dataset_val else 0
        logger.info(f'Num of Samples - Training: {len(dataset_train)}, Validation: {dataset_val_length}')
        
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
                decoders[task] = AttachHead(
                    decoder_class=DecoderClass,
                    n_classes=self.config['data']['num_of_classes'][task],
                    encoder_channels=encoder.feature_info.channels(), 
                    encoder_reductions=encoder.feature_info.reduction(), 
                    **decoder_config['params']
                )
                logger.info(f'Loaded decoder for task {task}: {decoder_config["name"]} with params {decoder_config["params"]}')
            
        self.model = load(
            'models.manager.Combiner', 
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
        for task, task_config in self.config['training']['tasks'].items():
            if task_config['enabled']:
                kd_config = task_config['knowledge_distillation']
                if kd_config['enabled']:
                    if kd_config['name'] == 'offline':
                        logger.subheader(f'Loading offline teacher outputs from disk for {task}')
                        self.kd_models[task] = 'offline'
                    else:
                        logger.subheader(f'Loading Teachers for {task}')
                        
                        if kd_config['name'] == 'mlflow':
                            model_run_id = get_model_run_id(kd_config['state'])
                            teacher_model = mlflow.pytorch.load_model(f'runs:/{model_run_id}/best_model', map_location=self.device)
                        else:
                            teacher_class = load(kd_config['name'])
                            teacher_model = teacher_class().to(self.device)
                            if kd_config['state']:
                                state_dict = torch.load(kd_config['state'], map_location=self.device)
                                teacher_model.load_state_dict(state_dict['model_state_dict'])
                        
                        for param in teacher_model.parameters(): param.requires_grad = False
                        teacher_model.eval()
                    
                        self.kd_models[task] = teacher_model
                        logger.info(f'Loaded teacher {kd_config["state"]} for task {task} with model run ID {model_run_id}')

    def _load_components(self) -> None:
        logger.subheader('Loading Components')

        self.criterions = {}
        tasks_config = self.config['training']['tasks']
        for task, task_config in tasks_config.items():
            if task_config['enabled']:
                criterion_config = task_config['criterion']
                CriterionClass = load(criterion_config['name'])
                self.criterions[task] = CriterionClass(**criterion_config['params'])
                logger.info(f'Criterion for task {task}: {criterion_config["name"]} with params {criterion_config["params"]}')
                
            kd_task_config = task_config['knowledge_distillation']
            if kd_task_config['enabled']:
                kd_criterion_config = kd_task_config['criterion']
                KdCriterionClass = load(kd_criterion_config['name'])
                self.criterions[f'{task}_teacher'] = KdCriterionClass(**kd_criterion_config['params'])
                logger.info(f'KD Criterion for task {task}: {kd_criterion_config["name"]} with params {kd_criterion_config["params"]}')

        self.automatic_weighted_loss = AutomaticWeightedLoss(self.criterions).to(self.device)
        optimizer_config = self.config['training']['optimizer']
        base_lr = optimizer_config['params']['lr']
        model_parameter_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if 'encoder' in n and p.requires_grad],
                'lr': base_lr * optimizer_config['encoder_lr_mod']
            },
            {
                'params': [p for n, p in self.model.named_parameters() if 'encoder' not in n and p.requires_grad],
                'lr': base_lr
            },
            {
                'params': self.automatic_weighted_loss.parameters(),
                'lr': base_lr,
                'weight_decay': 0.0
            }
        ]
        OptimizerClass = load(optimizer_config['name'])
        self.optimizer = OptimizerClass(
            model_parameter_groups,
            **optimizer_config['params'])
        logger.info(f'Optimizer: {optimizer_config["name"]} with params {optimizer_config["params"]}')
        
        scheduler_config = self.config['training']['scheduler']
        SchedulerClass = load(scheduler_config['name'])
        self.scheduler = SchedulerClass(
            self.optimizer,
            **scheduler_config['params'])
        logger.info(f'Scheduler: {scheduler_config["name"]} with params {scheduler_config["params"]}')
        
        train_log_interval = max(1, int(len(self.dataloader_train) * self.config['logging']['log_interval']))
        self.metrics_tracker = MetricsTracker(
            criterion_keys=list(self.criterions.keys()),
            log_interval=train_log_interval,
            dataloader_len=len(self.dataloader_train)
        )

    def _save_model(self):
        logger.info('Saving best model to mlflow')        
        
        with torch.no_grad():
            if 'disparity' not in self.tasks: signature_output_example = self.model(self.signature_input_example)
            else: signature_output_example = self.model(self.signature_input_example, self.signature_input_example_right)
            
            signature_output_example = {k: v.numpy() for k, v in signature_output_example.items()}
        signature = infer_signature(self.signature_input_example.numpy(), signature_output_example)
            
        for task_mode in ['segmentation', 'disparity', 'combined']:
            model_state_path = os.path.join('.temp', f'model_state_{task_mode}.pth')
            if os.path.exists(model_state_path):
                state_dict = torch.load(model_state_path)
                self.model.load_state_dict(state_dict['model_state_dict'])
                self.model.to('cpu')
                self.model.eval()
                
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    name=f'best_model_{task_mode}',
                    code_paths=['models/'],
                    signature=signature
                )

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        
        batch_tqdm = tqdm(self.dataloader_train, desc='Training', position=1, leave=False)
        for idx, data in enumerate(batch_tqdm):
            left_images = data['image'].to(self.device)
            right_images = data['right_image'].to(self.device) if 'right_image' in data else None
            targets = {task: data[task].to(self.device) for task in self.tasks}

            with torch.set_grad_enabled(True):
                outputs = self.model(left_images, right_images)
                
                if self.kd_models:
                    for task, kd_teacher in self.kd_models.items():
                        if self.kd_models[task] == 'offline':
                            teacher_outputs = data[f'teacher_{task}'].to(self.device)
                        else:
                            with torch.no_grad():
                                teacher_outputs = kd_teacher(left_images, right_images)[task]
                        outputs[f'{task}_teacher'] = outputs[task]
                        targets[f'{task}_teacher'] = teacher_outputs
                
                loss, raw_task_losses = self.automatic_weighted_loss(outputs, targets)
                        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    task_weights = {task: torch.exp(-s_param).item() for task, s_param in self.automatic_weighted_loss.logarithmic_variances.items()}
                
                self.metrics_tracker.update(loss.item(), raw_task_losses, task_weights)
            
            if self.metrics_tracker.should_log():
                lr = self.optimizer.param_groups[0]['lr']
                metrics = self.metrics_tracker.get_metrics(epoch, idx, lr)
                mlflow.log_metrics(metrics, step=self.metrics_tracker.global_step)
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss_weighted = 0.0
        total_raw_task_losses = {task: 0.0 for task in self.tasks}
        total_task_weights = {task: 0.0 for task in self.tasks}
        
        for task_metrics in self.metrics.values():
            for metric in task_metrics.values():
                metric.reset()
        
        batch_tqdm = tqdm(self.dataloader_val, desc='Validation', position=1, leave=False)
        for data in batch_tqdm:
            left_images = data['image'].to(self.device)
            right_images = data['right_image'].to(self.device) if 'right_image' in data else None
            targets = {task: data[task].to(self.device) for task in self.tasks}
            
            with torch.no_grad():
                outputs = self.model(left_images, right_images)
                
                loss, raw_task_losses = self.automatic_weighted_loss(outputs, targets)
                total_loss_weighted += loss.item()
                
                self._log_visuals(epoch=epoch, images=left_images, targets=targets, outputs=outputs)
                
                if 'segmentation' in outputs:
                    for metric in self.metrics['segmentation'].values():
                        metric.update(outputs['segmentation'], targets['segmentation'])
                if 'disparity' in outputs:
                    baseline, focal_length = data['baseline'].to(self.device), data['focal_length'].to(self.device)
                    for metric in self.metrics['disparity'].values():
                        metric.update(outputs['disparity'], targets['disparity'], baseline, focal_length)
                        
                batch_tqdm.set_postfix({'batch_loss': f'{loss.item():.4f}'})
                for task, raw_task_loss in raw_task_losses.items():
                    total_raw_task_losses[task] += raw_task_loss
                with torch.no_grad():
                    for task in self.tasks:
                        s_param = self.automatic_weighted_loss.logarithmic_variances[task]
                        total_task_weights[task] += torch.exp(-s_param).item()
        
        epoch_metrics = self._compute_metrics(mode='validation')
        epoch_metrics['optimization/validation/loss/auto_weighted_sum'] = total_loss_weighted / len(self.dataloader_val)
        for task in self.tasks:
            epoch_metrics[f'optimization/validation/loss/raw/{task}'] = total_raw_task_losses[task] / len(self.dataloader_val)
            epoch_metrics[f'optimization/validation/loss/auto_weights/{task}'] = total_task_weights[task] / len(self.dataloader_val)
                
        return epoch_metrics
    
    def train(self) -> Dict[str, float]:
        logger.header('Starting Training Loop')
        
        best_val_epoch = -1
        best_val_loss = float('inf')
        best_val_dice = 0.0
        best_val_bad3 = 1.0
        best_val_heuristic = float('inf')
        best_val_epoch_metrics = {}
        
        epochs = self.config['training']['epochs']
        epochs_tqdm = tqdm(range(epochs), desc='Epochs', position=0, leave=True)
        for epoch in epochs_tqdm:
            self.n_logged_images = 0
            
            self._train_epoch(epoch=epoch)
            
            val_epoch_metrics = self._validate_epoch(epoch=epoch)
            mlflow.log_metrics(val_epoch_metrics, step=self.metrics_tracker.global_step)
            
            val_loss = val_epoch_metrics['optimization/validation/loss/auto_weighted_sum']
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): self.scheduler.step(val_loss)
            else: self.scheduler.step()
            
            if 'segmentation' in self.tasks:
                val_dice = val_epoch_metrics['performance/validation/segmentation/DICE_score/instrument']
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    
                    torch.save({'model_state_dict': self.model.state_dict()}, os.path.join('.temp', 'model_state_segmentation.pth'))
                    mlflow.log_metric('best_segmentation/optimization/validation/epoch', epoch, step=self.metrics_tracker.global_step)
                    for key, value in val_epoch_metrics.items(): mlflow.log_metric(f'best_segmentation/{key}', value, step=self.metrics_tracker.global_step)
                    
            if 'disparity' in self.tasks:
                val_bad3 = val_epoch_metrics['performance/validation/disparity/Bad3_rate']
                if val_bad3 < best_val_bad3:
                    best_val_bad3 = val_bad3
                    
                    torch.save({'model_state_dict': self.model.state_dict()}, os.path.join('.temp', 'model_state_disparity.pth'))
                    mlflow.log_metric('best_disparity/optimization/validation/epoch', epoch, step=self.metrics_tracker.global_step)
                    for key, value in val_epoch_metrics.items(): mlflow.log_metric(f'best_disparity/{key}', value, step=self.metrics_tracker.global_step)
                    
            if 'segmentation' in self.tasks and 'disparity' in self.tasks:
                val_heuristic = ((1 - val_dice) ** 2 + val_bad3 ** 2) ** 0.5
                if val_heuristic < best_val_heuristic:
                    best_val_heuristic = val_heuristic
                    
                    torch.save({'model_state_dict': self.model.state_dict()}, os.path.join('.temp', 'model_state_combined.pth'))
                    mlflow.log_metric('best_combined/optimization/validation/epoch', epoch, step=self.metrics_tracker.global_step)
                    for key, value in val_epoch_metrics.items(): mlflow.log_metric(f'best_combined/{key}', value, step=self.metrics_tracker.global_step)
                
            log_vram(f'Trainer Epoch {epoch}')
            
        self._save_model()
        del self.model
        del self.optimizer
        del self.scheduler
            
        return best_val_epoch_metrics
    
    def train_without_validation(self) -> None:
        logger.header('Starting Training Loop without Validation')
        
        epochs = self.config['training']['epochs']
        epochs_tqdm = tqdm(range(epochs), desc='Epochs', position=0, leave=True)
        for epoch in epochs_tqdm:
            self._train_epoch(epoch=epoch)
            
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(self.metrics_tracker.loss_for_scheduler)
            else:
                self.scheduler.step()
            
            log_vram(f'Full Trainer Epoch {epoch}')
            
        
        torch.save({'model_state_dict': self.model.state_dict()}, os.path.join('.temp', 'model_state.pth'))
        self._save_model()
        del self.model
        del self.optimizer
        del self.scheduler

