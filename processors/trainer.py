import os
import logging
from tqdm import tqdm
from typing import cast, Any, List, Dict

import torch
import mlflow
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from mlflow.models.signature import infer_signature

from utils import helpers
from utils.helpers import load, get_model_run_id, logits2disparity
from models.manager import AttachHead
from processors.base import BaseProcessor
from data.transforms import build_transforms
from criterions.manager import LossComposer

from utils.logger import CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))


class MetricsTracker:
    def __init__(self, tasks: List[str], criterion_keys: List[str], log_interval: int, dataloader_length: int):
        self.tasks = tasks
        self.criterion_keys = criterion_keys
        self.log_interval = log_interval
        self.dataloader_length = dataloader_length
        self.global_step = 0
        
        self._reset()
    
    def _reset(self):
        self.running_inter_loss = 0.0
        self.running_inter_loss_weights = {task: 0.0 for task in self.tasks}
        
        self.running_intra_losses = {task: 0.0 for task in self.tasks}
        self.running_intra_loss_weights = {task: {'target': 0.0, 'distillation': 0.0} for task in self.tasks}
        
        self.running_raw_task_losses = {task: 0.0 for task in self.criterion_keys}
        
        self.loss_for_scheduler = 0.0
    
    def update(self, inter_loss: float, inter_loss_weights: Dict[str, float], intra_losses: Dict[str, float], intra_loss_weights: Dict[str, Dict[str, float]], raw_task_losses: Dict[str, float]):
        self.running_inter_loss += inter_loss
        for task, weight in inter_loss_weights.items():
            self.running_inter_loss_weights[task] += weight
            
        for task, intra_loss in intra_losses.items():
            self.running_intra_losses[task] += intra_loss.item()
            
        for task, weights in intra_loss_weights.items():
            for intra_type, weight in weights.items():
                self.running_intra_loss_weights[task][intra_type] += weight

        for task, raw_loss in raw_task_losses.items():
            self.running_raw_task_losses[task] += raw_loss
        
        self.global_step += 1
    
    def should_log(self) -> bool:
        return self.global_step % self.log_interval == 0
    
    def get_metrics(self, epoch: int, batch_idx: int, lr_encoder: float, lr_decoders: float) -> Dict[str, float]:
        metrics = {}
        metrics['epoch_train'] = epoch + (batch_idx / self.dataloader_length)
        metrics['optimization/training/loss/inter'] = self.running_inter_loss / self.log_interval
        
        for task in self.tasks:
            metrics[f'optimization/training/loss/intra_{task}'] = self.running_intra_losses[task] / self.log_interval
            
            metrics[f'optimization/training/loss/weights/inter_{task}'] = self.running_inter_loss_weights[task] / self.log_interval
            metrics[f'optimization/training/loss/weights/intra_{task}_target'] = self.running_intra_loss_weights[task]['target'] / self.log_interval
            metrics[f'optimization/training/loss/weights/intra_{task}_distillation'] = self.running_intra_loss_weights[task]['distillation'] / self.log_interval
        
        for key in self.criterion_keys:
            metrics[f'optimization/training/loss/raw_{key}'] = self.running_raw_task_losses[key] / self.log_interval
        
        metrics['optimization/training/learning_rate_encoder'] = lr_encoder
        metrics['optimization/training/learning_rate_decoders'] = lr_decoders
        
        self.loss_for_scheduler = self.running_inter_loss / self.log_interval
        self._reset()
        
        return metrics

class Trainer(BaseProcessor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._load_data()
        self._load_model()
        self._load_teachers()
        self._load_components()
        self._init_metrics()
        
    def _load_data(self) -> None:
        logger.subheader('Load Data')

        dataset_class = load(self.config['data']['dataset'])
        self.tasks = [task for task, task_config in self.config['training']['tasks'].items() if task_config['enabled']]
        
        train_transforms = build_transforms(self.config, mode='train')
        val_mode = 'test' if self.config['training']['validation'] else 'train'
        val_transforms = build_transforms(self.config, mode=val_mode)
        
        g = torch.Generator()
        g.manual_seed(self.config['general']['seed'])
        
        dataset_train = dataset_class(
            mode='train',
            config=self.config,
            transforms=train_transforms,
        )
        self.dataloader_train = DataLoader(
            dataset_train,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['general']['num_workers'],
            pin_memory=self.config['general']['pin_memory'],
            generator=g,
            persistent_workers=True
        )
        helpers.check_dataleakage('train', dataset_train)
        
        dataset_val = dataset_class(
            mode='val',
            config=self.config,
            transforms=val_transforms,
        )
        self.dataloader_val = DataLoader(
            dataset_val,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['general']['num_workers'],
            pin_memory=self.config['general']['pin_memory'],
            generator=g,
            persistent_workers=True
        )
        helpers.check_dataleakage('val', dataset_val)
        
        if not self.config['training']['validation']:
            dataset_full = ConcatDataset([dataset_train, dataset_val])
            self.dataloader_train = DataLoader(
                dataset_full,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['general']['num_workers'],
                pin_memory=self.config['general']['pin_memory'],
                generator=g,
                persistent_workers=True
            )
            
        logger.info(f'Loaded datasets: {self.config["data"]["dataset"]} with batch size {self.config["training"]["batch_size"]}, num_workers {self.config["general"]["num_workers"]}, pin_memory {self.config["general"]["pin_memory"]}')
        dataset_val_length = len(dataset_val) if self.config['training']['validation'] else 0
        logger.info(f'Num of Samples - Training: {len(dataset_train)}, Validation: {dataset_val_length}')
        
        self.signature_input_example = dataset_train[0]['image'].unsqueeze(0)
        if 'disparity' in self.tasks: self.signature_input_example_right = dataset_train[0]['right_image'].unsqueeze(0)
            
        if 'segmentation' in self.tasks:
            self.segmentation_class_mappings = dataset_train.segmentation_class_mappings
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
                kd_config = task_config['distillation']
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
                
            kd_task_config = task_config['distillation']
            if kd_task_config['enabled']:
                kd_criterion_config = kd_task_config['criterion']
                KdCriterionClass = load(kd_criterion_config['name'])
                self.criterions[f'{task}_distillation'] = KdCriterionClass(**kd_criterion_config['params'])
                logger.info(f'KD Criterion for task {task}: {kd_criterion_config["name"]} with params {kd_criterion_config["params"]}')

        self.loss_composer = LossComposer(
            config=self.config,
            criterions=self.criterions,
            tasks=self.tasks
        ).to(self.device)
        
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
            }
        ]
        
        composer_params = [p for p in self.loss_composer.parameters() if p.requires_grad]
        if composer_params:
            model_parameter_groups.append({
                'params': composer_params,
                'lr': base_lr,
                'weight_decay': 0.0
            })
        
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
        
        self.accumulation_steps = self.config['training']['accumulate_grad_batches']
        
        train_log_interval = max(1, int(len(self.dataloader_train) * self.config['logging']['log_interval']))
        self.metrics_tracker = MetricsTracker(
            tasks=self.tasks,
            criterion_keys=list(self.criterions.keys()),
            log_interval=train_log_interval,
            dataloader_length=len(self.dataloader_train)
        )

    def _save_model(self):
        logger.subheader('Model Saving to MLflow')
        
        self.model.to('cpu')
        self.model.eval()
        
        with torch.no_grad():
            if 'disparity' not in self.tasks: signature_output_example = self.model(self.signature_input_example)
            else: signature_output_example = self.model(self.signature_input_example, self.signature_input_example_right)
            
            signature_output_example = {k: v.numpy() for k, v in signature_output_example.items()}
        signature = infer_signature(self.signature_input_example.numpy(), signature_output_example)
            
        if self.config['training']['validation']:
            for task_mode in ['segmentation', 'disparity', 'combined']:
                model_state_path = os.path.join(self.temp_path, f'model_state_{task_mode}.pth')
                if os.path.exists(model_state_path):
                    state_dict = torch.load(model_state_path)
                    self.model.load_state_dict(state_dict['model_state_dict'])
                    
                    mlflow.pytorch.log_model(
                        pytorch_model=self.model,
                        name=f'best_model_{task_mode}',
                        code_paths=['models/'],
                        signature=signature
                    )
                
                    logger.info(f'Logged best {task_mode} model.')
        else:
            model_state_path = os.path.join(self.temp_path, 'model_state.pth')
            torch.save({'model_state_dict': self.model.state_dict()}, model_state_path)
            
            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                name='best_model_',
                code_paths=['models/'],
                signature=signature
            )
        
            logger.info('Logged best model.')

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        
        self.optimizer.zero_grad()
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
                        outputs[f'{task}_distillation'] = outputs[task]
                        targets[f'{task}_distillation'] = teacher_outputs
                
            inter_loss, inter_loss_weights, intra_losses, intra_loss_weights, raw_task_losses = self.loss_composer(outputs, targets)
            accumulated_loss = inter_loss / self.accumulation_steps
            accumulated_loss.backward()
            
            if (idx + 1) % self.accumulation_steps == 0 or (idx + 1) == len(self.dataloader_train):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                                
            self.metrics_tracker.update(inter_loss.item(), inter_loss_weights, intra_losses, intra_loss_weights, raw_task_losses)
            
            if self.metrics_tracker.should_log():
                lr_encoder = self.optimizer.param_groups[0]['lr']
                lr_decoders = self.optimizer.param_groups[1]['lr']
                metrics = self.metrics_tracker.get_metrics(epoch, idx, lr_encoder, lr_decoders)
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
                
                inter_loss, inter_loss_weights, intra_losses, intra_loss_weights, raw_task_losses = self.loss_composer(outputs, targets)
                total_loss_weighted += inter_loss.item()
                
                self._log_visuals(epoch=epoch, images=left_images, targets=targets, outputs=outputs)
                
                if 'segmentation' in outputs:
                    for metric in self.metrics['segmentation'].values():
                        metric.update(outputs['segmentation'], targets['segmentation'])
                    
                    intercept_segmentation_logits = outputs['segmentation_intercept_features']
                    segmentation_targets = F.interpolate(targets['segmentation'], size=intercept_segmentation_logits.shape[2:], mode='nearest-exact')
                    self.misc_metrics['interceptDICE'].update(intercept_segmentation_logits, segmentation_targets)
                        
                if 'disparity' in outputs:
                    baseline, focal_length = data['baseline'].to(self.device), data['focal_length'].to(self.device)
                    for metric in self.metrics['disparity'].values():
                        metric.update(outputs['disparity'], targets['disparity'], baseline, focal_length)
                        
                    intercept_disparity = logits2disparity(outputs['disparity_intercept_features'], size=outputs['disparity_intercept_features'].shape[2:])
                    disparity_targets = F.interpolate(targets['disparity'], size=outputs['disparity_intercept_features'].shape[2:], mode='nearest-exact')
                    self.misc_metrics['interceptAbsRel'].update(intercept_disparity, disparity_targets, baseline, focal_length)
                            
                for task, raw_task_loss in raw_task_losses.items():
                    total_raw_task_losses[task] += raw_task_loss
                    total_task_weights[task] += inter_loss_weights[task]
                    
        epoch_metrics = self._compute_metrics(mode='validation')
        epoch_metrics['optimization/validation/loss/inter'] = total_loss_weighted / len(self.dataloader_val)
        for task in self.tasks:
            epoch_metrics[f'optimization/validation/loss/raw_{task}'] = total_raw_task_losses[task] / len(self.dataloader_val)
            epoch_metrics[f'optimization/validation/loss/weights/inter_{task}'] = total_task_weights[task] / len(self.dataloader_val)
            
        if self.config['training']['tasks']['segmentation']['enabled']:
            epoch_metrics['performance/validation/misc/interceptDICE_score'] = self.misc_metrics['interceptDICE'].compute()[1]
        if self.config['training']['tasks']['disparity']['enabled']:
            epoch_metrics['performance/validation/misc/interceptAbsRel_rate'] = self.misc_metrics['interceptAbsRel'].compute()['AbsRel_rate']
                
        return epoch_metrics
    
    def train(self) -> Dict[str, float]:
        try:
            logger.header('Training Loop')
            
            logger.info('Get pretrained Baseline')
            val_epoch_metrics = self._validate_epoch(epoch=0)
            mlflow.log_metric('epoch_validation', 0, step=self.metrics_tracker.global_step)
            mlflow.log_metrics(val_epoch_metrics, step=self.metrics_tracker.global_step)
            
            logger.info('Start Training')
            ema_alpha = self.config['training']['ema_alpha']
            ema_val_dice = None
            ema_val_absrel = None
            ema_val_heuristic = None
            best_ema_val_dice = float('-inf')
            best_ema_val_absrel = float('inf')
            best_ema_val_heuristic = float('-inf')
            
            epochs = self.config['training']['epochs']
            epochs_tqdm = tqdm(range(epochs), desc='Epochs', position=0, leave=True)
            for epoch in epochs_tqdm:
                self.n_logged_images = 0
                
                self._train_epoch(epoch=epoch)
                
                val_epoch_metrics = self._validate_epoch(epoch=epoch + 1)
                if 'segmentation' in self.tasks:
                    val_dice = val_epoch_metrics['performance/validation/segmentation/DICE_score/instrument']
                    if ema_val_dice: ema_val_dice = (ema_alpha * ema_val_dice) + ((1 - ema_alpha) * val_dice)
                    else: ema_val_dice = val_dice
                    val_epoch_metrics['performance/validation/misc/ema_val_dice'] = ema_val_dice
                    
                    if ema_val_dice > best_ema_val_dice:
                        best_ema_val_dice = ema_val_dice
                        
                        torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.temp_path, 'model_state_segmentation.pth'))
                        mlflow.log_metric('best/segmentation/epoch', epoch + 1, step=self.metrics_tracker.global_step)
                        for key, value in val_epoch_metrics.items(): mlflow.log_metric(f'best/segmentation/{key.replace("/", "_")}', value, step=self.metrics_tracker.global_step)
                        
                if 'disparity' in self.tasks:
                    val_absrel = val_epoch_metrics['performance/validation/disparity/AbsRel_rate']
                    if ema_val_absrel: ema_val_absrel = (ema_alpha * ema_val_absrel) + ((1 - ema_alpha) * val_absrel)
                    else: ema_val_absrel = val_absrel
                    val_epoch_metrics['performance/validation/misc/ema_val_absrel'] = ema_val_absrel
                    
                    if ema_val_absrel < best_ema_val_absrel:
                        best_ema_val_absrel = ema_val_absrel
                        
                        torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.temp_path, 'model_state_disparity.pth'))
                        mlflow.log_metric('best/disparity/epoch', epoch + 1, step=self.metrics_tracker.global_step)
                        for key, value in val_epoch_metrics.items(): mlflow.log_metric(f'best/disparity/{key.replace("/", "_")}', value, step=self.metrics_tracker.global_step)
                        
                if 'segmentation' in self.tasks and 'disparity' in self.tasks:
                    absrel_clamped = max(0.0, min(1.0, 1 - val_absrel))
                    denominator = val_dice + absrel_clamped
                    if denominator == 0: val_heuristic = 0.0
                    else: val_heuristic = (2 * val_dice * absrel_clamped) / (val_dice + absrel_clamped)
                    val_epoch_metrics['performance/validation/misc/heuristic'] = val_heuristic
                    
                    if ema_val_heuristic: ema_val_heuristic = (ema_alpha * ema_val_heuristic) + ((1 - ema_alpha) * val_heuristic)
                    else: ema_val_heuristic = val_heuristic
                    val_epoch_metrics['performance/validation/misc/ema_val_heuristic'] = ema_val_heuristic
                    
                    if ema_val_heuristic > best_ema_val_heuristic:
                        best_ema_val_heuristic = ema_val_heuristic
                        
                        torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.temp_path, 'model_state_combined.pth'))
                        mlflow.log_metric('best/combined/epoch', epoch + 1, step=self.metrics_tracker.global_step)
                        for key, value in val_epoch_metrics.items(): mlflow.log_metric(f'best/combined/{key.replace("/", "_")}', value, step=self.metrics_tracker.global_step)
                
                mlflow.log_metric('epoch_validation', epoch + 1, step=self.metrics_tracker.global_step)
                mlflow.log_metrics(val_epoch_metrics, step=self.metrics_tracker.global_step)
                self.loss_composer.step_weighting(metrics=val_epoch_metrics)
                
        except KeyboardInterrupt:
            logger.warning('Training interrupted. Saving current model...')
        finally:
            self._save_model()
            if hasattr(self, 'model'): del self.model
            if hasattr(self, 'optimizer'): del self.optimizer
            if hasattr(self, 'scheduler'): del self.scheduler
            
    def full_train(self) -> None:
        try:
            logger.header('Starting Full Training Loop without Validation')
            
            epochs = self.config['training']['epochs']
            epochs_tqdm = tqdm(range(epochs), desc='Epochs', position=0, leave=True)
            for epoch in epochs_tqdm:
                self._train_epoch(epoch=epoch)
        
            torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.temp_path, 'model_state.pth'))
        except KeyboardInterrupt:
            logger.warning('Training interrupted. Saving current model...')
            torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.temp_path, 'model_state.pth'))
        finally:
            self._save_model()
            if hasattr(self, 'model'): del self.model
            if hasattr(self, 'optimizer'): del self.optimizer
            if hasattr(self, 'scheduler'): del self.scheduler

