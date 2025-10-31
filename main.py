import os
import yaml
import logging
import argparse
import datetime
import random
import numpy as np

from tqdm import tqdm
import mlflow
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from metrics.segmentation import IoU, Dice

from utils import helpers
from utils.loader import load
from utils.visualization import image_mask_overlay_figure

from typing import cast, Dict, Any, List, Optional
from utils.logger import setup_logging, CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))

# * TASKS TO DO
# TODO: FIX model saving, does not work with task dicts
# TODO: Implement MAE metric for disparity task
# TODO: Implement testing loop

# * Tested so far
# ! Disparity task not tested yet
# ! Metrics not tested yet
# ! Transforms not tested yet
# ! Knowledge Distillation not tested yet

class Trainer:
    def __init__(self, config: Dict[str, Any], train_subsets: List[str], val_subsets: Optional[List[str]] = None):
        logger.header('Initializing Trainer')
        self.config = config
        self.train_subsets = train_subsets
        self.val_subsets = val_subsets
        self._setup()
        self._load_data()
        self._load_model()
        self._load_teachers()
        self._load_components()
        self._init_metrics()
        
    def _setup(self) -> None:
        logger.subheader('Setup')
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        logger.info(f'Restricting to GPU {os.environ.get("CUDA_VISIBLE_DEVICES")}')
        
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
        
    def _load_data(self) -> None:
        logger.subheader('Load Data')

        data_config = self.config['data']
        dataset_class = load(data_config['dataset'])
        
        train_transforms = helpers.build_transforms(data_config['transforms']['train'])
        val_transforms = helpers.build_transforms(data_config['transforms']['val'])
        
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
            pin_memory=data_config['pin_memory'])
        
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
                pin_memory=data_config['pin_memory'])
            
        input_example_image, _ = dataset_train[0] 
        self.input_example = input_example_image.unsqueeze(0)
            
        logger.info(f'Loaded datasets: {data_config["dataset"]} with batch size {data_config["batch_size"]}, num_workers {data_config["num_workers"]}, pin_memory {data_config["pin_memory"]}')
        dataset_val_length = len(dataset_val) if dataset_val else 0
        logger.info(f'Num of Samples - Training: {len(dataset_train)}, Validation: {dataset_val_length}')
        
        if self.config['training']['tasks']['segmentation']['enabled']:
            self.segmentation_class_mappings = dataset_train.class_mappings
            num_classes = len(self.segmentation_class_mappings)
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
            'models.combiner.Combiner', 
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
                        'models.combiner.Combiner',
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
        # Load loss functions
        self.criterions = {}
        tasks_config = self.config['training']['tasks']
        for task, task_config in tasks_config.items():
            if task_config['enabled']:
                criterion_config = task_config['criterion']
                self.criterions[task] = load(
                    criterion_config['name'], 
                    **criterion_config['params'])
                logger.info(f'Criterion for task {task}: {criterion_config["name"]} with params {criterion_config["params"]}')

        # Setup optimizer and scheduler
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
        
    def _init_metrics(self) -> None:
        logger.subheader('Initializing Metrics')
        self.metrics = {}
        tasks_config = self.config['training']['tasks']
        
        if tasks_config['segmentation']['enabled']:
            self.metrics['segmentation'] = {}
            
            num_classes = tasks_config['segmentation']['decoder']['params']['num_classes']
            self.metrics['segmentation']['IoU'] = IoU(num_classes=num_classes, device=self.device)
            self.metrics['segmentation']['Dice'] = Dice(num_classes=num_classes, device=self.device)
            logger.info(f'Initialized IoU and Dice metrics for segmentation with {num_classes} classes.')
            
        if tasks_config['disparity']['enabled']:
            pass # TODO: implement disparity metrics
        
    def _log_visuals(self, epoch: int, images: torch.Tensor, targets: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> None:
        log_n_images = min(self.config['logging']['n_validation_images'], images.size(0))
        if log_n_images > 0:
            if self.config['training']['tasks']['segmentation']['enabled']:
                for i in range(log_n_images):
                    figure = image_mask_overlay_figure(
                        image=images[i].cpu(),
                        mask=targets['segmentation'][i].cpu(),
                        output=outputs['segmentation'][i].cpu(),
                        epoch=epoch
                    )
                    mlflow.log_figure(figure, artifact_file=f'images/epoch_{epoch}_image_{i}.png')
                    plt.close(figure)
    
    def _run_epoch(self, is_training: bool) -> Dict[str, float]:
        self.model.train(is_training)
        is_validation = not is_training
        total_loss = 0.0
        
        if is_validation:
            for task_metrics in self.metrics.values():
                for metric in task_metrics.values():
                    metric.reset()
        
        dataloader = self.dataloader_train if is_training else self.dataloader_val
        if not dataloader: return {'loss': float('inf')}
        
        phase = 'Training' if is_training else 'Validation'
        
        batch_tqdm = tqdm(dataloader, desc=phase, position=1, leave=False)
        for images, targets in batch_tqdm:
            images = images.to(self.device)
            targets = {key: value.to(self.device) for key, value in targets.items()}
            
            with torch.set_grad_enabled(is_training):
                total_loss_batch = torch.tensor(0.0, device=self.device)
                
                outputs = self.model(images)
                
                for task, output in outputs.items():
                    loss = self.criterions[task](output, targets[task])
                    weight = self.config['training']['tasks'][task]['criterion']['weight']
                    total_loss_batch += weight * loss
                    
                    if is_validation:
                        for metric in self.metrics[task].values():
                            metric.update(output, targets[task])
                    
                if is_training and self.kd_models:
                    for task, kd_teachers in self.kd_models.items():
                            student_output = outputs[task]
                            
                            with torch.no_grad():
                                teacher_outputs = [teacher(images)[task] for teacher in kd_teachers]
                                mean_teacher_output = torch.mean(torch.stack(teacher_outputs), dim=0)
                            
                            kd_loss = self.kd_criterions[task](student_output, mean_teacher_output)
                            kd_weight = self.config['training']['tasks'][task]['knowledge_distillation']['criterion']['weight']
                            total_loss_batch += kd_weight * kd_loss
                    
                if is_training:
                    self.optimizer.zero_grad()
                    total_loss_batch.backward()
                    self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            batch_tqdm.set_postfix({'batch_loss': f'{total_loss_batch.item():.4f}'})

        epoch_metrics = {'loss': total_loss / len(dataloader)}
        if is_validation:
            for task, task_metrics in self.metrics.items():
                for metric_name, metric in task_metrics.items():
                    metric_results = metric.compute()
                    
                    for key, value in metric_results.items():
                        if isinstance(key, int):
                            class_name = self.segmentation_class_mappings[key]
                            epoch_metrics[f'{task}_{metric_name}_{class_name}'] = value
                        else:
                            epoch_metrics[f'{task}_{metric_name}_{key}'] = value
                
        return epoch_metrics
    
    def train(self) -> Dict[str, float]:
        logger.header('Starting Training Loop')
        
        best_val_epoch = -1
        best_val_loss = float('inf')
        best_val_epoch_metrics = {}
        
        epochs = self.config['training']['epochs']
        epochs_tqdm = tqdm(range(epochs), desc='Epochs', position=0, leave=True)
        for epoch in epochs_tqdm:
            train_epoch_metrics = self._run_epoch(is_training=True)
            mlflow.log_metrics(train_epoch_metrics, step=epoch)
            
            val_epoch_metrics = self._run_epoch(is_training=False)
            mlflow.log_metrics(val_epoch_metrics, step=epoch)
            
            val_loss = val_epoch_metrics['loss']
            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']
            mlflow.log_metric('learning_rate', lr, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_epoch = epoch
                best_val_loss = val_loss
                best_val_epoch_metrics = val_epoch_metrics
                
                self.model.to('cpu')
                mlflow.pytorch.log_model( # type: ignore
                    pytorch_model=self.model, 
                    name='best_model', 
                    input_example=self.input_example.numpy().astype(np.float32)
                )
                self.model.to(self.device)
                mlflow.log_metric('best_val_loss', best_val_loss, step=epoch)
                
            epochs_tqdm.set_postfix({
                'lr': f'{lr:.2e}',
                'train_loss': f'{train_epoch_metrics['loss']:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'best_val_epoch': best_val_epoch,
                'best_val_loss': f'{best_val_loss:.4f}'
            })
            
        return best_val_epoch_metrics
    
    def train_without_validation(self) -> None:
        logger.header('Starting Training Loop without Validation')
        
        epochs = self.config['training']['epochs']
        epochs_tqdm = tqdm(range(epochs), desc='Epochs', position=0, leave=True)
        for epoch in epochs_tqdm:
            train_epoch_metrics = self._run_epoch(is_training=True)
            mlflow.log_metrics(train_epoch_metrics, step=epoch)
            
            lr = self.optimizer.param_groups[0]['lr']
            mlflow.log_metric('learning_rate', lr, step=epoch)
            
            epochs_tqdm.set_postfix({
                'lr': f'{lr:.2e}',
                'train_loss': f'{train_epoch_metrics['loss']:.4f}',
            })
        
        self.model.to('cpu')
        mlflow.pytorch.log_model( # type: ignore
            pytorch_model=self.model, 
            name='final_model', 
            input_example=self.input_example.numpy().astype(np.float32)
        )
        self.model.to(self.device)
                    
class Tester:
    def __init__(self) -> None:
        pass
    
    def test(self) -> None:
        pass
    
def main():
    parser = argparse.ArgumentParser(description='SIDE Training and Testing')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()
    
    with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
    with open(args.config, 'r') as f: experiment_config = yaml.safe_load(f)
    config = helpers.deep_merge(experiment_config, base_config)

    try:
        experiment_name = os.path.splitext(os.path.basename(args.config))[0]
        mlflow.set_experiment(experiment_name)
        run_datetime = datetime.datetime.now().strftime("%y%m%d:%H%M")
        
        log_filepath = os.path.join('logs', f'{run_datetime}_{experiment_name}.log')
        setup_logging(log_filepath=log_filepath)
        
        dataset_class = load(config['data']['dataset'])
        all_train_subsets = dataset_class(mode='train').get_all_subset_names()
        logger.info(f'Found {len(all_train_subsets)} training subsets: {all_train_subsets}')
        
        if config['data']['cross_validation']:
            with mlflow.start_run(run_name=f'{run_datetime}_cross_validation') as run:
                logger.header('Starting Cross-Validation Training')
                helpers.mlflow_log_run(config, log_filepath)
                
                fold_val_metrics_summary = {}
                
                for i, val_subset in enumerate(all_train_subsets):
                    with mlflow.start_run(run_name=f'fold_{i+1}', nested=True) as sub_run:
                        logger.info(f'Starting Fold {i+1}/{len(all_train_subsets)}: Validation Subset: {val_subset}')
                        mlflow.log_param('validation_subset', val_subset)
                        mlflow.log_param('fold', i+1)
                        
                        train_subsets = [s for s in all_train_subsets if s != val_subset]
                        trainer = Trainer(config, train_subsets=train_subsets, val_subsets=[val_subset])
                        best_val_epoch_metrics = trainer.train()
                        
                        for metric_name, metric_value in best_val_epoch_metrics.items():
                            if any(m in metric_name for m in ['mIoU', 'mDICE', 'mMAE']):
                                fold_val_metrics_summary.setdefault(metric_name, []).append(metric_value)
                        
                
                logger.subheader('Cross-Validation Summary')
                for metric_name, metric_values in fold_val_metrics_summary.items():
                    mean_metric = float(np.mean(metric_values))
                    std_metric = float(np.std(metric_values))
                    
                    logger.info(f'Metric: {metric_name}')
                    logger.info(f'Mean = {mean_metric:.4f}, Std = {std_metric:.4f}')
                    
                    for fold_idx, v in enumerate(metric_values):
                        logger.info(f'Fold {fold_idx+1} ({all_train_subsets[fold_idx]}): {v:.4f}')
                        mlflow.log_metric(f'cv_{metric_name}_fold_{fold_idx+1}', v)
                        
                    mlflow.log_metric(f'cv_mean_{metric_name}', mean_metric)
                    mlflow.log_metric(f'cv_std_{metric_name}', std_metric)
        else:
            logger.header(f'Starting Full Training: {run_datetime}')
            with mlflow.start_run(run_name=run_datetime) as run:
                helpers.mlflow_log_run(config, log_filepath)
                trainer = Trainer(config, train_subsets=all_train_subsets)
                trainer.train_without_validation()
            
        tester = Tester()
        tester.test()
        
    except KeyboardInterrupt:
        logger.warning('Training interrupted by user')
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        if mlflow.active_run(): mlflow.end_run()
        logger.single('MLflow run cleaned up')
            
if __name__ == "__main__":
    main()