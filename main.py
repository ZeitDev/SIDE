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

from utils.loader import load
from utils.visualization import image_mask_overlay_figure

from typing import cast
from utils.logger import setup_logging, CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))

# * TASKS TO DO
# TODO:
# - Add transforms to datasets (apply to images and masks )

# TODO: change cross-validation metric comparison to task-specific metrics instead of loss, for that implement metrics first
# - Implement testing loop, Add metrics (maybe also in validation?)

# TODO: check if epoch run is really correct, especially with kd, debug line by line

class Trainer:
    def __init__(self, config, train_subsets, val_subsets=None):
        logger.header('Initializing Trainer')
        self.config = config
        self.train_subsets = train_subsets
        self.val_subsets = val_subsets
        self._setup()
        self._load_model()
        self._load_teachers()
        self._load_components()
        self._load_data()
        
    def _setup(self):
        logger.subheader('Setup')
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        logger.info(f'Restricting CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES")}')
        
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
        
    def _load_model(self):
        logger.subheader('Loading Model')
        
        encoder_config = self.config['training']['encoder']
        encoder = load(encoder_config['name'], **encoder_config['params'])
        logger.info(f'Loaded encoder: {encoder_config["name"]} with params {encoder_config["params"]}')
        
        decoders = {}
        tasks_config = self.config['training']['tasks']
        for task, task_config in tasks_config.items():
            if task_config['enabled']:
                decoder_config = task_config['decoder']
                decoders[task] = load(decoder_config['name'], **decoder_config['params'])
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
                    
    def _load_teachers(self):
        logger.subheader('Loading Teachers')
        self.kd_models = {}
        self.kd_criterions = {}
        for task, task_config in self.config['training']['tasks'].items():
            kd_config = task_config['knowledge_distillation']
            if kd_config['enabled']:
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
        
    def _load_components(self):
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
        
    def _load_data(self):
        logger.subheader('Load Data')

        data_config = self.config['data']
        dataset_class = load(data_config['dataset'])
        
        dataset_train = dataset_class(
            mode='train',
            tasks=self.config['training']['tasks'],
            subset_names=self.train_subsets,
            transforms=data_config['transforms']
        )
        self.dataloader_train = DataLoader(
            dataset_train,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'])
        
        dataset_val = dataset_class(
            mode='train',
            tasks=self.config['training']['tasks'],
            subset_names=self.val_subsets,
            transforms=data_config['transforms']
        ) if self.val_subsets else None
        if dataset_val:
            self.dataloader_val = DataLoader(
                dataset_val,
                batch_size=data_config['batch_size'],
                shuffle=False,
                num_workers=data_config['num_workers'],
                pin_memory=data_config['pin_memory'])
        
        logger.info(f'Loaded datasets: {data_config["dataset"]} with batch size {data_config["batch_size"]}, num_workers {data_config["num_workers"]}, pin_memory {data_config["pin_memory"]}')
        dataset_val_length = len(dataset_val) if dataset_val else 0
        logger.info(f'Num of Samples - Training: {len(dataset_train)}, Validation: {dataset_val_length}')
        
    def _log_visuals(self, epoch, images, targets, outputs):
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
    
    def _run_epoch(self, is_training):
        self.model.train(is_training)
        total_loss = 0.0
        
        dataloader = self.dataloader_train if is_training else self.dataloader_val
        if not dataloader: return float('nan')
        
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
                
        return total_loss / len(dataloader)
    
    def train(self):
        logger.header('Starting Training Loop')
        
        best_val_epoch = -1
        best_val_loss = float('inf')
        
        epochs = self.config['training']['epochs']
        epochs_tqdm = tqdm(range(epochs), desc='Epochs', position=0, leave=True)
        for epoch in epochs_tqdm:
            train_loss = self._run_epoch(is_training=True)
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            
            val_loss = self._run_epoch(is_training=False)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            
            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']
            mlflow.log_metric('learning_rate', lr, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                mlflow.pytorch.log_model(self.model, artifact_path='best_model') # type: ignore
                mlflow.log_metric('best_val_loss', best_val_loss, step=epoch)
                
            epochs_tqdm.set_postfix({
                'lr': f'{lr:.2e}',
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'best_val_epoch': best_val_epoch,
                'best_val_loss': f'{best_val_loss:.4f}'
            })
            
        return best_val_loss
            
class Tester:
    def __init__(self):
        pass
    
    def test(self):
        pass
    
def main():
    parser = argparse.ArgumentParser(description='SIDE Training and Testing')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()
    
    with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
    with open(args.config, 'r') as f: experiment_config = yaml.safe_load(f)
    config = deep_merge(experiment_config, base_config)

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
                mlflow_log_run(config, log_filepath)
                fold_val_losses = []
                
                for i, val_subset in enumerate(all_train_subsets):
                    with mlflow.start_run(run_name=f'fold_{i+1}', nested=True) as sub_run:
                        logger.info(f'Starting Fold {i+1}/{len(all_train_subsets)}: Validation Subset: {val_subset}')
                        mlflow.log_param('validation_subset', val_subset)
                        mlflow.log_param('fold', i+1)
                        
                        train_subsets = [s for s in all_train_subsets if s != val_subset]
                        trainer = Trainer(config, train_subsets=train_subsets, val_subsets=[val_subset])
                        best_val_loss = trainer.train()
                        fold_val_losses.append(best_val_loss)
                        
                mean_val_loss = float(np.mean(fold_val_losses))
                std_val_loss = float(np.std(fold_val_losses))
                
                logger.subheader('Cross-Validation Summary')
                logger.info(f'Fold Validation Losses: {fold_val_losses}')
                logger.info(f'Mean Validation Loss: {mean_val_loss:.4f}')
                logger.info(f'Std Dev Validation Loss: {std_val_loss:.4f}')

                mlflow.log_metric('cv_mean_val_loss', mean_val_loss)
                mlflow.log_metric('cv_std_val_loss', std_val_loss)
        else:
            logger.header(f'Starting Full Training: {run_datetime}')
            with mlflow.start_run(run_name=run_datetime) as run:
                mlflow_log_run(config, log_filepath)
                trainer = Trainer(config, train_subsets=all_train_subsets)
                trainer.train()
            
        #     tester = Tester()
        #     tester.test()
        
    except KeyboardInterrupt:
        logger.warning('Training interrupted by user')
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        if mlflow.active_run(): mlflow.end_run()
        logger.single('MLflow run cleaned up')

def mlflow_log_run(config, log_filepath):
    mlflow.log_params(config)
    mlflow.log_artifact(__file__)
    mlflow.log_artifact(log_filepath, artifact_path='logs')
    for folder in ['configs', 'criterions', 'data', 'metrics', 'models', 'utils']:
        if os.path.isdir(folder):
            mlflow.log_artifacts(folder, artifact_path=folder)

def deep_merge(source, destination):
    """Recursively merges source dict into destination dict."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination

if __name__ == "__main__":
    main()