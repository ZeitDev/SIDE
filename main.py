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

# TODO:
# - Add transforms to datasets (apply to images and masks )
# - Implement testing loop, Add metrics (maybe also in validation?)
# - Add semantic test cases for debugging / making sure everything works as intended - Andrew Karpathy style
# ! check if epoch run is really correct, especially with kd

class Trainer:
    def __init__(self, config):
        logger.header('Initializing Trainer')
        self.config = config
        self._setup()
        self._load_model()
        self._load_teachers()
        self._load_components()
        self._load_data()
        
    def _setup(self):
        logger.subheader('Setup')
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        # Clear GPU memory and set seeds for reproducibility
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
        dataset_train = load(data_config['dataset'], mode='train')
        dataset_val = load(data_config['dataset'], mode='val')
        dataset_test = load(data_config['dataset'], mode='test')
        
        self.dataloader_train = DataLoader(
            dataset_train,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'])
        self.dataloader_val = DataLoader(
            dataset_val,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'])
        self.dataloader_test = DataLoader(
            dataset_test,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'])
        
        logger.info(f'Loaded datasets: {data_config["dataset"]} with batch size {data_config["batch_size"]}, num_workers {data_config["num_workers"]}, pin_memory {data_config["pin_memory"]}')
        logger.info(f'Num of Samples: Train: {len(dataset_train)}, Val: {len(dataset_val)}, Test: {len(dataset_test)}')
        
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
            
class Tester:
    def __init__(self):
        pass
    
    def test(self):
        pass
    
def main():
    parser = argparse.ArgumentParser(description='Train SIDE')
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
        
        with mlflow.start_run(run_name=run_datetime) as run:
            logger.header(f'Starting Run: {run_datetime}')
            
            mlflow.log_params(config)
            mlflow.log_artifact(__file__)
            mlflow.log_artifact(log_filepath, artifact_path='logs')
            for folder in ['configs', 'criterions', 'data', 'metrics', 'models', 'utils']:
                for file in os.listdir(folder):
                    mlflow.log_artifact(os.path.join(folder, file), artifact_path=folder)
            
            trainer = Trainer(config)
            trainer.train()
            
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