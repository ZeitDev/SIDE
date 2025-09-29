import os
import yaml
import argparse
import mlflow
import datetime
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from utils.loader import load

# TODO:
# - Add transforms to datasets (apply to images and masks )
# - Implement training loop
# - Implement testing loop
# - Add metrics
# - Add overlay image logging

class Trainer:
    def __init__(self, config):
        print('\n# Initializing Trainer')
        self.config = config
        self._setup()
        self._load_model()
        self._load_components()
        self._load_data()
        
        
    def _setup(self):
        print('\n## Setup')
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'### Using device: {self.device}')
        
        # Clear GPU memory and set seeds for reproducibility
        torch.cuda.empty_cache()
        seed = self.config['misc']['seed']
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f'### Set reproducibility seed to {seed}')
        
    def _load_model(self):
        print('\n## Loading Model')
        # Load encoder and decoder, combine into model, load checkpoint
        encoder_config = self.config['model']['encoder']
        encoder = load(encoder_config['name'], **encoder_config['params'])
        print(f'### Loaded encoder: {encoder_config["name"]} with params {encoder_config["params"]}')
        
        decoders = {}
        decoder_config = self.config['model']['decoders']
        for task, task_decoder in decoder_config.items():
            if task_decoder['enabled']:
                decoders[task] = load(task_decoder['name'], **task_decoder['params'])
                print(f'### Loaded decoder for task {task}: {task_decoder["name"]} with params {task_decoder["params"]}')
            
        self.model = load(
            'models.combiner.Combiner', 
            encoder=encoder, 
            decoders=decoders
        ).to(self.device)
        
        if self.config['model']['checkpoint']:
            print(f'### Resuming checkpoint {self.config["model"]["checkpoint"]}')
            state_dict = torch.load(self.config['model']['checkpoint'], map_location=self.device)
            self.model.load_state_dict(state_dict['model_state_dict'])
        elif self.config['model']['finetune']:
            print(f'### Finetuning {self.config["model"]["finetune"]} with frozen encoder')
            state_dict = torch.load(self.config['model']['finetune'], map_location=self.device)
            self.model.load_state_dict(state_dict['model_state_dict'])
            for param in self.model.encoder.parameters():
                param.requires_grad = False
                
        if self.config['training']['knowledge_distillation']['enabled']: self._load_teachers()
    
    def _load_teachers(self):
        print('\n## Loading Teachers')
        self.teachers = []
        kd_config = self.config['training']['knowledge_distillation']
        for teacher_path in kd_config['teacher_paths']:
            teacher_encoder = load(kd_config['encoder']['name'], **kd_config['encoder']['params'])
            print(f'### Loaded teacher encoder: {kd_config["encoder"]["name"]} with params {kd_config["encoder"]["params"]}')
            
            teacher_decoder = load(kd_config['decoder']['name'], **kd_config['decoder']['params'])
            print(f'### Loaded teacher decoder: {kd_config["decoder"]["name"]} with params {kd_config["decoder"]["params"]}')
            
            teacher_model = load(
                'models.combiner.Combiner',
                encoder=teacher_encoder,
                decoders={'segmentation': teacher_decoder}
            ).to(self.device)
            
            state_dict = torch.load(teacher_path, map_location=self.device)
            teacher_model.load_state_dict(state_dict['model_state_dict'])
            teacher_model.eval()
            self.teachers.append(teacher_model)
            print(f'### Loaded teacher {teacher_path}')
        
    def _load_components(self):
        print('\n## Loading Components')
        # Load loss function
        criterion_config = self.config['training']['criterion']
        self.criterion = load(
            criterion_config['name'], 
            **criterion_config['params'])
        print(f'### Criterion: {criterion_config["name"]} with params {criterion_config["params"]}')
        
        # Setup optimizer and scheduler
        optimizer_config = self.config['training']['optimizer']
        optimizer_class = load(optimizer_config['name'])
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **optimizer_config['params'])
        print(f'### Optimizer: {optimizer_config["name"]} with params {optimizer_config["params"]}')
        
        scheduler_config = self.config['training']['scheduler']
        scheduler_class = load(scheduler_config['name'])
        self.scheduler = scheduler_class(
            self.optimizer,
            **scheduler_config['params'])
        print(f'### Scheduler: {scheduler_config["name"]} with params {scheduler_config["params"]}')
        
    def _load_data(self):
        print('\n## Load Data')

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
        
        print(f'### Loaded datasets: {data_config["dataset"]} with batch size {data_config["batch_size"]}, num_workers {data_config["num_workers"]}, pin_memory {data_config["pin_memory"]}')
        print(f'### Num of Samples: Train: {len(dataset_train)}, Val: {len(dataset_val)}, Test: {len(dataset_test)}')
        
    def train(self):
        pass

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
        mlflow.set_experiment(os.path.splitext(os.path.basename(args.config))[0])
        run_datetime = datetime.datetime.now().strftime("%y%m%d:%H%M")
        with mlflow.start_run(run_name=run_datetime) as run:
            print(f'# Starting training run: {run_datetime}')
            
            mlflow.log_params(config)
            mlflow.log_artifact(__file__)
            for folder in ['models', 'criterions', 'metrics', 'data']:
                for file in os.listdir(folder):
                    mlflow.log_artifact(os.path.join(folder, file), artifact_path=folder)
            
            trainer = Trainer(config)
            trainer.train()
            
            tester = Tester()
            tester.test()
        
    except KeyboardInterrupt:
        print('# Training interrupted')
    finally:
        if mlflow.active_run(): mlflow.end_run()
        print('# MLflow run cleaned up')

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