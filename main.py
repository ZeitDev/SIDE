import os
import yaml
import argparse
import mlflow
import datetime
import torch
import random
import numpy as np

from utils.loader import load

# TODO:
# - Add transforms to datasets
# - Implement training loop
# - Implement testing loop
# - Add metrics
# - Add overlay image logging

class Trainer:
    def __init__(self, config):
        self.config = config
        self._setup()
        
    def _setup(self):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Clear GPU memory and set seeds for reproducibility
        torch.cuda.empty_cache()
        os.environ['PYTHONHASHSEED'] = str(self.config['misc']['seed'])
        random.seed(self.config['misc']['seed'])
        np.random.seed(self.config['misc']['seed'])
        torch.manual_seed(self.config['misc']['seed'])
        torch.cuda.manual_seed(self.config['misc']['seed'])
        
        # Load model and criterion (loss function)
        self.model = load(self.config['model']['name']).to(self.device)
        self.criterion = load(
            self.config['training']['criterion']['name'], 
            **self.config['training']['criterion'].get('params', {})
        )
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['optimizer']['lr'], 
            weight_decay=self.config['training']['optimizer']['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            factor=self.config['training']['scheduler']['factor'], 
            patience=self.config['training']['scheduler']['patience']
        )
        
        # Setup dataloaders
        self.dataset_train = load(self.config['dataset']['name'], mode='train') # TODO: add transforms
        self.dataset_val = load(self.config['dataset']['name'], mode='val')
        self.dataset_test = load(self.config['dataset']['name'], mode='test')
        
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
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    try:
        mlflow.set_experiment(config['experiment_name'])
        run_datetime = datetime.datetime.now().strftime("%y%m%d:%H%M")
        with mlflow.start_run(run_name=run_datetime) as run:
            print(f'Starting training run: {run_datetime}')
            
            mlflow.log_params(config)
            mlflow.log_artifact(__file__)
            for folder in ['models', 'criterions', 'metrics', 'data', 'utils']:
                for file in os.listdir(folder):
                    mlflow.log_artifact(os.path.join(folder, file), artifact_path=folder)
            
            trainer = Trainer(config)
            trainer.train()
            
            tester = Tester()
            tester.test()
        
    except KeyboardInterrupt:
        print('Training interrupted')
    finally:
        if mlflow.active_run(): mlflow.end_run()
        print('MLflow run cleaned up')


if __name__ == "__main__":
    main()