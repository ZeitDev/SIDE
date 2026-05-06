import os
import sys
import yaml
import logging
import argparse
import datetime
from typing import cast

import mlflow

from processors.tester import Tester
from processors.trainer import Trainer

from utils import helpers
from utils.setup import setup_environment
from utils.helpers import load

from utils.logger import setup_logging, CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))
logging.getLogger('mlflow.utils.environment').setLevel(logging.ERROR)

def main():
    try:
        parser = argparse.ArgumentParser(description='SIDE Training and Testing')
        parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
        parser.add_argument('--cuda_device', type=int, default=1, help='CUDA device to use.')
        args = parser.parse_args()
        setup_environment(skip_cuda=False, cuda_device=args.cuda_device, delete_temp=True)

        with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
        with open(args.config, 'r') as f: experiment_config = yaml.safe_load(f)
        config = helpers.deep_merge(experiment_config, base_config)
        
        if not os.path.exists('logs'): os.makedirs('logs')

        experiment_name = os.path.splitext(os.path.basename(args.config))[0]
        mlflow.set_experiment(experiment_name)
        run_datetime = datetime.datetime.now().strftime("%y%m%d:%H%M")
        
        log_filepath = os.path.join('logs', f'{run_datetime}_{experiment_name}.log')
        setup_logging(log_filepath=log_filepath, vram_only=config['logging']['vram'])
        
        dataset_class = load(config['data']['dataset'])
        
        with mlflow.start_run(run_name=run_datetime) as run:
            helpers.mlflow_log_misc(log_filepath)
            tags = {}
            tags['parent_name'] = experiment_name
            tags['run_type'] = 'root'
            tags['description'] = config['description']
            mlflow.set_tag('mlflow.note.content', config['description'])
            helpers.mlflow_log_run(config, tags=tags)

            logger.header('Mode: Val Training' if config['training']['validation'] else 'Mode: Full Training')
            with mlflow.start_run(run_name=f'{run.info.run_name}/train', nested=True) as train_run:
                tags['parent_name'] = run.info.run_name
                tags['run_type'] = 'train'
                tags['run_mode'] = 'validation' if config['training']['validation'] else 'full'
                tags['description'] = config['description']
                helpers.mlflow_log_run(config, tags=tags)
                    
                trainer = Trainer(config)
                if config['training']['validation']: trainer.train()
                else: trainer.full_train()

            logger.header('Mode: Testing')
            with mlflow.start_run(run_name=f'{run.info.run_name}/test', nested=True) as test_run:
                tags['parent_name'] = run.info.run_name
                tags['run_type'] = 'test'
                tags['description'] = config['description']
                helpers.mlflow_log_run(config, tags=tags)
                
                tester = Tester(config, run_id=train_run.info.run_id)
                test_metrics = tester.test()
                
                logger.subheader('Test Results')
                for task_mode, _test_metrics in test_metrics.items():
                    for metric_key, metric_value in _test_metrics.items(): 
                        mlflow.log_metric(f'best_{task_mode}/{metric_key}', metric_value)

                for task_mode, _test_metrics in test_metrics.items():
                    logger.subheader(f'Test Results for {task_mode}')
                    for metric_key, metric_value in _test_metrics.items():
                        logger.info(f'{metric_key}: {metric_value:.4f}')
        
    except KeyboardInterrupt:
        logger.warning('Training interrupted by user')
    except Exception as error:
        logger.error(error)
        raise error
    finally:
        if mlflow.active_run(): mlflow.end_run()
        logger.single('MLflow run cleaned up')
            
if __name__ == "__main__":
    main()
    sys.exit(0)