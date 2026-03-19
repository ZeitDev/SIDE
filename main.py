import os
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

# ! Gradient clipping expects normalized input ranges, if disparity is changed to raw pixels we need to accomodate
# * TASKS
# TODO: Disparity in pixel vs normalized range, what about multi task balancing?
# TODO: Wait for comparison run without weighting, if it is successful consider DWA weighting instead of Kendall, no not DWA, try DTP, before that compare different methods
# TODO: FIX TEST METRIC CASES

def main():
    try:
        setup_environment()
        parser = argparse.ArgumentParser(description='SIDE Training and Testing')
        parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
        args = parser.parse_args()

        with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
        with open(args.config, 'r') as f: experiment_config = yaml.safe_load(f)
        config = helpers.deep_merge(experiment_config, base_config)
        
        if not os.path.exists('.temp'): os.makedirs('.temp')
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
            helpers.mlflow_log_run(config, tags=tags)

            logger.header('Mode: Val Training' if config['data']['validation'] else 'Mode: Full Training')
            with mlflow.start_run(run_name=f'{run.info.run_name}/train', nested=True) as train_run:
                tags['parent_name'] = run.info.run_name
                tags['run_type'] = 'train'
                tags['run_mode'] = 'validation' if config['data']['validation'] else 'full'
                tags['description'] = config['description']
                helpers.mlflow_log_run(config, tags=tags)
                    
                trainer = Trainer(config)
                if config['data']['validation']: trainer.train()
                else: trainer.full_train()

            logger.header('Mode: Testing')
            with mlflow.start_run(run_name=f'{run.info.run_name}/test', nested=True) as test_run:
                tags['parent_name'] = run.info.run_name
                tags['run_type'] = 'test'
                tags['description'] = config['description']
                helpers.mlflow_log_run(config, tags=tags)
                
                tester = Tester(config, run_id=train_run.info.run_id)
                tester.test()
        
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