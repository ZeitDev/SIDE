import os
import yaml
import copy
import logging
import argparse
import datetime
import numpy as np
from typing import cast

import mlflow

from processors.tester import Tester
from processors.trainer import Trainer

from utils import helpers
from setup import setup_environment
from utils.helpers import load, log_vram

from utils.logger import setup_logging, CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))
logging.getLogger('mlflow.utils.environment').setLevel(logging.ERROR)


# * TASKS
# TODO: Test Segmentation only vs. Disparity only vs. Multi-Task with MLflow logging
# TODO: Test Knowledge Distillation end-to-end

def main():
    setup_environment()
    parser = argparse.ArgumentParser(description='SIDE Training and Testing')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()

    with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
    with open(args.config, 'r') as f: experiment_config = yaml.safe_load(f)
    config = helpers.deep_merge(experiment_config, base_config)
    
    if not os.path.exists('cache'): os.makedirs('cache')
    if not os.path.exists('logs'): os.makedirs('logs')

    try:
        experiment_name = os.path.splitext(os.path.basename(args.config))[0]
        mlflow.set_experiment(experiment_name)
        run_datetime = datetime.datetime.now().strftime("%y%m%d:%H%M")
        
        log_filepath = os.path.join('logs', f'{run_datetime}_{experiment_name}.log')
        setup_logging(log_filepath=log_filepath, vram_only=config['logging']['vram'])
        
        dataset_class = load(config['data']['dataset'])
        all_train_subsets = dataset_class(mode='train').get_all_subset_names()
        logger.info(f'Found {len(all_train_subsets)} training subsets: {all_train_subsets}')
        
        with mlflow.start_run(run_name=run_datetime) as run:
            helpers.mlflow_log_misc(log_filepath)
            tags = {}
            tags['description'] = config['description']
            tags['parent_name'] = experiment_name
            tags['run_type'] = 'main'
            helpers.mlflow_log_run(config, tags=tags)

            best_model_run_id = ''
            
            if config['data']['cross_validation']:
                logger.header('Mode: Cross-Validation Training')
                with mlflow.start_run(run_name=f'{run.info.run_name}/train', nested=True) as train_run:
                    tags['parent_name'] = run.info.run_name
                    tags['run_type'] = 'train'
                    tags['run_mode'] = 'cross_validation'
                    helpers.mlflow_log_run(config, tags=tags)
                    
                    fold_val_metrics_summary = {}
                    best_fold = -1
                    best_fold_loss = float('inf')
                    
                    for i, val_subset in enumerate(all_train_subsets):
                        with mlflow.start_run(run_name=f'{run.info.run_name}/fold_{i+1}', nested=True) as fold_run:
                            logger.header(f'Starting Fold {i+1}/{len(all_train_subsets)} - Validation Subset: {val_subset}')
                            tags['fold'] = str(i + 1)
                            tags['val_subset'] = val_subset
                            helpers.mlflow_log_run(config, tags=tags)
                            
                            train_subsets = [s for s in all_train_subsets if s != val_subset]
                            trainer = Trainer(copy.deepcopy(config), train_subsets=train_subsets, val_subsets=[val_subset])
                            best_val_epoch_metrics = trainer.train()
                            
                            if best_val_epoch_metrics['optimization/validation/loss/weighted'] < best_fold_loss:
                                best_fold = i + 1
                                best_fold_loss = best_val_epoch_metrics['optimization/validation/loss/weighted']
                                best_model_run_id = fold_run.info.run_id
                            
                            for metric_name, metric_value in best_val_epoch_metrics.items():
                                if not metric_name.rsplit('/', 1)[-1][0].isdigit():
                                    fold_val_metrics_summary.setdefault(metric_name, []).append(metric_value)
                                    
                            log_vram(f'Fold {i+1}')
                    
                    logger.header('Cross-Validation Summary')
                    logger.info(f'Best Fold: {best_fold} with Loss: {best_fold_loss:.4f}')
                    mlflow.log_metric('cross_validation/best_fold', best_fold, run_id=train_run.info.run_id)
                    mlflow.log_metric('cross_validation/best_fold_loss', best_fold_loss, run_id=train_run.info.run_id)
                    for metric_name, metric_values in fold_val_metrics_summary.items():
                        mean_metric = float(np.mean(metric_values))
                        std_metric = float(np.std(metric_values))
                        
                        logger.subheader(f'Metric: {metric_name}')
                        logger.info(f'Mean={mean_metric:.4f}, Std={std_metric:.4f}')
                        mlflow.log_metric(f'cross_validation/{metric_name}/mean', mean_metric, run_id=train_run.info.run_id)
                        mlflow.log_metric(f'cross_validation/{metric_name}/std', std_metric, run_id=train_run.info.run_id)
                        
                        for fold_idx, v in enumerate(metric_values):
                            logger.info(f'Fold {fold_idx+1} ({all_train_subsets[fold_idx]}): {v:.4f}')
                            mlflow.log_metric(f'cross_validation/{metric_name}/fold_{fold_idx+1}', v)
                            
            else:
                logger.header('Mode: Full Training')
                with mlflow.start_run(run_name=f'{run.info.run_name}/train', nested=True) as train_run:
                    tags['parent_name'] = run.info.run_name
                    tags['run_type'] = 'train'
                    tags['run_mode'] = 'full_training'
                    helpers.mlflow_log_run(config, tags=tags)
                    log_vram('Full Training Start')
                    
                    best_model_run_id = train_run.info.run_id
                    trainer = Trainer(config, train_subsets=all_train_subsets)
                    trainer.train_without_validation()
                    log_vram('Full Training End')
            
            logger.header('Mode: Testing Best Model')
            with mlflow.start_run(run_name=f'{run.info.run_name}/test', nested=True) as test_run:
                tags['parent_name'] = run.info.run_name
                tags['run_type'] = 'test'
                helpers.mlflow_log_run(config, tags=tags)
                log_vram('Testing Start')
                
                tester = Tester(config, run_id=best_model_run_id)
                test_metrics = tester.test()
                mlflow.log_metrics(test_metrics)
                log_vram('Testing End')
        
    except KeyboardInterrupt:
        logger.warning('Training interrupted by user')
    except Exception as error:
        logger.error(error)
        raise error
    finally:
        if mlflow.active_run(): mlflow.end_run()
        log_vram('End')
        logger.single('MLflow run cleaned up')
            
if __name__ == "__main__":
    main()