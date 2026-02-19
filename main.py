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
from utils.setup import setup_environment
from utils.helpers import load, log_vram

from utils.logger import setup_logging, CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))
logging.getLogger('mlflow.utils.environment').setLevel(logging.ERROR)


# * TASKS
# TODO: Integrate FoundationStereo Class as a teacher for disparity KD
# TODO: Extract cost volume mapping for disparity KD

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
        all_train_subsets = dataset_class(mode='train').get_all_subset_names()
        logger.info(f'Found {len(all_train_subsets)} training subsets: {all_train_subsets}')
        
        with mlflow.start_run(run_name=run_datetime) as run:
            helpers.mlflow_log_misc(log_filepath)
            tags = {}
            tags['parent_name'] = experiment_name
            tags['run_type'] = 'root'
            helpers.mlflow_log_run(config, tags=tags)
            mlflow.set_tag('mlflow.note.content', config['description'])

            best_model_run_id = ''
            
            if config['data']['cross_validation']:
                logger.header('Mode: Cross-Validation Training')
                with mlflow.start_run(run_name=f'{run.info.run_name}/train', nested=True) as train_run:
                    tags['parent_name'] = run.info.run_name
                    tags['run_type'] = 'train'
                    tags['run_mode'] = 'cross_validation'
                    helpers.mlflow_log_run(config, tags=tags)
                    mlflow.set_tag('mlflow.note.content', config['description'])
                    
                    fold_val_metrics_summary = {}
                    best_fold_idx = -1
                    best_fold_loss = float('inf')
                    best_fold_val_epoch_metrics = {}
                    
                    for i, val_subset in enumerate(all_train_subsets):
                        with mlflow.start_run(run_name=f'{run.info.run_name}/fold_{i+1}', nested=True) as fold_run:
                            logger.header(f'Starting Fold {i+1}/{len(all_train_subsets)} - Validation Subset: {val_subset}')
                            tags['fold'] = str(i + 1)
                            tags['val_subset'] = val_subset
                            helpers.mlflow_log_run(config, tags=tags)
                            mlflow.set_tag('mlflow.note.content', config['description'])
                            
                            train_subsets = [s for s in all_train_subsets if s != val_subset]
                            trainer = Trainer(copy.deepcopy(config), train_subsets=train_subsets, val_subsets=[val_subset])
                            best_val_epoch_metrics = trainer.train()
                            
                            if best_val_epoch_metrics['optimization/validation/loss/auto_weighted_sum'] < best_fold_loss:
                                best_fold_idx = i + 1
                                best_fold_loss = best_val_epoch_metrics['optimization/validation/loss/auto_weighted_sum']
                                best_fold_val_epoch_metrics = best_val_epoch_metrics
                                best_model_run_id = fold_run.info.run_id
                            
                            for metric_name, metric_value in best_val_epoch_metrics.items():
                                if not metric_name.rsplit('/', 1)[-1][0].isdigit():
                                    if metric_name not in fold_val_metrics_summary:
                                        fold_val_metrics_summary[metric_name] = {}
                                    fold_val_metrics_summary[metric_name][i+1] = metric_value
                                    
                            log_vram(f'Fold {i+1}')
                    
                    logger.header('Cross-Validation Summary')
                    logger.info(f'Best Fold: {best_fold_idx} with Loss: {best_fold_loss:.4f}')
                    mlflow.log_metric('cross_validation/best_fold/index', best_fold_idx, run_id=train_run.info.run_id)
                    mlflow.log_metric('cross_validation/best_fold/auto_weighted_sum', best_fold_loss, run_id=train_run.info.run_id)
                    
                    for metric_name, metric_values in fold_val_metrics_summary.items():
                        values = list(metric_values.values())
        
                        mean_metric = float(np.mean(values))
                        std_metric = float(np.std(values))
        
                        logger.subheader(f'Metric: {metric_name}')
                        logger.info(f'Mean={mean_metric:.4f}, Std={std_metric:.4f}')
                        mlflow.log_metric(f'cross_validation/{metric_name}/mean', mean_metric, run_id=train_run.info.run_id)
                        mlflow.log_metric(f'cross_validation/{metric_name}/std', std_metric, run_id=train_run.info.run_id)
        
                        for fold_idx, metric_value in metric_values.items():
                            logger.info(f'Fold {fold_idx} ({all_train_subsets[fold_idx-1]}): {metric_value:.4f}')
                            mlflow.log_metric(f'cross_validation/{metric_name}/folds', metric_value, step=fold_idx)
                            
            else:
                logger.header('Mode: Full Training')
                with mlflow.start_run(run_name=f'{run.info.run_name}/train', nested=True) as train_run:
                    tags['parent_name'] = run.info.run_name
                    tags['run_type'] = 'train'
                    tags['run_mode'] = 'full_training'
                    helpers.mlflow_log_run(config, tags=tags)
                    mlflow.set_tag('mlflow.note.content', config['description'])
                    
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
                mlflow.set_tag('mlflow.note.content', config['description'])
                
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