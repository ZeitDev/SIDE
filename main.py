import os
import yaml
import logging
import argparse
import datetime
import numpy as np
from typing import cast

import mlflow

from processors.trainer import Trainer
from processors.tester import Tester

from utils import helpers
from utils.helpers import load

from utils.logger import setup_logging, CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)

# * TASKS
# TODO: Implement MAE metric for disparity task
# ! Disparity task not tested yet
# ! Knowledge Distillation not tested yet

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
        
        with mlflow.start_run(run_name=run_datetime) as run:
            helpers.mlflow_log_run(config, log_filepath)

            best_model_run_id = ''
            
            if config['data']['cross_validation']:
                logger.header('Mode: Cross-Validation Training')
                with mlflow.start_run(run_name='train', nested=True) as train_run:
                    fold_val_metrics_summary = {}
                    best_fold = -1
                    best_fold_loss = float('inf')
                    
                    for i, val_subset in enumerate(all_train_subsets):
                        with mlflow.start_run(run_name=f'fold_{i+1}', nested=True) as fold_run:
                            logger.header(f'Starting Fold {i+1}/{len(all_train_subsets)} - Validation Subset: {val_subset}')
                            mlflow.log_param('validation_subset', val_subset)
                            
                            train_subsets = [s for s in all_train_subsets if s != val_subset]
                            trainer = Trainer(config, train_subsets=train_subsets, val_subsets=[val_subset])
                            best_val_epoch_metrics = trainer.train()
                            
                            if best_val_epoch_metrics['optimization/loss_validation'] < best_fold_loss:
                                best_fold = i + 1
                                best_fold_loss = best_val_epoch_metrics['optimization/loss_validation']
                                best_model_run_id = fold_run.info.run_id
                            
                            for metric_name, metric_value in best_val_epoch_metrics.items():
                                if any(m in metric_name for m in ['mIoU', 'mDICE', 'mMAE']):
                                    fold_val_metrics_summary.setdefault(metric_name, []).append(metric_value)
                    
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
                logger.header(f'Mode: Full Training')
                with mlflow.start_run(run_name='train', nested=True) as train_run:
                    best_model_run_id = train_run.info.run_id
                    trainer = Trainer(config, train_subsets=all_train_subsets)
                    trainer.train_without_validation()
            
            logger.header('Mode: Testing Best Model')
            with mlflow.start_run(run_name='test', nested=True) as test_run:
                tester = Tester(config, run_id=best_model_run_id)
                test_metrics = tester.test()
                mlflow.log_metrics(test_metrics)
                
        
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