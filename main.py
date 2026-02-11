import os
import yaml
import copy
import logging
import argparse
import datetime
import numpy as np
from typing import cast, List, Dict, Any

import mlflow
from mlflow.entities import Run

from processors.tester import Tester
from processors.trainer import Trainer

from utils import helpers
from setup import setup_environment
from utils.helpers import load, log_vram

from utils.logger import setup_logging, CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))
logging.getLogger('mlflow.utils.environment').setLevel(logging.ERROR)


# * TASKS
# TODO: Disparity Overlay
# TODO: Test Knowledge Distillation end-to-end

def _load_configuration() -> tuple[Dict[str, Any], argparse.Namespace]:
    parser = argparse.ArgumentParser(description='SIDE Training and Testing')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()

    with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
    with open(args.config, 'r') as f: experiment_config = yaml.safe_load(f)
    config = helpers.deep_merge(experiment_config, base_config)
    return config, args

def _run_single_fold(config: Dict[str, Any], parent_run: Run, fold_idx: int, val_subset: str, all_train_subsets: List[str], tags: Dict[str, str]) -> tuple[str, float, Dict[str, float]]:
    with mlflow.start_run(run_name=f'{parent_run.info.run_name}/fold_{fold_idx+1}', nested=True) as fold_run:
        logger.header(f'Starting Fold {fold_idx+1}/{len(all_train_subsets)} - Validation Subset: {val_subset}')
        tags['fold'] = str(fold_idx + 1)
        tags['val_subset'] = val_subset
        helpers.mlflow_log_run(config, tags=tags)
        mlflow.set_tag('mlflow.note.content', config['description'])
        
        train_subsets = [s for s in all_train_subsets if s != val_subset]
        trainer = Trainer(copy.deepcopy(config), train_subsets=train_subsets, val_subsets=[val_subset])
        best_val_epoch_metrics = trainer.train()
        
        metric_key = 'optimization/validation/loss/auto_weighted_sum'
        fold_loss = best_val_epoch_metrics[metric_key]
        
        log_vram(f'Fold {fold_idx+1}')
        return fold_run.info.run_id, fold_loss, best_val_epoch_metrics

def _log_cross_validation_summary(train_run_id: str, best_fold: int, best_fold_loss: float, fold_val_metrics_summary: Dict[str, List[float]], all_train_subsets: List[str]):
    logger.header('Cross-Validation Summary')
    logger.info(f'Best Fold: {best_fold} with Loss: {best_fold_loss:.4f}')
    mlflow.log_metric('cross_validation/best_fold', best_fold, run_id=train_run_id)
    mlflow.log_metric('cross_validation/best_fold_loss', best_fold_loss, run_id=train_run_id)
    
    for metric_name, metric_values in fold_val_metrics_summary.items():
        mean_metric = float(np.mean(metric_values))
        std_metric = float(np.std(metric_values))
        
        logger.subheader(f'Metric: {metric_name}')
        logger.info(f'Mean={mean_metric:.4f}, Std={std_metric:.4f}')
        mlflow.log_metric(f'cross_validation/{metric_name}/mean', mean_metric, run_id=train_run_id)
        mlflow.log_metric(f'cross_validation/{metric_name}/std', std_metric, run_id=train_run_id)
        
        for fold_idx, v in enumerate(metric_values):
            logger.info(f'Fold {fold_idx+1} ({all_train_subsets[fold_idx]}): {v:.4f}')
            mlflow.log_metric(f'cross_validation/{metric_name}/fold_{fold_idx+1}', v)

def _run_cross_validation(config: Dict[str, Any], parent_run: Run, all_train_subsets: List[str], tags: Dict[str, str]) -> str:
    logger.header('Mode: Cross-Validation Training')
    best_model_run_id = ''
    
    with mlflow.start_run(run_name=f'{parent_run.info.run_name}/train', nested=True) as train_run:
        tags['parent_name'] = parent_run.info.run_name
        tags['run_type'] = 'train'
        tags['run_mode'] = 'cross_validation'
        helpers.mlflow_log_run(config, tags=tags)
        mlflow.set_tag('mlflow.note.content', config['description'])
        
        fold_val_metrics_summary = {}
        best_fold = -1
        best_fold_loss = float('inf')
        
        for i, val_subset in enumerate(all_train_subsets):
            fold_run_id, fold_loss, best_val_epoch_metrics = _run_single_fold(config, parent_run, i, val_subset, all_train_subsets, tags)

            if fold_loss < best_fold_loss:
                best_fold = i + 1
                best_fold_loss = fold_loss
                best_model_run_id = fold_run_id
            
            for metric_name, metric_value in best_val_epoch_metrics.items():
                if not metric_name.rsplit('/', 1)[-1][0].isdigit():
                    fold_val_metrics_summary.setdefault(metric_name, []).append(metric_value)
                        
        _log_cross_validation_summary(train_run.info.run_id, best_fold, best_fold_loss, fold_val_metrics_summary, all_train_subsets)

    return best_model_run_id

def _run_full_training(config: Dict[str, Any], parent_run: Run, all_train_subsets: List[str], tags: Dict[str, str]) -> str:
    logger.header('Mode: Full Training')
    with mlflow.start_run(run_name=f'{parent_run.info.run_name}/train', nested=True) as train_run:
        tags['parent_name'] = parent_run.info.run_name
        tags['run_type'] = 'train'
        tags['run_mode'] = 'full_training'
        helpers.mlflow_log_run(config, tags=tags)
        mlflow.set_tag('mlflow.note.content', config['description'])
        
        log_vram('Full Training Start')
        trainer = Trainer(config, train_subsets=all_train_subsets)
        trainer.train_without_validation()
        log_vram('Full Training End')
        
        return train_run.info.run_id

def _run_testing(config: Dict[str, Any], parent_run: Run, best_model_run_id: str, tags: Dict[str, str]):
    logger.header('Mode: Testing Best Model')
    with mlflow.start_run(run_name=f'{parent_run.info.run_name}/test', nested=True) as test_run:
        tags['parent_name'] = parent_run.info.run_name
        tags['run_type'] = 'test'
        helpers.mlflow_log_run(config, tags=tags)
        mlflow.set_tag('mlflow.note.content', config['description'])
        
        log_vram('Testing Start')
        tester = Tester(config, run_id=best_model_run_id)
        test_metrics = tester.test()
        mlflow.log_metrics(test_metrics)
        log_vram('Testing End')

def main():
    setup_environment()
    config, args = _load_configuration()

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
            tags['parent_name'] = experiment_name
            tags['run_type'] = 'root'
            helpers.mlflow_log_run(config, tags=tags)
            mlflow.set_tag('mlflow.note.content', config['description'])

            if config['data']['cross_validation']:
                best_model_run_id = _run_cross_validation(config, run, all_train_subsets, tags)
            else:
                best_model_run_id = _run_full_training(config, run, all_train_subsets, tags)
            
            _run_testing(config, run, best_model_run_id, tags)
        
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