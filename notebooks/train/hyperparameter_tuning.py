import argparse
import copy
import datetime
import logging
import math
import os
from typing import Any, Dict, cast

import gc
import mlflow
import optuna
import torch
import yaml
from optuna.samplers import RandomSampler, TPESampler

from processors.trainer import Trainer
from utils import helpers
from utils.logger import CustomLogger, setup_logging
from utils.setup import setup_environment

logger = cast(CustomLogger, logging.getLogger(__name__))
logging.getLogger('mlflow.utils.environment').setLevel(logging.ERROR)


OBJECTIVE_DIRECTIONS = {
	'segmentation_dice': 'maximize',
	'disparity_bad3': 'minimize',
	'combined_heuristic': 'minimize',
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Optuna hyperparameter search for SIDE')
	parser.add_argument('--config', type=str, required=True, help='Path to the experiment config file.')
	parser.add_argument('--trials', type=int, default=10, help='Number of Optuna trials.')
	parser.add_argument('--timeout', type=int, default=None, help='Optional study timeout in seconds.')
	parser.add_argument(
		'--objective',
		type=str,
		default='auto',
		choices=['auto', *OBJECTIVE_DIRECTIONS.keys()],
		help='Metric to optimize. Auto infers a sensible default from enabled tasks.',
	)
	parser.add_argument(
		'--study-name',
		type=str,
		default=None,
		help='Custom Optuna study name. Defaults to <config-name>_optuna.',
	)
	parser.add_argument(
		'--mlflow-experiment',
		type=str,
		default=None,
		help='Optional MLflow experiment name. Defaults to <config-name>_optuna.',
	)
	parser.add_argument(
		'--storage',
		type=str,
		default=None,
		help='Optuna storage URL. Defaults to a local sqlite database in ./optuna_studies/.',
	)
	parser.add_argument(
		'--sampler',
		type=str,
		default='tpe',
		choices=['tpe', 'random'],
		help='Optuna sampler to use.',
	)
	parser.add_argument('--seed', type=int, default=42, help='Seed for the Optuna sampler.')
	parser.add_argument(
		'--resume',
		action='store_true',
		help='Resume an existing study if the name and storage already exist.',
	)
	return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
	with open(os.path.join('configs', 'base.yaml'), 'r', encoding='utf-8') as file:
		base_config = yaml.safe_load(file)
	with open(config_path, 'r', encoding='utf-8') as file:
		experiment_config = yaml.safe_load(file)

	return helpers.deep_merge(experiment_config, base_config)


def set_nested_value(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
	current = config
	parts = dotted_key.split('.')
	for part in parts[:-1]:
		current = current.setdefault(part, {})
	current[parts[-1]] = value


def build_storage_url(study_name: str, storage: str | None) -> str:
	if storage:
		return storage

	os.makedirs('optuna_studies', exist_ok=True)
	db_path = os.path.join('optuna_studies', f'{study_name}.db')
	return f'sqlite:///{db_path}'


def infer_objective_name(config: Dict[str, Any], requested_objective: str) -> str:
	if requested_objective != 'auto':
		return requested_objective

	tasks = config['training']['tasks']
	has_segmentation = tasks['segmentation']['enabled']
	has_disparity = tasks['disparity']['enabled']

	if has_segmentation and has_disparity:
		return 'combined_heuristic'
	if has_segmentation:
		return 'segmentation_dice'
	if has_disparity:
		return 'disparity_bad3'

	raise ValueError('No training task is enabled. Optuna needs at least one active task.')


def suggest_hyperparameters(trial: optuna.Trial, config: Dict[str, Any]) -> Dict[str, Any]:
	suggested_values: Dict[str, Any] = {}

	# Start with optimizer settings because they are safe to vary across all current tasks.
	suggested_values['training.optimizer.params.lr'] = trial.suggest_float(
		'training.optimizer.params.lr', 1e-5, 3e-3, log=True
	)
	suggested_values['training.optimizer.params.weight_decay'] = trial.suggest_float(
		'training.optimizer.params.weight_decay', 1e-6, 1e-2, log=True
	)
	suggested_values['training.optimizer.encoder_lr_mod'] = trial.suggest_float(
		'training.optimizer.encoder_lr_mod', 0.05, 0.5, log=True
	)

	current_batch_size = int(config['data'].get('batch_size', 1))
	batch_size_choices = sorted({max(1, current_batch_size // 2), current_batch_size})
	suggested_values['data.batch_size'] = trial.suggest_categorical(
		'data.batch_size', batch_size_choices
	)

	scheduler_name = config['training']['scheduler']['name']
	if scheduler_name == 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts':
		suggested_values['training.scheduler.params.eta_min'] = trial.suggest_float(
			'training.scheduler.params.eta_min', 1e-8, 1e-5, log=True
		)
		suggested_values['training.scheduler.params.T_0'] = trial.suggest_categorical(
			'training.scheduler.params.T_0', [5, 10, 25, 50]
		)

	if config['training']['tasks']['disparity']['enabled']:
		suggested_values['training.tasks.disparity.criterion.params.beta'] = trial.suggest_categorical(
			'training.tasks.disparity.criterion.params.beta', [0.5, 1.0, 2.0]
		)

	for dotted_key, value in suggested_values.items():
		set_nested_value(config, dotted_key, value)

	return suggested_values


def extract_objective_value(metrics: Dict[str, float], objective_name: str) -> float:
	if objective_name == 'segmentation_dice':
		metric_key = 'best/segmentation/performance_validation_segmentation_DICE_score_instrument'
		return metrics[metric_key]

	if objective_name == 'disparity_bad3':
		metric_key = 'best/disparity/performance_validation_disparity_Bad3_rate'
		return metrics[metric_key]

	if objective_name == 'combined_heuristic':
		dice_key = 'best/combined/performance_validation_segmentation_DICE_score_instrument'
		bad3_key = 'best/combined/performance_validation_disparity_Bad3_rate'
		dice = metrics[dice_key]
		bad3 = metrics[bad3_key]
		return math.sqrt((1.0 - dice) ** 2 + bad3 ** 2)

	raise ValueError(f'Unsupported objective: {objective_name}')


def cleanup_after_trial() -> None:
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def build_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
	if name == 'random':
		return RandomSampler(seed=seed)
	return TPESampler(seed=seed)


def optimize_trial(
	trial: optuna.Trial,
	base_config: Dict[str, Any],
	objective_name: str,
	study_name: str,
) -> float:
	config = copy.deepcopy(base_config)
	suggested_values = suggest_hyperparameters(trial, config)
	config['logging']['n_validation_images'] = 0
	config['logging']['notebook_mode'] = False
	config['description'] = f'{config["description"]} | Optuna trial {trial.number}'

	run_name = f'trial_{trial.number:03d}'
	with mlflow.start_run(run_name=run_name, nested=True) as trial_run:
		tags = {
			'parent_name': study_name,
			'run_type': 'optuna_trial',
			'objective': objective_name,
		}
		helpers.mlflow_log_run(config, tags=tags)
		mlflow.set_tag('mlflow.note.content', config['description'])
		mlflow.set_tag('optuna.trial_number', str(trial.number))
		mlflow.set_tag('optuna.study_name', study_name)

		try:
			trainer = Trainer(config)
			trainer.train()

			metrics = mlflow.get_run(trial_run.info.run_id).data.metrics
			objective_value = extract_objective_value(metrics, objective_name)
			mlflow.log_metric('optuna/objective', objective_value)
			trial.set_user_attr('mlflow_run_id', trial_run.info.run_id)
			trial.set_user_attr('objective_name', objective_name)
			trial.set_user_attr('suggested_values', suggested_values)

			logger.info(
				'Trial %s finished with %s=%.6f',
				trial.number,
				objective_name,
				objective_value,
			)
			return objective_value
		except Exception:
			mlflow.set_tag('optuna.status', 'failed')
			raise
		finally:
			cleanup_after_trial()


def main() -> None:
	args = parse_args()

	setup_environment()

	if not os.path.exists('logs'):
		os.makedirs('logs')

	base_config = load_config(args.config)
	if not base_config['training']['validation']:
		raise ValueError('Optuna search needs validation enabled to produce a reliable objective.')

	config_name = os.path.splitext(os.path.basename(args.config))[0]
	study_name = args.study_name or f'{config_name}_optuna'
	mlflow_experiment = args.mlflow_experiment or study_name
	objective_name = infer_objective_name(base_config, args.objective)
	direction = OBJECTIVE_DIRECTIONS[objective_name]
	storage_url = build_storage_url(study_name, args.storage)

	run_datetime = datetime.datetime.now().strftime('%y%m%d:%H%M')
	log_filepath = os.path.join('logs', f'{run_datetime}_{study_name}.log')
	setup_logging(log_filepath=log_filepath, vram_only=base_config['logging']['vram'])

	mlflow.set_experiment(mlflow_experiment)

	logger.header('Starting Optuna Hyperparameter Search')
	logger.info('Study name: %s', study_name)
	logger.info('Objective: %s (%s)', objective_name, direction)
	logger.info('Storage: %s', storage_url)
	logger.info('Trials run sequentially because setup_environment() pins a single GPU.')

	sampler = build_sampler(args.sampler, args.seed)
	study = optuna.create_study(
		study_name=study_name,
		direction=direction,
		sampler=sampler,
		storage=storage_url,
		load_if_exists=args.resume,
	)

	try:
		with mlflow.start_run(run_name=run_datetime) as root_run:
			helpers.mlflow_log_misc(log_filepath)
			mlflow.log_artifact(__file__, artifact_path='scripts')
			helpers.mlflow_log_run(
				base_config,
				tags={
					'parent_name': config_name,
					'run_type': 'optuna_root',
					'objective': objective_name,
				},
			)
			mlflow.set_tag(
				'mlflow.note.content',
				f'Optuna study for {base_config["description"]}',
			)
			mlflow.set_tag('optuna.study_name', study_name)
			mlflow.set_tag('optuna.direction', direction)
			mlflow.log_param('optuna.trials_requested', args.trials)
			mlflow.log_param('optuna.storage', storage_url)
			mlflow.log_param('optuna.sampler', args.sampler)

			study.optimize(
				lambda trial: optimize_trial(
					trial=trial,
					base_config=base_config,
					objective_name=objective_name,
					study_name=study_name,
				),
				n_trials=args.trials,
				timeout=args.timeout,
				gc_after_trial=True,
			)

			mlflow.log_metric('optuna/best_value', study.best_value)
			mlflow.log_metric('optuna/best_trial_number', study.best_trial.number)
			for key, value in study.best_params.items():
				mlflow.log_param(f'optuna.best.{key}', value)

			logger.subheader('Best Trial')
			logger.info('Best trial number: %s', study.best_trial.number)
			logger.info('Best value: %.6f', study.best_value)
			logger.info('Best params: %s', study.best_params)
			logger.info('MLflow root run ID: %s', root_run.info.run_id)

	except KeyboardInterrupt:
		logger.warning('Optuna search interrupted by user')
	finally:
		if mlflow.active_run():
			mlflow.end_run()
		cleanup_after_trial()


if __name__ == '__main__':
	main()
