# %% Imports
# Imports
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import yaml

import mlflow
import mlflow.artifacts

from processors.tester import Tester

from utils import helpers

from setup import setup_environment
setup_environment()

# %% Settings
# Settings
experiment = 'overfit'
run = '251121:1437'
model_path = 'train/fold_2'
show_n_images = 10 # None for all images

# %% Load mlflow data
# Load mlflow data
mlflow.set_tracking_uri('../mlruns')
mlflow_experiment = mlflow.get_experiment_by_name(experiment) 
mlflow_run = mlflow.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name = '{run}'").iloc[0] 
model_path = f'{run}/{model_path}'

base_config_filepath = mlflow.artifacts.download_artifacts(run_id=mlflow_run.run_id, artifact_path='configs/base.yaml', dst_path='../cache')
experiment_config_filepath = mlflow.artifacts.download_artifacts(run_id=mlflow_run.run_id, artifact_path=f'configs/{experiment}.yaml', dst_path='../cache')

with open(base_config_filepath, 'r') as f: base_config = yaml.safe_load(f)
with open(experiment_config_filepath, 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)
config['logging']['notebook_mode'] = True

# %% Find model run id
# Find model run id
run_parts = model_path.split('/')
path_depth = len(run_parts)
model_run_id = ''

run_id = mlflow.search_runs(
    experiment_ids=[mlflow_experiment.experiment_id], 
    filter_string=f"tags.mlflow.runName = '{run}'",
    order_by=["attributes.start_time DESC"],
    max_results=1
).iloc[0].run_id 
if path_depth == 2:
    sub_run_name = f'{run_parts[0]}/{run_parts[1]}'
    model_run_id = mlflow.search_runs(
        experiment_ids=[mlflow_experiment.experiment_id], 
        filter_string=f"tags.mlflow.runName = '{sub_run_name}' and tags.mlflow.parentRunId = '{run_id}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    ).iloc[0].run_id 
    
elif path_depth == 3:
    sub_run_name = f'{run_parts[0]}/{run_parts[1]}'
    sub_run_id = mlflow.search_runs(
        experiment_ids=[mlflow_experiment.experiment_id], 
        filter_string=f"tags.mlflow.runName = '{sub_run_name}' and tags.mlflow.parentRunId = '{run_id}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    ).iloc[0].run_id 

    subsub_run_name = f'{run_parts[0]}/{run_parts[2]}'
    model_run_id = mlflow.search_runs(
        experiment_ids=[mlflow_experiment.experiment_id], 
        filter_string=f"tags.mlflow.runName = '{subsub_run_name}' and tags.mlflow.parentRunId = '{sub_run_id}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    ).iloc[0].run_id 

# %%
print(f'Testing Model \nExperiment: {experiment} \nRun: {run} \nModel Path: {model_path} \nID: {model_run_id}\n')
tester = Tester(config, run_id=model_run_id)

if show_n_images: config['logging']['n_validation_images'] = show_n_images
else: config['logging']['n_validation_images'] = len(tester.dataloader_test.dataset) 

test_metrics = tester.test()
[print(f'{k}: {v}') for k, v in test_metrics.items()];
# %%