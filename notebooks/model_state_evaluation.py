# %% Imports
# Imports
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import yaml

import mlflow
import mlflow.artifacts
from mlflow.tracking import MlflowClient

from processors.tester import Tester

from utils import helpers

from setup import setup_environment
setup_environment()

# %% Settings
# Settings
state_path = 'debug/260212:1518/train' # '260206:1650/train/fold_1'

show_n_images = None # None for all images

# %% Load mlflow data
# Load mlflow data
state_path_parts = state_path.split('/')
experiment = state_path_parts[0]
run_path = '/'.join(state_path_parts[1:])

mlflow.set_tracking_uri('../mlruns')
mlflow_experiment = mlflow.get_experiment_by_name(experiment)
mlflow_run = mlflow.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name = '{state_path_parts[1]}'").iloc[0] 

base_config_filepath = mlflow.artifacts.download_artifacts(run_id=mlflow_run.run_id, artifact_path='configs/base.yaml', dst_path='../cache')
experiment_config_filepath = mlflow.artifacts.download_artifacts(run_id=mlflow_run.run_id, artifact_path=f'configs/{experiment}.yaml', dst_path='../cache')

with open(base_config_filepath, 'r') as f: base_config = yaml.safe_load(f)
with open(experiment_config_filepath, 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)
config['logging']['notebook_mode'] = True

# %%
model_run_id = mlflow.search_runs(
    experiment_ids=[mlflow_experiment.experiment_id], 
    filter_string=f"tags.mlflow.runName = '{run_path}'", 
    order_by=["attributes.start_time DESC"], 
    max_results=1
).iloc[0].run_id

# %%
model_run_metrics = mlflow.get_run(model_run_id).data.metrics
iou_metric_key = None
# Find the specific key for the folds
for key in model_run_metrics.keys():
    if 'auto_weighted_sum' in key and 'folds' in key:
        iou_metric_key = key
        break
    
client = MlflowClient()
model_run_metrics = client.get_metric_history(model_run_id, iou_metric_key)


# %%
print(f'Testing Model \nExperiment: {state_path} \nID: {model_run_id}\n')
tester = Tester(config, run_id=model_run_id)

if show_n_images: config['logging']['n_validation_images'] = show_n_images
else: config['logging']['n_validation_images'] = len(tester.dataloader_test.dataset) 

test_metrics = tester.test()
[print(f'{k}: {v}') for k, v in test_metrics.items()];
# %%