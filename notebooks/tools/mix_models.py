# %% Imports
# Imports
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml

import mlflow
import mlflow.artifacts
from mlflow.tracking import MlflowClient

from utils import helpers

import torch
import mlflow.pytorch

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)

# %% Settings
# Settings
state_path = 'wMT-KD/260328:1209/train'

show_n_images = None # None for all images

# %% Load mlflow data
# Load mlflow data
state_path_parts = state_path.split('/')
experiment = state_path_parts[0]
run_path = '/'.join(state_path_parts[1:])

#mlflow.set_tracking_uri('../mlruns')
mlflow_experiment = mlflow.get_experiment_by_name(experiment)
mlflow_run = mlflow.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name = '{state_path_parts[1]}'").iloc[0] 

base_config_filepath = mlflow.artifacts.download_artifacts(run_id=mlflow_run.run_id, artifact_path='configs/base.yaml', dst_path='../.temp')
experiment_config_filepath = mlflow.artifacts.download_artifacts(run_id=mlflow_run.run_id, artifact_path=f'configs/{experiment}.yaml', dst_path='../.temp')

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

# %% Load Models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading best_segmentation and best_disparity...")
model_seg = mlflow.pytorch.load_model(f"runs:/{model_run_id}/best_model_segmentation", map_location='cpu')
model_disp = mlflow.pytorch.load_model(f"runs:/{model_run_id}/best_model_disparity", map_location='cpu')

# %% Mix Models
print("Mixing models with weight averaging...")
alpha = 0.1  # Weight averaging parameter: 0.5 means equal contribution

seg_state = model_seg.state_dict()
disp_state = model_disp.state_dict()

mixed_state = {}
for name, param in seg_state.items():
    if name in disp_state:
        mixed_param = alpha * param + (1 - alpha) * disp_state[name]
        mixed_state[name] = mixed_param
    else:
        mixed_state[name] = param

# Create mixed model using one of the existing models architecture
model_mixed = model_seg
model_mixed.load_state_dict(mixed_state)
model_mixed.to(device)
model_mixed.eval()
print("Mixed model created successfully!")

# %% Test Mixed Model
from processors.tester import Tester
from tqdm import tqdm

tester = Tester(config, model_run_id)

tester.model = model_mixed

# Reset metrics
for task_metrics in tester.metrics.values():
    for metric in task_metrics.values():
        metric.reset()

print("Starting custom testing loop for mixed model...")
batch_tqdm = tqdm(tester.dataloader_test, desc='Testing Mixed', position=0, leave=True)
with torch.no_grad():
    for data in batch_tqdm:
        left_images = data['image'].to(device)
        right_images = data['right_image'].to(device) if 'right_image' in data else None
        targets = {task: data[task].to(device) for task in tester.tasks}
        
        outputs = tester.model(left_images, right_images)
        
        if 'segmentation' in outputs and 'segmentation' in tester.metrics:
            for metric in tester.metrics['segmentation'].values():
                metric.update(outputs['segmentation'], targets['segmentation'])
                
        if 'disparity' in outputs and 'disparity' in tester.metrics:
            baseline = data['baseline'].to(device)
            focal_length = data['focal_length'].to(device)
            for metric in tester.metrics['disparity'].values():
                metric.update(outputs['disparity'], targets['disparity'], baseline, focal_length)

test_metrics = tester._compute_metrics(mode='testing')

print("\Mixed Model Test Results:")
for metric_key, metric_value in test_metrics.items():
    print(f"{metric_key}: {metric_value:.4f}")

# %%
