# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import helpers, visualization

import mlflow

from data.transforms import build_transforms

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)

# %%
# Settings
run = 'debug/260309:1146/train'
task_mode = 'disparity'

# %%
# Load model
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model_run_id = helpers.get_model_run_id(run)
model = mlflow.pytorch.load_model(f'runs:/{model_run_id}/best_model_{task_mode}').to(device)

# %%
# Load data
config_name = run.split('/')[0]

with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
with open(os.path.join('configs', config_name + '.yaml'), 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)

dataset_class = helpers.load(config['data']['dataset'])

test_transforms = build_transforms(config, mode='test')
dataset_test = dataset_class(
    mode='test',
    config=config,
    transforms=test_transforms,
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=config['general']['num_workers'],
    pin_memory=config['general']['pin_memory'],
    persistent_workers=False
)
helpers.check_dataleakage('test', dataset_test)

# %%
# Visualize
index = 500

sample = dataset_test[index]
# Segmentation Teacher
# outputs = model(sample['image'].unsqueeze(0).to(device), None)
# upsampled_logits = F.interpolate(
#     outputs.logits,
#     size=sample['segmentation'].shape[-2:],
#     mode='bilinear',
#     align_corners=False
# )

outputs = model(sample['image'].unsqueeze(0).to(device), sample['right_image'].unsqueeze(0).to(device))

sample_targets = {}
if 'segmentation' in sample: sample_targets['segmentation'] = sample['segmentation']
if 'disparity' in sample: sample_targets['disparity'] = sample['disparity']

sample_outputs = {}
# sample_outputs['segmentation'] = upsampled_logits.cpu().detach().squeeze()
sample_outputs['disparity'] = outputs['disparity'].cpu().detach().squeeze()

sample_image = sample['image'].cpu().detach()

figure = visualization.get_multitask_visuals(
    image=sample_image,
    targets=sample_targets,
    outputs=sample_outputs,
    num_of_segmentation_classes=2,
    epoch='test',
    index=index,
    max_disparity=512
)

# %%