# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml

from torch.utils.data import DataLoader
from utils import helpers

import torch
import torch.nn.functional as F

from data.transforms import build_transforms
from metrics.segmentation import Dice
from metrics.disparity import EPE

from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)

# %%
with open(os.path.join('configs', 'base.yaml'), 'r') as f: config = yaml.safe_load(f)
config['training']['tasks']['segmentation']['enabled'] = True
config['training']['tasks']['segmentation']['distillation']['enabled'] = True
config['training']['tasks']['disparity']['enabled'] = True
config['training']['tasks']['disparity']['distillation']['enabled'] = True

data_config = config['data']
dataset_class = helpers.load(data_config['dataset'])

transforms = build_transforms(config, mode='test')
dataset = dataset_class(
    mode='val',
    config=config,
    transforms=transforms,
)
dataloader = DataLoader(
    dataset,
    batch_size=data_config['batch_size'],
    shuffle=False,
    num_workers=data_config['num_workers'],
    pin_memory=data_config['pin_memory'],
    persistent_workers=False
)
helpers.check_dataleakage('val', dataset)

# %%
dice_metric = Dice(n_classes=config['data']['num_of_classes']['segmentation'], device='cpu')
epe_metric = EPE(max_disparity=config['data']['max_disparity'], device='cpu')

tasks = ['segmentation', 'disparity']
for data in tqdm(dataloader):
    targets = {task: data[task] for task in tasks}
    baseline, focal_length = data['baseline'], data['focal_length']
    
    teacher_logits = {}
    teacher_logits['segmentation'] = F.interpolate(data['teacher_segmentation'], size=targets['segmentation'].shape[2:], mode='bilinear', align_corners=False)
    teacher_logits['disparity'] = helpers.logits2disparity(data['teacher_disparity'], size=targets['disparity'].shape[2:])
    
    dice_metric.update(teacher_logits['segmentation'], targets['segmentation'])
    epe_metric.update(teacher_logits['disparity'], targets['disparity'], baseline, focal_length)
    
    break
    
dice_result = dice_metric.compute()
epe_result = epe_metric.compute()

# %%
print(dice_result)
print(epe_result)

# %%
