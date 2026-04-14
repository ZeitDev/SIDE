# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml

from torch.utils.data import DataLoader
from utils import helpers

import torch.nn.functional as F

from data.transforms import build_transforms
from metrics.segmentation import Dice
from metrics.disparity import AbsRel

from tqdm import tqdm

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)

# %%
with open(os.path.join('configs', 'base.yaml'), 'r') as f: config = yaml.safe_load(f)
config['training']['tasks']['segmentation']['enabled'] = True
config['training']['tasks']['segmentation']['distillation']['enabled'] = True
config['training']['tasks']['disparity']['enabled'] = True
config['training']['tasks']['disparity']['distillation']['enabled'] = True

dataset_class = helpers.load(config['data']['dataset'])

transforms = build_transforms(config, mode='test')
dataset = dataset_class(
    mode='val',
    config=config,
    transforms=transforms,
)
dataloader = DataLoader(
    dataset,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=config['general']['num_workers'],
    pin_memory=config['general']['pin_memory'],
    persistent_workers=False
)
helpers.check_dataleakage('val', dataset)

# %%
dice_metric = Dice(n_classes=config['data']['num_of_classes']['segmentation'], device='cpu')
absrel_metric = AbsRel(max_disparity=config['data']['max_disparity'], device='cpu')

tasks = ['segmentation', 'disparity']
for data in tqdm(dataloader):
    targets = {task: data[task] for task in tasks}
    baseline, focal_length = data['baseline'], data['focal_length']
    

    teacher_logits = {}
    teacher_logits['segmentation'] = data['teacher_segmentation'] #F.interpolate(data['teacher_segmentation'], size=targets['segmentation'].shape[2:], mode='bilinear', align_corners=False)
    teacher_logits['disparity'] = helpers.logits2disparity(data['teacher_disparity'], size=(256, 256)) # helpers.logits2disparity(data['teacher_disparity'], size=targets['disparity'].shape[2:])
    
    segmentation_targets = F.interpolate(targets['segmentation'], size=(256, 256), mode='nearest-exact')
    disparity_targets = F.interpolate(targets['disparity'], size=(256, 256), mode='nearest-exact')
    
    dice_metric.update(teacher_logits['segmentation'], segmentation_targets)
    absrel_metric.update(teacher_logits['disparity'], disparity_targets, baseline, focal_length)
    
dice_result = dice_metric.compute()
absrel_result = absrel_metric.compute()

# %%
print(dice_result)
print(absrel_result)

# %%
