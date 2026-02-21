# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))
from utils.setup import setup_environment

import numpy as np
import yaml
import torch
from omegaconf import OmegaConf

from utils import helpers
from utils.helpers import load
from torch.utils.data import DataLoader 
from data.transforms import build_transforms

from models.external.FoundationStereo.foundation_stereo import FoundationStereo
from models.external.FoundationStereo.core.utils.utils import InputPadder

os.chdir('/data/Zeitler/code/SIDE')
setup_environment()

# %%
# load debug config
with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
with open(os.path.join('configs', 'debug.yaml'), 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)

# %%
data_config = config['data']
dataset_class = load(data_config['dataset'])
train_transforms = build_transforms(config, mode='train')
train_subsets = dataset_class(mode='train').get_all_subset_names()

dataset_train = dataset_class(
    mode='train',
    transforms=train_transforms,
    tasks=config['training']['tasks'],
    subset_names=train_subsets
)
dataloader_train = DataLoader(
    dataset_train,
    batch_size=data_config['batch_size'],
    shuffle=True,
    num_workers=data_config['num_workers'],
    pin_memory=data_config['pin_memory'],
    persistent_workers=False
)

# %%

state_path = '/data/Zeitler/code/SIDE/models/external/FoundationStereo/state'

cfg = OmegaConf.load(os.path.join(state_path, 'cfg.yaml'))
args = OmegaConf.create(cfg)

model = FoundationStereo(args)
ckpt = torch.load(os.path.join(state_path, 'model_best_bp2.pth'), weights_only=False)
model.load_state_dict(ckpt['model'])
model.to('cuda')
model.eval()

# %%
for data in dataloader_train:
    with torch.cuda.amp.autocast(True) and torch.no_grad():
        left_images = data['image'].to('cuda')
        right_images = data['right_image'].to('cuda') if 'right_image' in data else None
        padder = InputPadder(left_images.shape, divis_by=32, force_square=False)
        left_images, right_images = padder.pad(left_images, right_images)
        
        # if hierarchical = False
        # outputs = model.forward(left_images, right_images, iters=args.valid_iters, test_mode=True)
        # if hierarchical = True
        outputs = model.run_hierachical(left_images, right_images, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
        outputs = padder.unpad(outputs.float())
        
        yy, xx = torch.meshgrid(torch.arange(outputs.shape[2], device=outputs.device), torch.arange(outputs.shape[3], device=outputs.device), indexing='ij')
        xx = xx.unsqueeze(0).unsqueeze(0)
        us_right = xx - outputs
        invalid = us_right < 0
        outputs[invalid] = 0
        
    break        


# %%
