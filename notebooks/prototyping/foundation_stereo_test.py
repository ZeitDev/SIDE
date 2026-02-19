# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))
from utils.setup import setup_environment

import yaml
import torch
from omegaconf import OmegaConf

from utils import helpers
from utils.helpers import load
from torch.utils.data import DataLoader 
from data.transforms import build_transforms

from models.teachers.FoundationStereo.foundation_stereo import FoundationStereo

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

# TODO: build remove_invisible into the forward pass

cfg = OmegaConf.load('/data/Zeitler/code/SIDE/models/teachers/FoundationStereo/state/cfg.yaml')
args = OmegaConf.create(cfg)

model = FoundationStereo(args)
model.to('cuda')
model.eval()
# %%

for data in dataloader_train:
    with torch.no_grad():
        left_images = data['image'].to('cuda')
        right_images = data['right_image'].to('cuda') if 'right_image' in data else None
        outputs = model.forward(left_images, right_images, iters=args.valid_iters, test_mode=True)
    break        

# %%
print(outputs.shape)
print(outputs[0])

# %%
