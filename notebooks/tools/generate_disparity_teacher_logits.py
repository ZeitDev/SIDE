# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml
import torch
from utils import helpers
from utils.helpers import load
from torch.utils.data import DataLoader
from data.transforms import build_transforms
from models.teachers.foundation_stereo_wrapper import FoundationStereoWrapper


from utils.setup import setup_environment
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

model = FoundationStereoWrapper()
model.to('cuda')
model.eval()


# %%
for data in dataloader_train:
    with torch.cuda.amp.autocast(True) and torch.no_grad():
        left_images = data['image'].to('cuda')
        right_images = data['right_image'].to('cuda') if 'right_image' in data else None
        
        # outputs = model(left_images, right_images)
        logits = model.get_logits(left_images, right_images)
        
        torch.save(logits.squeeze().cpu(), 'debug_logits.pt') # care to only save [128, H, W] tensor, not the whole batch
    break


# %%