# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))


import os
import utils.helpers
from utils.helpers import load, deep_merge
from data.transforms import build_transforms
import yaml
from torch.utils.data import Dataset, DataLoader
import datetime
import torch
from mlflow.models.signature import infer_signature
import mlflow
from monai.networks.nets import SwinUNETR
from tqdm import tqdm

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment()

# %%
with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
with open(os.path.join('configs', 'debug.yaml'), 'r') as f: experiment_config = yaml.safe_load(f)
config = deep_merge(experiment_config, base_config)

# %%
data_config = config['data']
dataset_class = load(data_config['dataset'])
test_transforms = build_transforms(config, mode='test')
test_subsets = dataset_class(mode='test', config=config).get_all_subset_names()

dataset_test = dataset_class(
    mode='test',
    config=config,
    transforms=test_transforms,
    subset_names=test_subsets
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=data_config['batch_size'],
    shuffle=True,
    num_workers=data_config['num_workers'],
    pin_memory=data_config['pin_memory'],
    persistent_workers=False
)

# %%
model = SwinUNETR(
    in_channels=3,
    out_channels=2,
    use_checkpoint=True,
    spatial_dims=2,
    use_v2=True,
    feature_size=48,
)
    
model.to('cuda')
model.eval()

# %% Simple test loop
IoUs = []
DICEs = []
for data in tqdm(dataloader_test):
    left_img = data['image'].to('cuda')
    segmentation_mask = data['segmentation'].to('cuda')
    
    with torch.no_grad():
        output = model(left_img)
    
    # calculate DICE and IoU
    pred = torch.argmax(output, dim=1, keepdim=True).float()
    intersection = (pred * segmentation_mask).sum()
    union = pred.sum() + segmentation_mask.sum()
    dice = (2. * intersection) / (union + 1e-8)
    iou = intersection / (union - intersection + 1e-8)
    IoUs.append(iou.item())
    DICEs.append(dice.item())
    
print(f'Average DICE: {sum(DICEs)/len(DICEs):.4f}, Average IoU: {sum(IoUs)/len(IoUs):.4f}')

# %%