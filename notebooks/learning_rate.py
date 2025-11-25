# %% Imports
# Imports
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import yaml

from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter

from utils import helpers
from utils.helpers import load
from processors.trainer import Trainer

# %% Settings
EXPERIMENT = 'overfit'
TASK = 'segmentation'
START_LR = 1e-10
END_LR = 10
NUM_ITER = 100

# TODO: adjust for multi TASK

# %%
with open('../configs/base.yaml', 'r') as f: base_config = yaml.safe_load(f)
with open(f'../configs/{EXPERIMENT}.yaml', 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)
config['data']['batch_size'] = 8

# %%
print(f'LR Finder for EXPERIMENT configuration: {EXPERIMENT}')

dataset_class = load(config['data']['dataset'])
all_train_subsets = dataset_class(mode='train').get_all_subset_names()
trainer = Trainer(config, train_subsets=all_train_subsets)
trainloader = trainer.dataloader_train

# %%
# TODO: Implement multi-TASK loss that can optionally include tasks
import monai
import torch.nn as nn
class FlexibleMultiTaskLoss(nn.Module):
    def __init__(self, train_disparity=False, seg_weight=1.0, disp_weight=1.0):
        super().__init__()
        self.train_disparity = train_disparity
        
        # Define Criteria
        self.seg_criterion = monai.losses.DiceCELoss( # type: ignore
            to_onehot_y=True,
            softmax=True,
            
        ) 
        self.disp_criterion = nn.L1Loss() # or whatever you use
        
        # Weights (only relevant if both are active)
        self.w_seg = seg_weight
        self.w_disp = disp_weight

    def forward(self, outputs, targets):
        """
        outputs: Model predictions dictionary
        targets: Ground truth dictionary
        """
        # --- 1. Always Calculate Segmentation ---
        # We assume 'segmentation' is the primary TASK and always exists
        loss_seg = self.seg_criterion(outputs['segmentation'], targets['segmentation'])
        
        total_loss = self.w_seg * loss_seg

        # --- 2. Optionally Calculate Disparity ---
        if self.train_disparity:
            # We only access these keys if the flag is True.
            # This prevents crashing if targets['disparity'] is None or garbage.
            loss_disp = self.disp_criterion(outputs['disparity'], targets['disparity'])
            total_loss += (self.w_disp * loss_disp)
            
        return total_loss
    
class DictToDevice:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        
    def to(self, device, non_blocking=False, **kwargs):
        return {
            k: v.to(device, non_blocking=non_blocking, **kwargs) 
            for k, v in self.data_dict.items()
        }

class MultiTaskIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        inputs, targets = batch_data
        return inputs, DictToDevice(targets)


# %%
model = trainer.model
criterion = FlexibleMultiTaskLoss(train_disparity=False)
train_iter = MultiTaskIter(trainloader)
optimizer_config = config['training']['optimizer']
optimizer_config['params']['lr'] = START_LR
optimizer_class = load(optimizer_config['name'])
optimizer = optimizer_class(
    model.parameters(),
    **optimizer_config['params'])

lr_finder = LRFinder(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda')
lr_finder.range_test(
    train_loader=train_iter,
    end_lr=END_LR,
    num_iter=NUM_ITER)
lr_finder.plot(suggest_lr=True)
lr_finder.reset()

# %%
