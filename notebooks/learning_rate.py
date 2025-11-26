# %% Imports
# Imports
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import yaml

import torch
import torch.nn as nn
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter

from utils import helpers
from utils.helpers import load
from processors.trainer import Trainer
from criterions.automatic_weighted_loss import AutomaticWeightedLoss

# %% Settings
EXPERIMENT = 'debug'
START_LR = 1e-7
END_LR = 10
NUM_ITER = 100

# %%
with open('../configs/base.yaml', 'r') as f: base_config = yaml.safe_load(f)
with open(f'../configs/{EXPERIMENT}.yaml', 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)
config['data']['batch_size'] = 8

# %%
class AutomaticWeightedLossWraper(nn.Module):
    def __init__(self, automatic_weighted_loss):
        super().__init__()
        self.automatic_weighted_loss = automatic_weighted_loss
        
    def forward(self, outputs, targets):
        total_loss, _ = self.automatic_weighted_loss(outputs, targets)
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
print(f'LR Finder for configuration: {EXPERIMENT}')

dataset_class = load(config['data']['dataset'])
trainer = Trainer(config, train_subsets=dataset_class(mode='train').get_all_subset_names())
model = trainer.model
raw_criterion = AutomaticWeightedLoss(trainer.criterions, freeze=True).to('cuda')
criterion = AutomaticWeightedLossWraper(raw_criterion).to('cuda')
train_iter = MultiTaskIter(trainer.dataloader_train)

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
