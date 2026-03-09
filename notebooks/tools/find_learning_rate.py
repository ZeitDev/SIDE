# %% Imports
# Imports
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml

import torch.nn as nn
from torch_lr_finder import LRFinder, TrainDataLoaderIter

from data.transforms import build_transforms
from utils import helpers
from utils.helpers import load
from processors.trainer import Trainer
from criterions.automatic_weighted_loss import AutomaticWeightedLoss

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment()

# %% Settings
EXPERIMENT = 'debug'
START_LR = 1e-7
END_LR = 10
NUM_ITER = 100

# %%
with open('./configs/base.yaml', 'r') as f: base_config = yaml.safe_load(f)
with open(f'./configs/{EXPERIMENT}.yaml', 'r') as f: experiment_config = yaml.safe_load(f)
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

class ModelInputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, inputs):
        if isinstance(inputs, dict):
            return self.model(inputs['image'], inputs.get('right_image'))
        
        return self.model(inputs)

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
        inputs = {}
        targets = {}
        
        for k, v in batch_data.items():
            if k in ['image', 'right_image']:
                inputs[k] = v
            else:
                targets[k] = v
        
        return DictToDevice(inputs), DictToDevice(targets)


# %%
print(f'LR Finder for configuration: {EXPERIMENT}')


trainer = Trainer(config)
model = ModelInputWrapper(trainer.model)
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
"""
# --- HOW TO READ A LOGARITHMIC LEARNING RATE GRAPH (LR FINDER) ---
1. MAJOR TICKS (The Exponents):
   These step up by multiplying by 10.
   10^-5 = 1e-5 = 0.00001
   10^-4 = 1e-4 = 0.0001
   10^-3 = 1e-3 = 0.001

2. MINOR TICKS (The Multipliers):
   They are linear multipliers (2x, 3x, 4x...) of the major tick to their left.
   
   Example: Reading the space between 10^-4 and 10^-3:
   - Major Tick:     10^-4 (1e-4)
   - 1st minor tick: 2e-4 (2 * 10^-4)
   - 2nd minor tick: 3e-4 (3 * 10^-4) 
   - 3rd minor tick: 4e-4
   - ...
   - 8th minor tick: 9e-4
   - Next Major Tick: 10^-3 (1e-3)

3. THE VISUAL TRICK:
   Because it's a log scale, the physical space between 1e-4 and 2e-4 is much wider 
   than the space between 8e-4 and 9e-4. The lines look "squeezed" together as they 
   approach the next major exponent.

RULE OF THUMB (Leslie Smith): 
Find the absolute lowest loss value before the graph spikes upwards. 
Take that learning rate and divide it by 10. 
(Or find the steepest, longest downward slope and pick the middle point).


DIFFERENT BATCH SIZES:
Rule of thumb: LR_new = LR_old * (BatchSize_new / BatchSize_old)
"""