# %%
# Usage
# tmux new -s zeitler
# uv run notebooks/train/train_segmentation_teacher.py
# Detach with Ctrl+B, then D. Re-attach with `tmux attach -t zeitler`

# %%
# Imports
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml
import json
import random
import logging
import datetime
from tqdm import tqdm
from typing import cast
import numpy as np


import mlflow
from mlflow.models.signature import infer_signature

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F


from transformers import SegformerForSemanticSegmentation

from torch_lr_finder import LRFinder, TrainDataLoaderIter


from metrics.segmentation import IoU, Dice
from utils import helpers
from utils.helpers import load, upsample_logits
from data.transforms import build_transforms
from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)

from utils.logger import setup_logging, CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))
logging.getLogger('mlflow.utils.environment').setLevel(logging.ERROR)

mlflow.pytorch.autolog()

# %%
# Dataloader
config_name = 'segmentation_teacher'

with open(os.path.join('configs', 'base.yaml'), 'r') as f: base_config = yaml.safe_load(f)
with open(os.path.join('configs', config_name + '.yaml'), 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)

data_config = config['data']
dataset_class = helpers.load(data_config['dataset'])
train_transforms = build_transforms(config, mode='train')
val_transforms = build_transforms(config, mode='test')

# ! For LR Finder only
# config['data']['batch_size'] = 1


dataset_train = dataset_class(
    mode='train',
    config=config,
    transforms=train_transforms,
)
dataloader_train = DataLoader(
    dataset_train,
    batch_size=data_config['batch_size'],
    shuffle=True,
    num_workers=data_config['num_workers'],
    pin_memory=data_config['pin_memory'],
    persistent_workers=False
)
helpers.check_dataleakage('train', dataset_train)

dataset_val = dataset_class(
    mode='val',
    config=config,
    transforms=val_transforms,
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=data_config['batch_size'],
    shuffle=False,
    num_workers=data_config['num_workers'],
    pin_memory=data_config['pin_memory'],
    persistent_workers=False
)
helpers.check_dataleakage('val', dataset_val)

test_transforms = build_transforms(config, mode='test')
dataset_test = dataset_class(
    mode='test',
    config=config,
    transforms=test_transforms,
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=data_config['batch_size'],
    shuffle=False,
    num_workers=data_config['num_workers'],
    pin_memory=data_config['pin_memory'],
    persistent_workers=False
)
helpers.check_dataleakage('test', dataset_test)


# %%
# model
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

model_name = 'nvidia/mit-b5' # Huggingface SegFormer
model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=config['data']['num_of_classes']['segmentation'],
    ignore_mismatched_sizes=True
).to(device)
print(model.config.semantic_loss_ignore_index) # 255

# %%
# Settings
EPOCHS = config['training']['epochs']

OptimizerClass = load(config['training']['optimizer']['name'])
optimizer = OptimizerClass(
    model.parameters(),
    **config['training']['optimizer']['params'])

#config['training']['scheduler']['params']['num_warmup_steps'] = len(dataloader_train) * config['training']['scheduler']['warmup_epochs']

SchedulerClass = load(config['training']['scheduler']['name'])
scheduler = SchedulerClass(
    optimizer,
    **config['training']['scheduler']['params']
)

torch.cuda.empty_cache()
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# %%
# Find Learning Rate
if False:
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
            inputs = batch_data['image']
            targets = batch_data['segmentation'].squeeze(1).long() # (B, 1, H, W) -> (B, H, W)
            return inputs, targets
        
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, pixel_values):
            outputs = self.model(pixel_values=pixel_values)
            upsampled_logits = F.interpolate(
                outputs.logits, 
                size=pixel_values.shape[-2:], # (H, W) from input
                mode="bilinear", 
                align_corners=False
            )
            return upsampled_logits
        
    
    find_model_hf = model
    find_model = ModelWrapper(find_model_hf)
    find_model = find_model.to('cuda')
    
    train_iter = MultiTaskIter(dataloader_train)
    find_criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to('cuda')
    find_optimizer = AdamW(find_model.parameters(), lr=1e-7, weight_decay=1e-2)
    
    lr_finder = LRFinder(
        model=find_model,
        optimizer=find_optimizer,
        criterion=find_criterion,
        device='cuda')
    lr_finder.range_test(
        train_loader=train_iter,
        end_lr=10,
        num_iter=100)
    lr_finder.plot(suggest_lr=True)
    lr_finder.reset()

# %%
# Training
mlflow.set_experiment(config_name)
run_datetime = datetime.datetime.now().strftime("%y%m%d:%H%M")

log_filepath = os.path.join('logs', f'{run_datetime}_{config_name}.log')
setup_logging(log_filepath=log_filepath, vram_only=config['logging']['vram'])

train_log_interval = max(1, int(len(dataloader_train) * config['logging']['log_interval']))

IoU_metric = IoU(n_classes=config['data']['num_of_classes']['segmentation'], device=device)
DICE_metric = Dice(n_classes=config['data']['num_of_classes']['segmentation'], device=device)

global_step = 0
train_loss_running = 0.0
best_val_dice = 0.0
best_metrics = {}
with mlflow.start_run(run_name=run_datetime) as run:
    helpers.mlflow_log_misc(log_filepath)
    tags = {}
    tags['parent_name'] = config_name
    tags['run_type'] = 'root'
    tags['description'] = config['description']
    helpers.mlflow_log_run(config, tags=tags)
    
    with mlflow.start_run(run_name=f'{run.info.run_name}/train', nested=True) as train_run:
        tags['parent_name'] = run.info.run_name
        tags['run_type'] = 'train'
        tags['run_mode'] = 'validation' if config['data']['validation'] else 'full_training'
        helpers.mlflow_log_run(config, tags=tags)
        mlflow.set_tag('mlflow.note.content', config['description'])
        
        scaler = torch.amp.GradScaler(device.type)
        accumulation_steps = config['data']['accumulate_grad_batches']
        for epoch in range(EPOCHS):
            print(f'Epoch {epoch+1}/{EPOCHS} - Training')
            
            ### Training Loop ###
            model.train()
            optimizer.zero_grad()
            
            train_batch_tqdm = tqdm(dataloader_train, desc='Training')
            for idx, data in enumerate(train_batch_tqdm):
                images = data['image'].to(device)
                targets = data['segmentation'].to(device)
                
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(pixel_values=images, labels=targets.squeeze(1).long())
                    loss = outputs.loss
                train_loss_running += loss.item()
                
                accumulated_loss = loss / accumulation_steps
                scaler.scale(accumulated_loss).backward()
        
                if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(dataloader_train):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
                    scaler.step(optimizer)
                    scaler.update()
            
                    scheduler.step()
                    optimizer.zero_grad()
                
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # scheduler.step()
                
                global_step += 1
                if global_step % train_log_interval == 0:
                    fractional_epoch = epoch + (idx / len(dataloader_train))
                    average_loss = train_loss_running / train_log_interval
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    mlflow.log_metric('epoch_train', fractional_epoch, step=global_step)
                    mlflow.log_metric('optimization/training/loss', average_loss, step=global_step)
                    mlflow.log_metric('optimization/training/learning_rate', current_lr, step=global_step)
                    train_batch_tqdm.set_postfix({'loss': average_loss, 'lr': current_lr})
                    
                    train_loss_running = 0.0
                
            ### Validation Loop ###
            if config['data']['validation']:
                model.eval()
                val_loss = 0.0
                
                IoU_metric.reset()
                DICE_metric.reset()
                
                val_batch_tqdm = tqdm(dataloader_val, desc='Validation')
                with torch.no_grad():
                    for data in val_batch_tqdm:
                        images = data['image'].to(device)
                        targets = data['segmentation'].to(device)
                        
                        with torch.autocast(device_type=device.type, dtype=torch.float16):
                            outputs = model(pixel_values=images, labels=targets.squeeze(1).long())
                            loss = outputs.loss
                        val_loss += loss.item()
                        
                        logits = outputs.logits
                        upsampled_logits = upsample_logits(logits, size=targets.shape[-2:])
                        
                        IoU_metric.update(upsampled_logits, targets)
                        DICE_metric.update(upsampled_logits, targets)

                val_loss /= len(dataloader_val)
                IoU_results = IoU_metric.compute()
                DICE_results = DICE_metric.compute()
                
                val_metrics = {}
                val_metrics['optimization/validation/loss'] = val_loss
                
                for key, value in IoU_results.items():
                    if key == 0: key = 'background'
                    if key == 1: key = 'instrument'
                    val_metrics[f'performance/validation/segmentation/IoU_score/{key}'] = value
                for key, value in DICE_results.items():
                    if key == 0: key = 'background'
                    if key == 1: key = 'instrument'
                    val_metrics[f'performance/validation/segmentation/DICE_score/{key}'] = value
                print(f'Validation Loss: {val_loss:.4f}')
                mlflow.log_metrics(val_metrics, step=global_step)
                
                dice_instrument = DICE_results[1]
                if dice_instrument > best_val_dice:
                    best_epoch = epoch
                    best_val_dice = dice_instrument
                    best_metrics = {f'best/segmentation/{k.replace("/", "_")}': v for k, v in val_metrics.items()}
                    best_metrics['best/segmentation/epoch'] = best_epoch + 1
                    mlflow.log_metrics(best_metrics, step=global_step)
                    
                    torch.save({'model_state_dict': model.state_dict()}, os.path.join('.temp', 'model_state.pth'))
                    print(f'New best model saved with DICE Instrument: {best_val_dice:.4f}')
            else:
                if epoch == EPOCHS - 1:
                    torch.save({'model_state_dict': model.state_dict()}, os.path.join('.temp', 'model_state.pth'))
                    best_metrics = {'best/segmentation/epoch': epoch + 1}
                    mlflow.log_metrics(best_metrics, step=global_step)
                    print('No validation - model from last epoch saved as best model.')
            
            mlflow.log_metric('epoch_validation', epoch + 1, step=global_step)
            print()

        state_dict = torch.load(os.path.join('.temp', 'model_state.pth'))
        model.load_state_dict(state_dict['model_state_dict'])
        model.to('cpu')
        model.eval()

        signature_input_example = torch.rand(1, images.shape[1], images.shape[2], images.shape[3])
        signature_output_example = model(pixel_values=signature_input_example).logits.detach()
        signature = infer_signature(signature_input_example.numpy(), signature_output_example.numpy())

        mlflow.pytorch.log_model( 
            pytorch_model=model,
            name='best_model',
            code_paths=['models/'],
            signature=signature
        )

    print('Training completed and best model logged to MLflow.')

    with mlflow.start_run(run_name=f'{run.info.run_name}/test', nested=True) as test_run:
        tags['parent_name'] = run.info.run_name
        tags['run_type'] = 'test'
        helpers.mlflow_log_run(config, tags=tags)
        mlflow.set_tag('mlflow.note.content', config['description'])
        
        del model
        model = mlflow.pytorch.load_model(f'runs:/{train_run.info.run_id}/best_model')
        model.to(device)
        model.eval()

        IoU_metric.reset()
        DICE_metric.reset()

        batch_tqdm = tqdm(dataloader_test, desc='Testing')
        with torch.no_grad():
            for data in batch_tqdm:
                images = data['image'].to(device)
                targets = data['segmentation'].to(device)
                
                outputs = model(pixel_values=images, labels=targets.squeeze(1).long())
                loss = outputs.loss
                
                logits = outputs.logits
                upsampled_logits = upsample_logits(logits, size=targets.shape[-2:])
                
                IoU_metric.update(upsampled_logits, targets)
                DICE_metric.update(upsampled_logits, targets)
                
        IoU_results = IoU_metric.compute()
        DICE_results = DICE_metric.compute()

        test_metrics = {}
        for key, value in IoU_results.items():
            if key == 0: key = 'background'
            if key == 1: key = 'instrument'
            test_metrics[f'performance/testing/segmentation/IoU_score/{key}'] = value
        for key, value in DICE_results.items():
            if key == 0: key = 'background'
            if key == 1: key = 'instrument'
            test_metrics[f'performance/testing/segmentation/DICE_score/{key}'] = value
            
        mlflow.log_metrics(test_metrics)

        print('Testing completed and metrics logged to MLflow.')
        print()
        
        print('Test Metrics:')
        print(json.dumps(test_metrics, indent=4))
        print()
        

# %%