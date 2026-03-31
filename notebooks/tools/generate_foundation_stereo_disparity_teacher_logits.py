# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml
import torch
import cv2
from utils import helpers
from utils.helpers import load, logits2disparity
from torch.utils.data import DataLoader
from data.transforms import build_transforms
from models.teachers.foundation_stereo import FoundationStereoWrapper

from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from PIL import Image

from tqdm import tqdm

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment()


# %%
# load debug config
with open(os.path.join('configs', 'base.yaml'), 'r') as f: config = yaml.safe_load(f)
#with open(os.path.join('configs', 'debug.yaml'), 'r') as f: experiment_config = yaml.safe_load(f)
#config = helpers.deep_merge(experiment_config, base_config)

# %%
# Custom EndoVis Dataloader

class EndoVisTeacherDataset(Dataset):
    def __init__(self, mode, transforms=None):
        self.root_dir = f'/data/Zeitler/SIDED/EndoVis17/processed/{mode}'
        self.transforms = transforms
        
        self.subsets = sorted(os.listdir(self.root_dir))
        self.file_names = sorted(os.listdir(os.path.join(self.root_dir, self.subsets[0], 'input', 'left_images')))
        
        self.left_image_paths = []
        self.right_image_paths = []
        for subset in self.subsets:
            for file_name in self.file_names:
                self.left_image_paths.append(os.path.join(self.root_dir, subset, 'input', 'left_images', file_name))
                self.right_image_paths.append(os.path.join(self.root_dir, subset, 'input', 'right_images', file_name))
        
        self.sample_paths = []
        for left_path, right_path in zip(self.left_image_paths, self.right_image_paths):
            self.sample_paths.append({'left_image': left_path, 'right_image': right_path})
        
    def __len__(self):
        return len(self.left_image_paths)
    
    def __getitem__(self, idx):
        data = {}
        left_image_path = self.left_image_paths[idx]
        right_image_path = self.right_image_paths[idx]
        data['image'] = np.array(Image.open(left_image_path).convert('RGB'))
        data['right_image'] = np.array(Image.open(right_image_path).convert('RGB'))
        
        if self.transforms: data = self.transforms(**data)
        
        data['image_path'] = left_image_path
        
        return data


# %%
model = FoundationStereoWrapper()
model.to('cuda')
model.eval()

# %% Save Disparity Maps
if False:
    mode = 'test'  # ! Do train, val and test
    
    transform_mode = 'train' if mode == 'train' else 'test'
    config['data']['transforms'][transform_mode] = [
        # {'name': 'CenterCrop', 'params': {'height': 1024, 'width': 1024}},
        # {'name': 'Resize', 'params': {'height': 1024, 'width': 1024}},
        {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
    ]
    transforms_disparity = build_transforms(config, mode=transform_mode)

    dataset_disparity = EndoVisTeacherDataset(
        mode=mode,
        transforms=transforms_disparity
    )

    dataloader_disparity = DataLoader(
        dataset_disparity,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    helpers.check_dataleakage(mode, dataset_disparity)

    for data in tqdm(dataloader_disparity):
        with torch.cuda.amp.autocast(True) and torch.no_grad():
            left_image = data['image'].to('cuda')
            right_image = data['right_image'].to('cuda')
            
            disparity = model.get_disparity(left_image, right_image)
            
            image_path = data['image_path'][0]
            save_path = image_path.replace('input', 'target').replace('left_images', 'disparity')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            disp = disparity.squeeze().cpu().numpy()
            disp_scaled = disp * 128.0
            disp_scaled = np.clip(disp_scaled, 0, 65535)
            disp_scaled = disp_scaled.astype(np.uint16)
                
            cv2.imwrite(save_path, disp_scaled)

# %% Save Disparity Teacher Logits
if False:
    mode = 'test'
    
    transform_mode = 'train' if mode == 'train' else 'test'
    config['data']['transforms'][transform_mode] = [
        # {'name': 'CenterCrop', 'params': {'height': 1024, 'width': 1024}},
        # {'name': 'Resize', 'params': {'height': 1024, 'width': 1024}},
        {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
    ]
    train_transform = build_transforms(config, mode=transform_mode)

    dataset_train = EndoVisTeacherDataset(
        mode=mode,
        transforms=train_transform
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    helpers.check_dataleakage(mode, dataset_train)

    FP16_MAX = torch.finfo(torch.float16).max
    for data in tqdm(dataloader_train):
        with torch.cuda.amp.autocast(True) and torch.no_grad():
            left_image = data['image'].to('cuda')
            right_image = data['right_image'].to('cuda')
            
            logit = model.get_logits(left_image, right_image)

            logit_save = logit.squeeze().half().cpu()
            assert logit_save.abs().max() < FP16_MAX, f'Logit values exceed FP16 max value: {logit_save.abs().max()}'
            assert not torch.isnan(logit_save).any(), 'Logit contains NaN values'

            # raw_disparity = logits2disparity(logit, size=left_image.shape[2:]) * 512.0
            
            image_path = data['image_path'][0]
            save_path = image_path.replace('input', 'teacher').replace('left_images', 'disparity_128_256_256').replace('.png', '.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(logit_save, save_path)


# %% Saving Right Disparity Maps

if False:
    mode = 'test' # ! Do both train and test
    
    transform_mode = 'train' if mode == 'train' else 'test'
    config['data']['transforms'][transform_mode] = [
        {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
    ]
    transforms_disparity = build_transforms(config, mode=transform_mode)

    dataset_disparity = EndoVisTeacherDataset(
        mode=mode,
        transforms=transforms_disparity
    )

    dataloader_disparity = DataLoader(
        dataset_disparity,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    helpers.check_dataleakage(mode, dataset_disparity)

    valid_percentages = []
    for data in tqdm(dataloader_disparity):
        with torch.cuda.amp.autocast(True) and torch.no_grad():
            left_image = data['image'].to('cuda')
            right_image = data['right_image'].to('cuda')
            #left_disparity = model.get_disparity(left_image, right_image)
            
            left_image_flipped = torch.flip(left_image, dims=[3])
            right_image_flipped = torch.flip(right_image, dims=[3])
            right_disparity_flipped = model.get_disparity(right_image_flipped, left_image_flipped)
            right_disparity = torch.flip(right_disparity_flipped, dims=[3])
            
            image_path = data['image_path'][0]
            save_path = image_path.replace('input', 'target').replace('left_images', 'disparity_right')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            disp = right_disparity.squeeze().cpu().numpy()
            disp_scaled = disp * 128.0
            disp_scaled = np.clip(disp_scaled, 0, 65535)
            disp_scaled = disp_scaled.astype(np.uint16)
                
            cv2.imwrite(save_path, disp_scaled)
            
# %% Speed Evaluation
transforms_speed = build_transforms(config, mode='test')

dataset_speed = EndoVisTeacherDataset(mode='train', transforms=transforms_speed)

left_image = dataset_speed[0]['image'].unsqueeze(0).to('cuda')
right_image = dataset_speed[0]['right_image'].unsqueeze(0).to('cuda')

num_warmup = 50
num_iterations = 1000

print('Warm Up')
with torch.autocast('cuda'), torch.no_grad():
    for _ in range(num_warmup):
        _ = model.get_disparity(left_image, right_image)

print('Benchmarking')
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.autocast('cuda'), torch.no_grad():
    torch.cuda.synchronize()
    start_event.record()

    for _ in range(num_iterations):
        _ = model.get_disparity(left_image, right_image)

    end_event.record()
    torch.cuda.synchronize()

total_time_ms = start_event.elapsed_time(end_event)
total_time_seconds = total_time_ms / 1000.0

fps = num_iterations / total_time_seconds

print('\n--- Results ---')
print(f'Total time for {num_iterations} images: {total_time_seconds:.4f} seconds')
print(f'Inference Speed: {fps:.2f} FPS')
print(f'Time per image: {(total_time_ms / num_iterations):.2f} ms')
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) # Convert Bytes to Megabytes
print(f'Peak VRAM Usage: {peak_vram:.2f} MB')

# %%