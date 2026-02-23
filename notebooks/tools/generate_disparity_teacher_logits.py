# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml
import torch
import cv2
from utils import helpers
from utils.helpers import load, soft_argmin
from torch.utils.data import DataLoader
from data.transforms import build_transforms
from models.teachers.foundation_stereo_wrapper import FoundationStereoWrapper

from torch.utils.data import Dataset
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
# data_config = config['data']
# dataset_class = load(data_config['dataset'])
# train_transforms = build_transforms(config, mode='train')
# train_subsets = dataset_class(mode='train').get_all_subset_names()

# dataset_train = dataset_class(
#     mode='train',
#     transforms=train_transforms,
#     tasks=config['training']['tasks'],
#     subset_names=train_subsets
# )
# dataloader_train = DataLoader(
#     dataset_train,
#     batch_size=data_config['batch_size'],
#     shuffle=True,
#     num_workers=data_config['num_workers'],
#     pin_memory=data_config['pin_memory'],
#     persistent_workers=False
# )

# %%
# Custom EndoVis Dataloader

class EndoVisTeacherDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = '/data/Zeitler/SIDED/EndoVis17/processed/train'
        self.transforms = transforms
        
        self.subsets = sorted(os.listdir(self.root_dir))
        self.file_names = sorted(os.listdir(os.path.join(self.root_dir, self.subsets[0], 'input', 'left_images')))
        
        self.left_image_paths = []
        self.right_image_paths = []
        for subset in self.subsets:
            for file_name in self.file_names:
                self.left_image_paths.append(os.path.join(self.root_dir, subset, 'input', 'left_images', file_name))
                self.right_image_paths.append(os.path.join(self.root_dir, subset, 'input', 'right_images', file_name))
        
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
# ! remember to do it with train and test both
config['data']['transforms']['train'] = [
    {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
]
train_transforms_disparity = build_transforms(config, mode='train')

dataset_train_disparity = EndoVisTeacherDataset(
    root_dir='/data/Zeitler/SIDED/EndoVis17/processed/train',
    transforms=train_transforms_disparity)

dataloader_train_disparity = DataLoader(
    dataset_train_disparity,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

for data in tqdm(dataloader_train_disparity):
    with torch.cuda.amp.autocast(True) and torch.no_grad():
        left_image = data['image'].to('cuda')
        right_image = data['right_image'].to('cuda') if 'right_image' in data else None
        
        disparity = model.get_disparity(left_image, right_image)
        
        image_path = data['image_path'][0]
        save_path = image_path.replace('input', 'ground_truth').replace('left_images', 'disparity')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        disp = disparity.squeeze().cpu().numpy()
        disp_scaled = disp * 128.0
        disp_scaled = np.clip(disp_scaled, 0, 65535)
        disp_scaled = disp_scaled.astype(np.uint16)
            
        cv2.imwrite(save_path, disp_scaled)
    

# %% Save Disparity Teacher Logits
# config['data']['transforms']['train'] = [
#     {'name': 'CenterCrop', 'params': {'height': 1024, 'width': 1024}},
#     {'name': 'Resize', 'params': {'height': 1024, 'width': 1024}},
#     {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
# ]
# train_transform = build_transforms(config, mode='train')

# dataset_train = EndoVisTeacherDataset(
#     root_dir='/data/Zeitler/SIDED/EndoVis17/processed/train',
#     transforms=train_transform)

# dataloader_train = DataLoader(
#     dataset_train,
#     batch_size=1,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=False,
#     persistent_workers=False
# )

# for data in tqdm(dataloader_train):
#     with torch.cuda.amp.autocast(True) and torch.no_grad():
#         left_image = data['image'].to('cuda')
#         right_image = data['right_image'].to('cuda') if 'right_image' in data else None
        
#         logit = model.get_logits(left_image, right_image)
#         # raw_disparity = soft_argmin(logit, size=left_image.shape[2:]) * 512.0
        
#         image_path = data['image_path'][0]
#         save_path = image_path.replace('input', 'teacher').replace('left_images', 'disparity_128_256_256').replace('.png', '.pt')
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         torch.save(logit.squeeze().cpu(), save_path)
    
#     # break


# %%