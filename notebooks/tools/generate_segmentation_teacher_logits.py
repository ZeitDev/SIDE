# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml
import torch
from utils import helpers
from torch.utils.data import DataLoader
from data.transforms import build_transforms

import mlflow

from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from tqdm import tqdm

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment()

# %%
# Settings
run = 'segmentation_teacher/260226:1718/train'

# %%
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
        
        self.sample_paths = []
        for left_path in self.left_image_paths:
            self.sample_paths.append({'left_image': left_path})
        
    def __len__(self):
        return len(self.left_image_paths)
    
    def __getitem__(self, idx):
        data = {}
        left_image_path = self.left_image_paths[idx]
        data['image'] = np.array(Image.open(left_image_path).convert('RGB'))
        
        if self.transforms: data = self.transforms(**data)
        
        data['image_path'] = left_image_path
        
        return data
    
# %% 
with open(os.path.join('configs', 'base.yaml'), 'r') as f: config = yaml.safe_load(f)

mode = 'test' # 'train' or 'test'

config['data']['transforms'][mode] = [
    {'name': 'CenterCrop', 'params': {'height': 1024, 'width': 1024}},
    {'name': 'Resize', 'params': {'height': 1024, 'width': 1024}},
    {'name': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
]
transform = build_transforms(config, mode=mode)

dataset = EndoVisTeacherDataset(
    mode=mode,
    transforms=transform)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

# %%
# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_run_id = helpers.get_model_run_id(run)
model = mlflow.pytorch.load_model(f'runs:/{model_run_id}/best_model').to(device)
model.eval()

# %%
FP16_MAX = torch.finfo(torch.float16).max
for data in tqdm(dataloader):
    with torch.cuda.amp.autocast(True) and torch.no_grad():
        left_image = data['image'].to('cuda')
        
        logit = model(pixel_values=left_image).logits
        
        logit_save = logit.squeeze().half().cpu()
        assert logit_save.abs().max() < FP16_MAX, f'Logit values exceed FP16 max value: {logit_save.abs().max()}'
        assert not torch.isnan(logit_save).any(), 'Logit contains NaN values'
        
        #logit_upsampled = helpers.upsample_logits(logit_save.float().unsqueeze(0), size=left_image.shape[2:])
        #raw_segmentation = torch.argmax(logit_upsampled, dim=1)
        
        image_path = data['image_path'][0]
        save_path = image_path.replace('input', 'teacher').replace('left_images', 'segmentation_2_256_256').replace('.png', '.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(logit_save, save_path)

# %%