# %% Imports
# Imports
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import time
import yaml
import numpy as np
from PIL import Image

import mlflow
import mlflow.artifacts
from mlflow.tracking import MlflowClient

import torch
from torch.utils.data import DataLoader, Dataset
from data.transforms import build_transforms


from processors.tester import Tester
from thop import profile



from utils import helpers

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)

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

# %% Settings
# Settings
arch = 'convnext'
state_path = 'SEG/260407:1208/train'
task_mode = 'segmentation' # 'disparity' or 'segmentation' or 'combined'

# %% Load mlflow data
# Load mlflow data
state_path_parts = state_path.split('/')
experiment = state_path_parts[0]
run_path = '/'.join(state_path_parts[1:])

#mlflow.set_tracking_uri('../mlruns')
mlflow_experiment = mlflow.get_experiment_by_name(experiment)
mlflow_run = mlflow.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f'run_name = "{state_path_parts[1]}"').iloc[0] 

base_config_filepath = mlflow.artifacts.download_artifacts(run_id=mlflow_run.run_id, artifact_path='configs/base.yaml', dst_path='../.temp')
experiment_config_filepath = mlflow.artifacts.download_artifacts(run_id=mlflow_run.run_id, artifact_path=f'configs/{arch}/{experiment}.yaml', dst_path='../.temp')

with open(base_config_filepath, 'r') as f: base_config = yaml.safe_load(f)
with open(experiment_config_filepath, 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)
config['logging']['notebook_mode'] = True

# %%
model_run_id = mlflow.search_runs(
    experiment_ids=[mlflow_experiment.experiment_id], 
    filter_string=f'tags.mlflow.runName = "{run_path}"', 
    order_by=['attributes.start_time DESC'], 
    max_results=1
).iloc[0].run_id

# %%

transform = build_transforms(config, mode='test')
dataset = EndoVisTeacherDataset(
    mode='train',
    transforms=transform
)

image = dataset[0]['image'].unsqueeze(0).to('cuda')
image_right = dataset[0]['right_image'].unsqueeze(0).to('cuda') if 'right_image' in dataset[0] else None

# model_path = f'runs:/{model_run_id}/best_model' #best_model_{task_mode}'
model_path = f'runs:/{model_run_id}/best_model_{task_mode}'
model = mlflow.pytorch.load_model(model_path, map_location='cuda')
model.eval()

# %%
num_warmup = 50
num_iterations = 1000

print('Warm Up')
torch.backends.cudnn.benchmark = True

with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(image, image_right)
        
print('Benchmarking')
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    torch.cuda.synchronize()
    start_event.record()

    for _ in range(num_iterations):
        _ = model(image, image_right)

    end_event.record()
    torch.cuda.synchronize()
    
# %%
total_time_ms = start_event.elapsed_time(end_event)
total_time_seconds = total_time_ms / 1000.0

fps = num_iterations / total_time_seconds

print('\n--- Results ---')
print(f'Total time for {num_iterations} images: {total_time_seconds:.4f} seconds')
print(f'Inference Speed: {fps:.2f} FPS')
print(f'Time per image: {(total_time_ms / num_iterations):.2f} ms')
peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) # Convert Bytes to Megabytes
print(f'Peak VRAM Usage: {peak_vram:.2f} MB')
print(f'VRAM Usage in GB: {peak_vram / 1024:.2f} GB')

# %%
model1_fps = 20.08
model2_fps = 13.28

# NVIDIA SF = 7.14 FPS // 140.06 ms // 2114.68 MB
# NVIDIA FS = 0.49 FPS // 2040.82 ms // 7126.30 MB in half precision
# Sequence Teacher = 0.46 FPS // 2180.88 ms

# SEG = 20.08 FPS // 49.80 ms // 1389.72 MB
# DISP = 13.28 FPS // 75.30 ms // 1435.92 MB
# Sequence Single Task = 7.99 FPS // 125.10 ms
# wMT (convnext/wMT/260406:2037/train) = 10.52 FPS // 95.02 ms // 1.44 GB // 540.29 G // 42.41 M
# wMT-KD-segmentation (convnext/wMT-KD/260406:2036/train) = 10.50 FPS // 95.23 ms // 1.44 GB // 540.29 G // 42.41 M

print((model1_fps * model2_fps) / (model1_fps + model2_fps))

# %% Parameters and FLOPs
print('\n--- Model Complexity ---')
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total Parameters: {total_params / 1e6:.2f} M')
print(f'Trainable Parameters: {trainable_params / 1e6:.2f} M')

inputs = (image, image_right) if image_right is not None else (image,)
macs, _ = profile(model, inputs=inputs, verbose=False)

print(f'MACs: {macs / 1e9:.2f} G (Giga-MACs)')

# %%
