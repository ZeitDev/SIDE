# %% Imports
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import math
import torch
import yaml
from tqdm import tqdm

import mlflow
import mlflow.artifacts

from utils import helpers
from data.transforms import build_transforms

from torch.utils.data import DataLoader, Dataset

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment()

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Restrict to GPU 1 for this notebook

# %% Helpers

def get_layer_entropy(features: torch.Tensor):
    B, C, H, W = features.shape
    features = features.permute(0, 2, 3, 1).reshape(-1, C) # flattens to feature vectors of shape [B, C, H, W] -> [B*H*W, C] = [N, C]
    
    features_centred = features - features.mean(dim=0, keepdim=True) # zero-center features
    covariance_matrix = torch.cov(features_centred.T) # [C, N] * [N, C] -> [C, C]
    
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix) # eigh for symmetric matrix
    eigenvalues = torch.clamp(eigenvalues, min=1e-9) # prevent negatives and zeros
    
    singular_values = torch.sqrt(eigenvalues) # [C]
    probs = singular_values / torch.sum(singular_values) # normalize
    entropy = -torch.sum(probs * torch.log(probs)) # Shannon Entropy using log natural as in IDRR paper
    
    max_entropy = math.log(C) # max entropy when all singular values are equal
    normalized_entropy = entropy / max_entropy # make entropy log independent and scale to [0, 1]
    
    effective_rank = torch.exp(entropy) # effective rank as exp(entropy) as in IDRR paper
    effective_rank_ratio = effective_rank / C # normalize by number of channels
    
    # Normalized Entropy: How balanced is the workload between effective channels?
    # Effective Rank Ratio: How many effective channels are being used relative to total channels?
    return normalized_entropy, effective_rank_ratio
    

class LayerEntropyTracker:
    def __init__(self, channels: int, device='cuda'):
        self.C = channels
        self.N = 0
        self.device = device
        self.sum_features = torch.zeros(channels, dtype=torch.float64, device=device) # [C]
        self.sum_features_squared = torch.zeros((channels, channels), dtype=torch.float64, device=device) # [C, C]
        
    @torch.no_grad()
    def update(self, features: torch.Tensor):
        features = features.permute(0, 2, 3, 1).reshape(-1, self.C) # [N, C]
        features = features.to(dtype=torch.float64, device=self.device)
        
        self.N += features.shape[0]
        self.sum_features += features.sum(dim=0) # [C]
        self.sum_features_squared += features.T @ features # [C, N] @ [N, C] -> [C, C]
        
    def compute_entropy(self):
        # Cov = (sum_features_squared - mean / N) / (N - 1)
        sum_outer_product = torch.outer(self.sum_features, self.sum_features) / self.N # [C] @ [C]^T -> [C, C]
        covariance_matrix = (self.sum_features_squared - sum_outer_product) / (self.N - 1) # [C, C]
        
        eigenvalues, _ = torch.linalg.eigh(covariance_matrix) # [C]
        eigenvalues = torch.clamp(eigenvalues, min=1e-12) # prevent negatives and zeros
        
        singular_values = torch.sqrt(eigenvalues) # [C]
        probs = singular_values / torch.sum(singular_values) # normalize
        probs_safe = probs[probs > 1e-12]
        entropy = -torch.sum(probs_safe * torch.log(probs_safe)).item() # Shannon Entropy using
        
        max_entropy = math.log(self.C) # max entropy when all singular values are equal
        normalized_entropy = entropy / max_entropy # make entropy log independent and scale to [0, 1]
        
        effective_rank = math.exp(entropy) # effective rank as exp(entropy) as in IDRR paper
        effective_rank_ratio = effective_rank / self.C # normalize by number of channels
        
        return normalized_entropy, effective_rank_ratio
    
# %% Test 1: Pure Chaos with Tracker
B, C, H, W = 2, 768, 16, 16

features_chaos = torch.randn(B, C, H, W)
norm_entropy_1, erank_ratio_1 = get_layer_entropy(features_chaos)

tracker_1 = LayerEntropyTracker(C, device=features_chaos.device)
tracker_1.update(features_chaos)
norm_entropy_1_tracker, erank_ratio_1_tracker = tracker_1.compute_entropy()

print("Test 1: Pure Chaos with Tracker")
print(f"Direct Normalized Entropy:   {norm_entropy_1.item():.2%}")
print(f"Tracker Normalized Entropy:  {norm_entropy_1_tracker:.2%}")
print(f"Direct Effective Rank Ratio: {erank_ratio_1.item():.2%}")
print(f"Tracker Effective Rank Ratio: {erank_ratio_1_tracker:.2%}\n")

# Test 2: Total Collapse (1 Dominant Channel) with Tracker
features_collapse = torch.zeros(B, C, H, W)
features_collapse[:, 0, :, :] = torch.randn(B, H, W) * 100 # Channel 0 goes crazy

norm_entropy_2, erank_ratio_2 = get_layer_entropy(features_collapse)

tracker_2 = LayerEntropyTracker(C, device=features_collapse.device)
tracker_2.update(features_collapse)
norm_entropy_2_tracker, erank_ratio_2_tracker = tracker_2.compute_entropy()

print("Test 2: Total Collapse (1 Dominant Channel) with Tracker")
print(f"Direct Normalized Entropy:   {norm_entropy_2.item():.2%}")
print(f"Tracker Normalized Entropy:  {norm_entropy_2_tracker:.2%}")
print(f"Direct Effective Rank Ratio: {erank_ratio_2.item():.2%}")
print(f"Tracker Effective Rank Ratio: {erank_ratio_2_tracker:.2%}\n")

# Test 3: Simulated 30% Output Target (7 Active Channels) with Tracker
features_bio = torch.zeros(B, C, H, W)
features_bio[:, :7, :, :] = torch.randn(B, 7, H, W)
norm_entropy_3, erank_ratio_3 = get_layer_entropy(features_bio)

tracker_3 = LayerEntropyTracker(C, device=features_bio.device)
tracker_3.update(features_bio)
norm_entropy_3_tracker, erank_ratio_3_tracker = tracker_3.compute_entropy()

print("Test 3: Simulated 30% Output Target (7 Active Channels) with Tracker")
print(f"Direct Normalized Entropy:   {norm_entropy_3.item():.2%}")
print(f"Tracker Normalized Entropy:  {norm_entropy_3_tracker:.2%}")
print(f"Direct Effective Rank Ratio: {erank_ratio_3.item():.2%}")
print(f"Tracker Effective Rank Ratio: {erank_ratio_3_tracker:.2%}\n")

# %% Usage
# # 1. Create a dictionary of trackers for the layers you care about
# trackers = {
#     'stage_1': StreamingLayerEntropy(channels=96),
#     'stage_2': StreamingLayerEntropy(channels=192),
#     'bottleneck': StreamingLayerEntropy(channels=768),
#     'seg_head': StreamingLayerEntropy(channels=4) # Assuming 4 classes
# }

# # 2. Define a hook generator so each hook knows which tracker to update
# def get_hook(layer_name):
#     def hook_fn(module, input, output):
#         trackers[layer_name].update(output)
#     return hook_fn

# # 3. Attach them to your network
# model.downsample_layers[0].register_forward_hook(get_hook('stage_1'))
# model.downsample_layers[1].register_forward_hook(get_hook('stage_2'))
# model.bottleneck.register_forward_hook(get_hook('bottleneck'))
# model.segmentation_head.register_forward_hook(get_hook('seg_head'))

# # 4. Run your test dataset ONCE. 
# # As the images flow through, the hooks silently update all matrices simultaneously.
# model.eval()
# for images, _ in test_dataloader:
#     _ = model(images) 

# # 5. Print the trajectory
# for layer_name, tracker in trackers.items():
#     norm_entropy, e_rank = tracker.compute_metrics()
#     print(f"{layer_name} -> Entropy: {norm_entropy:.2%}, Effective Rank: {e_rank:.2f}")

# %% Settings
experiment = 'exp01'
config = 'MT'
run_name = '260518:2102/train'
task_mode = 'combined' # 'disparity', 'segmentation', 'combined'

# 'MT-KD': '260520:0455/train'
# 'MT': '260518:2102/train'

# %% Load Model
mlflow.set_tracking_uri(f'/data/Zeitler/code/SIDE/mlruns_experiments/{experiment}')
mlflow_experiment = mlflow.get_experiment_by_name(config)
run_date = run_name.split('/')[0]
mlflow_run = mlflow.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f'run_name = "{run_date}"').iloc[0]

base_config_filepath = mlflow.artifacts.download_artifacts(run_id=mlflow_run.run_id, artifact_path='configs/base.yaml', dst_path='../.temp')
experiment_config_filepath = mlflow.artifacts.download_artifacts(run_id=mlflow_run.run_id, artifact_path=f'configs/{experiment}/{config}.yaml', dst_path='../.temp')

with open(base_config_filepath, 'r') as f: base_config = yaml.safe_load(f)
with open(experiment_config_filepath, 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)
config['logging']['notebook_mode'] = True

model_run_id = mlflow.search_runs(
    experiment_ids=[mlflow_experiment.experiment_id],
    filter_string=f'tags.mlflow.runName = "{run_name}"',
    order_by=['attributes.start_time DESC'],
    max_results=1
).iloc[0].run_id

model_path = f'runs:/{model_run_id}/best_model_{task_mode}'
model = mlflow.pytorch.load_model(model_path, map_location='cuda')
model.eval()

# %% Load Data
dataset_class = helpers.load(config['data']['dataset'])
transforms = build_transforms(config, mode='test')
dataset = dataset_class(
    mode='test',
    config=config,
    transforms=transforms,
)
dataloader = DataLoader(
    dataset,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=config['general']['num_workers'],
    pin_memory=config['general']['pin_memory'],
    persistent_workers=False
)
helpers.check_dataleakage('test', dataset)

# %%
trackers = {
    'encoder.stages_0': LayerEntropyTracker(channels=96),
    'encoder.stages_1': LayerEntropyTracker(channels=192),
    'encoder.stages_2': LayerEntropyTracker(channels=384),
    'encoder.stages_3': LayerEntropyTracker(channels=768),
    
    'decoders.segmentation.decoder.blocks.0': LayerEntropyTracker(channels=384),
    'decoders.segmentation.decoder.blocks.1': LayerEntropyTracker(channels=192),
    'decoders.segmentation.decoder.blocks.2': LayerEntropyTracker(channels=96),
    'decoders.segmentation.decoder.final_block': LayerEntropyTracker(channels=96),
    'decoders.segmentation.intercept_head': LayerEntropyTracker(channels=8),
    'decoders.segmentation.head': LayerEntropyTracker(channels=8),
    
    'decoders.disparity.decoder.blocks.0': LayerEntropyTracker(channels=384),
    'decoders.disparity.decoder.blocks.1': LayerEntropyTracker(channels=192),
    'decoders.disparity.decoder.blocks.2': LayerEntropyTracker(channels=96),
    'decoders.disparity.decoder.final_block': LayerEntropyTracker(channels=96),
    'decoders.disparity.intercept_head': LayerEntropyTracker(channels=128),
    #'decoders.disparity.head': LayerEntropyTracker(channels=1),
}

def get_hook(layer_name):
    def hook_fn(module, input, output):
        features = output[0] if isinstance(output, tuple) else output
        trackers[layer_name].update(features)
    return hook_fn

for name, module in model.named_modules():
    if name in trackers:
        module.register_forward_hook(get_hook(name))

# %% Run Dataloader

model.eval()
with torch.no_grad():
    for data in tqdm(dataloader):
        images = data['image'].to('cuda')
        right_images = data['right_image'].to('cuda') if 'right_image' in data else None
        _ = model(images, right_images)

# %% Compute and Print Entropy
print("\n--- Layer Entropy Results ---")
for layer_name, tracker in trackers.items():
    norm_entropy, e_rank = tracker.compute_entropy()
    print(f"{layer_name: <45} -> Normalized Entropy: {norm_entropy: >7.2%}, Effective Rank Ratio: {e_rank: >7.2%}")


# %%
