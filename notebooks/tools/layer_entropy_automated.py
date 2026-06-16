# %% Imports
import os, sys, gc
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import math
import torch
import yaml
import pandas as pd
from tqdm import tqdm

import mlflow
import mlflow.artifacts

from utils import helpers
from data.transforms import build_transforms
from torch.utils.data import DataLoader

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment()

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Restrict to GPU 1 for this notebook

# %% Helpers

class LayerEntropyTracker:
    def __init__(self, channels: int, device='cuda'):
        self.C = channels
        self.N = 0
        self.device = device
        self.sum_features = torch.zeros(channels, dtype=torch.float64, device=device)
        self.sum_features_squared = torch.zeros((channels, channels), dtype=torch.float64, device=device)
        
    @torch.no_grad()
    def update(self, features: torch.Tensor):
        features = features.permute(0, 2, 3, 1).reshape(-1, self.C)
        features = features.to(dtype=torch.float64, device=self.device)
        
        self.N += features.shape[0]
        self.sum_features += features.sum(dim=0)
        self.sum_features_squared += features.T @ features
        
    def compute_entropy(self):
        sum_outer_product = torch.outer(self.sum_features, self.sum_features) / self.N
        covariance_matrix = (self.sum_features_squared - sum_outer_product) / (self.N - 1)
        
        eigenvalues, _ = torch.linalg.eigh(covariance_matrix)
        eigenvalues = torch.clamp(eigenvalues, min=1e-12)
        
        singular_values = torch.sqrt(eigenvalues)
        probs = singular_values / torch.sum(singular_values)
        probs_safe = probs[probs > 1e-12]
        entropy = -torch.sum(probs_safe * torch.log(probs_safe)).item()
        
        max_entropy = math.log(self.C)
        normalized_entropy = entropy / max_entropy
        
        effective_rank = math.exp(entropy)
        effective_rank_ratio = effective_rank / self.C
        
        return normalized_entropy, effective_rank_ratio

# Helper to dynamically define which trackers we need based on config
def get_trackers_for_config(config_name):
    trackers = {
        'encoder.stages_0': LayerEntropyTracker(channels=96),
        'encoder.stages_1': LayerEntropyTracker(channels=192),
        'encoder.stages_2': LayerEntropyTracker(channels=384),
        'encoder.stages_3': LayerEntropyTracker(channels=768),
    }
    
    if config_name in ['SEG', 'MT', 'MT-KD']:
        trackers.update({
            'decoders.segmentation.decoder.blocks.0': LayerEntropyTracker(channels=384),
            'decoders.segmentation.decoder.blocks.1': LayerEntropyTracker(channels=192),
            'decoders.segmentation.decoder.blocks.2': LayerEntropyTracker(channels=96),
            'decoders.segmentation.decoder.final_block': LayerEntropyTracker(channels=96),
            'decoders.segmentation.intercept_head': LayerEntropyTracker(channels=8),
            'decoders.segmentation.head': LayerEntropyTracker(channels=8),
        })
        
    if config_name in ['DISP', 'MT', 'MT-KD']:
        trackers.update({
            'decoders.disparity.decoder.blocks.0': LayerEntropyTracker(channels=384),
            'decoders.disparity.decoder.blocks.1': LayerEntropyTracker(channels=192),
            'decoders.disparity.decoder.blocks.2': LayerEntropyTracker(channels=96),
            'decoders.disparity.decoder.final_block': LayerEntropyTracker(channels=96),
            'decoders.disparity.intercept_head': LayerEntropyTracker(channels=128),
            #'decoders.disparity.head': LayerEntropyTracker(channels=1),
        })
        
    return trackers


# %% Automation Setup

# 1. Just define the experiments and configs you want to process.
#    The script will automatically discover all 120+ runs inside them.
experiments_to_run = ['exp01'] # Add 'exp02', etc. when ready
configs_to_run = ['SEG', 'DISP', 'MT', 'MT-KD']

# Ensure temp directory exists for MLflow artifacts
os.makedirs('../.temp', exist_ok=True)

# List to hold all aggregated metrics across all runs
all_results = []

# %% Execution Loop

for experiment in experiments_to_run:
    mlflow.set_tracking_uri(f'/data/Zeitler/code/SIDE/mlruns_experiments/{experiment}')
    
    for config_name in configs_to_run:
        # Resolve task mode
        if config_name in ['MT', 'MT-KD']:
            task_mode = 'combined'
        elif config_name == 'SEG':
            task_mode = 'segmentation'
        elif config_name == 'DISP':
            task_mode = 'disparity'
        else:
            raise ValueError(f'Unknown config: {config_name}')

        mlflow_experiment = mlflow.get_experiment_by_name(config_name)
        if mlflow_experiment is None:
            print(f'Skipping {experiment}/{config_name} (Experiment not found in MLflow)')
            continue
            
        # Dynamically fetch ALL runs for this config
        df_runs = mlflow.search_runs(experiment_ids=[mlflow_experiment.experiment_id])
        if df_runs.empty or 'tags.mlflow.runName' not in df_runs.columns:
            print(f'Skipping {experiment}/{config_name} (No runs found)')
            continue
            
        # Extract unique base dates (e.g., '260510:1000') by splitting off any '/'
        all_run_names = df_runs['tags.mlflow.runName'].dropna()
        base_dates = sorted(list(set([name.split('/')[0] for name in all_run_names])))

        for run_date in base_dates:
            run_name_raw = f'{run_date}/train'
            run_name_test = f'{run_date}/test'
            identifier = f'{experiment}/{config_name}/{run_name_test}'
            
            # Check if the '/train' subrun actually exists in MLflow (skips early crashes)
            if not any(all_run_names == run_name_raw):
                print(f'Skipping {identifier} - No "/train" subrun found in MLflow.')
                continue
                
            print(f'\n{"="*60}')
            print(f'Processing: {identifier} | Task Mode: {task_mode}')
            print(f'{"="*60}')

            # ---------------------------
            # 1. Load MLflow & Configs
            # ---------------------------
            # Get the parent run to download configs (base.yaml, etc.)
            mlflow_parent_run = df_runs[df_runs['tags.mlflow.runName'] == run_date].iloc[0]

            base_config_filepath = mlflow.artifacts.download_artifacts(
                run_id=mlflow_parent_run.run_id, artifact_path='configs/base.yaml', dst_path='../.temp'
            )
            experiment_config_filepath = mlflow.artifacts.download_artifacts(
                run_id=mlflow_parent_run.run_id, artifact_path=f'configs/{experiment}/{config_name}.yaml', dst_path='../.temp'
            )

            with open(base_config_filepath, 'r') as f: base_config = yaml.safe_load(f)
            with open(experiment_config_filepath, 'r') as f: experiment_config = yaml.safe_load(f)
            
            config = helpers.deep_merge(experiment_config, base_config)
            config['logging']['notebook_mode'] = True

            # ---------------------------
            # 2. Load Model
            # ---------------------------
            # Get the specific child run ID to load the actual model weights
            model_run_id = df_runs[df_runs['tags.mlflow.runName'] == run_name_raw].iloc[0].run_id

            model_path = f'runs:/{model_run_id}/best_model_{task_mode}'
            model = mlflow.pytorch.load_model(model_path, map_location='cuda')
            model.eval()

            # ---------------------------
            # 3. Load Data
            # ---------------------------
            dataset_class = helpers.load(config['data']['dataset'])
            transforms = build_transforms(config, mode='test')
            dataset = dataset_class(mode='test', config=config, transforms=transforms)
            
            dataloader = DataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['general']['num_workers'],
                pin_memory=config['general']['pin_memory'],
                persistent_workers=False
            )
            helpers.check_dataleakage('test', dataset)

            # ---------------------------
            # 4. Attach Trackers & Hooks
            # ---------------------------
            trackers = get_trackers_for_config(config_name)
            hook_handles = []

            def get_hook(layer_name):
                def hook_fn(module, input, output):
                    features = output[0] if isinstance(output, tuple) else output
                    trackers[layer_name].update(features)
                return hook_fn

            for name, module in model.named_modules():
                if name in trackers:
                    handle = module.register_forward_hook(get_hook(name))
                    hook_handles.append(handle)

            # ---------------------------
            # 5. Run Inference
            # ---------------------------
            with torch.no_grad():
                for data in tqdm(dataloader, desc=f'Eval {run_name_test}'):
                    images = data['image'].to('cuda')
                    right_images = data['right_image'].to('cuda') if 'right_image' in data else None
                    _ = model(images, right_images)

            # ---------------------------
            # 6. Compute Metrics & Save
            # ---------------------------
            run_metrics = {
                'identifier': identifier,
                'experiment': experiment,
                'config': config_name,
                'run_name': run_name_test,
                'mode': 'test'
            }

            for layer_name, tracker in trackers.items():
                norm_entropy, e_rank = tracker.compute_entropy()
                run_metrics[f'{layer_name}_norm_entropy'] = norm_entropy
                run_metrics[f'{layer_name}_erank_ratio'] = e_rank

            all_results.append(run_metrics)

            # ---------------------------
            # 7. Cleanup & Memory Management
            # ---------------------------
            for handle in hook_handles:
                handle.remove()
            
            del model, dataset, dataloader, trackers
            gc.collect()
            torch.cuda.empty_cache()

# %% Save to Pickle DataFrame
df_results = pd.DataFrame(all_results)
save_path = './notebooks/evaluation/storage/entropy_metrics.pkl'
df_results.to_pickle(save_path)

print('\nFinished')
print(df_results.head())

# %%