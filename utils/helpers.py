import os
import mlflow
import logging
import importlib
import collections.abc
from typing import cast, Dict, Any

import torch
import torch.nn.functional as F

from utils.logger import CustomLogger
logger = cast(CustomLogger, logging.getLogger(__name__))

def load(class_string, **kwargs):
    """
    Dynamically loads a class from a string.
    Format: "module_path.ClassName" or just "ClassName" for standard packages
    Lazy workaround to dynamically load classes from strings
    Recommended to use decorators or registries for larger projects
    Args:
        class_string: String representation of the class
        **kwargs: Parameters to pass to the constructor
        
    Returns:
        Instantiated object
    """
    
    if '.' not in class_string:
        raise ValueError('class_string must be in the format "module_path.ClassName"')

    module_path, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_path)
    class_obj = getattr(module, class_name)
        
    if not kwargs: 
        return class_obj
    
    return class_obj(**kwargs)

def deep_merge(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges source config into destination config."""
    for key, value in source.items():
        if isinstance(value, dict) and key != 'params':
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination

def mlflow_log_misc(log_filepath: str) -> None:    
    mlflow.log_artifact(log_filepath, artifact_path='logs')
    mlflow.log_artifact(os.path.join(os.path.dirname(__file__), '..', 'main.py'))
    for folder in ['configs', 'criterions', 'data', 'metrics', 'models', 'processors', 'utils']:
        if os.path.isdir(folder):
            mlflow.log_artifacts(folder, artifact_path=folder)
            
def mlflow_log_run(config: Dict[str, Any], tags: Dict[str, str]) -> None:
    flat_config = _flatten_config(config)
    mlflow.log_params(flat_config)
    mlflow.set_tags(tags)
            
def _flatten_config(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            if not v:
                items.append((new_key, v))
            else:
                items.extend(_flatten_config(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if not v:
                items.append((new_key, v))
            else:
                for i, item in enumerate(v):
                    list_key = f'{new_key}{sep}{i}'
                    if isinstance(item, collections.abc.MutableMapping):
                        items.extend(_flatten_config(item, list_key, sep=sep).items())
                    else:
                        items.append((list_key, item))
        else:
            items.append((new_key, v))
    return dict(items)

def log_vram(stage: str = ''):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        logger.vram(f'[{stage}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Total: {total:.2f}GB')
        
def get_state_run_id(state_path: str) -> str:
    model_state_parts = state_path.split('/')
    experiment = model_state_parts[0]
    run_path = '/'.join(model_state_parts[1:])
    
    mlflow_experiment = mlflow.get_experiment_by_name(experiment)
    
    model_run_id = mlflow.search_runs(
        experiment_ids=[mlflow_experiment.experiment_id], 
        filter_string=f"tags.mlflow.runName = '{run_path}'", 
        order_by=["attributes.start_time DESC"], 
        max_results=1
    ).iloc[0].run_id  
        
    return model_run_id

def soft_argmin(outputs: torch.Tensor) -> torch.Tensor:
    output_probabilities = F.softmax(outputs, dim=1)
    
    B, D, H, W = output_probabilities.shape
    disparity_indices = torch.arange(1, D + 1, device=output_probabilities.device, dtype=output_probabilities.dtype, requires_grad=False) # shift disparity bins by 1 because 0 == invalid
    disparity_indices = disparity_indices.view(1, D, 1, 1)
    
    predictions = torch.sum(output_probabilities * disparity_indices, dim=1, keepdim=True) / D
    print(predictions)
    
    return predictions