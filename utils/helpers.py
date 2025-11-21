import os
import mlflow
import importlib
import collections.abc
from typing import Dict, Any, List, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

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

def build_transforms(transform_config: Optional[List[Dict[str, Any]]]) -> A.Compose:
    additional_targets = {
        'right_image': 'image',
        'segmentation': 'mask',
        # ! 'disparity': 'image'  ! ??????????? not tested
    }
    
    def _instantiate(config):
        name = config['name']
        params = config.get('params', {}).copy() 
        if 'transforms' in params:
            params['transforms'] = [_instantiate(c) for c in params['transforms']]
            
        return load(f'albumentations.{name}', **params)
    
    transforms = []
    if not transform_config:
        transforms.append(A.Resize(height=256, width=256))
        transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    else:
        for _transform_config in transform_config:
            transforms.append(_instantiate(_transform_config))
    transforms.append(ToTensorV2())

    return A.Compose(transforms, additional_targets=additional_targets)

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
    for folder in ['configs', 'criterions', 'data', 'metrics', 'models', 'utils']:
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
                    list_key = f"{new_key}{sep}{i}"
                    if isinstance(item, collections.abc.MutableMapping):
                        items.extend(_flatten_config(item, list_key, sep=sep).items())
                    else:
                        items.append((list_key, item))
        else:
            items.append((new_key, v))
    return dict(items)