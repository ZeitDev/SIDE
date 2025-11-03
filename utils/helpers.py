import os
import mlflow
import importlib
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, List, Optional

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
        # ! 'disparity': 'mask'  ! ??????????? not tested
    }
    
    transforms = []
    if not transform_config:
        transforms.append(A.Resize(height=256, width=256))
        transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    else:
        for _transform_config in transform_config:
            transform_class = load(f'albumentations.{_transform_config["name"]}')
            params = _transform_config['params']
            transforms.append(transform_class(**params))
    transforms.append(ToTensorV2())

    return A.Compose(transforms, additional_targets=additional_targets)

def mlflow_log_run(config: Dict[str, Any], log_filepath: str) -> None:
    mlflow.log_params(config)
    mlflow.log_artifact(log_filepath, artifact_path='logs')
    mlflow.log_artifact(os.path.join(os.path.dirname(__file__), '..', 'main.py'))
    for folder in ['configs', 'criterions', 'data', 'metrics', 'models', 'utils']:
        if os.path.isdir(folder):
            mlflow.log_artifacts(folder, artifact_path=folder)

def deep_merge(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges source config into destination config."""
    for key, value in source.items():
        if isinstance(value, dict) and key != 'params':
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination