import os
import mlflow
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.loader import load
from typing import Dict, Any, List, Optional

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
    mlflow.log_artifact(__file__)
    mlflow.log_artifact(log_filepath, artifact_path='logs')
    for folder in ['configs', 'criterions', 'data', 'metrics', 'models', 'utils']:
        if os.path.isdir(folder):
            mlflow.log_artifacts(folder, artifact_path=folder)

def deep_merge(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges source dict into destination dict."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination