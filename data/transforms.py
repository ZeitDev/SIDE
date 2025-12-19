from typing import Any, Dict, List, Optional

from utils.helpers import load

import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(transform_config: Optional[List[Dict[str, Any]]]) -> A.Compose:
    additional_targets = {
        'right_image': 'image',
        'segmentation': 'mask',
        'disparity': 'image'
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