from typing import Any, Dict, List
from utils.helpers import load
import albumentations as A
from albumentations.pytorch import ToTensorV2

pixel_transforms = [
    # Color & Brightness
    'RandomBrightnessContrast', 'RGBShift', 'HueSaturationValue', 
    'RandomGamma', 'CLAHE', 'ChannelShuffle', 'Invert', 
    'ToGray', 'ToSepia', 'Solarize', 'Posterize', 'Equalize', 
    'FancyPCA', 'ColorJitter', 'RandomToneCurve',
        
    # Blur & Sharpen
    'GaussianBlur', 'MotionBlur', 'MedianBlur', 'Blur', 
    'GlassBlur', 'ZoomBlur', 'Defocus', 'Sharpen', 'Emboss',
        
    # Noise
    'GaussNoise', 'ISONoise', 'MultiplicativeNoise',
        
    # Weather & Effects
    'RandomFog', 'RandomRain', 'RandomShadow', 'RandomSnow', 
    'RandomSunFlare',
        
    # Quality & Dropout
    'CoarseDropout', 'PixelDropout', 'ImageCompression', 
    'Downscale', 'Superpixels', 'RingingOvershoot', 
    'Normalize',
    
    # Domain Adaptation
    'FDA', 'HistogramMatching', 'PixelDistributionAdaptation', 'TemplateTransform'
]

class IsolationCompose(A.Compose):
    def __init__(self, transforms, excluded_keys: List[str], p: float = 1.0, additional_targets: dict = None, renaming: dict = None):
        if additional_targets: additional_targets = {k: v for k, v in additional_targets.items() if k not in excluded_keys}
        super().__init__(transforms, p=p, additional_targets=additional_targets)
        self.excluded_keys = excluded_keys
        self.renaming = renaming or {}

    def __call__(self, **data):
        held_data = {k: data.pop(k) for k in self.excluded_keys if k in data}
        print('transforms', self.transforms, 'data', data.keys(), 'held_data', held_data.keys())
        
        if data:
            for old_key, new_key in self.renaming.items():
                data[new_key] = data.pop(old_key)
            
            data = super().__call__(**data)
            
            for old_key, new_key in self.renaming.items():
                data[old_key] = data.pop(new_key)
            
        data.update(held_data)
        
        return data
    

def build_transforms(config: Dict[str, Any], mode: str = 'train') -> A.Compose:
    transform_config = config['data']['transforms'][mode]
    additional_targets = {
        'image': 'image',
        'right_image': 'image',
        'segmentation': 'mask',
        'disparity': 'image'
    }
    
    def _instantiate(transform_config_item: Dict[str, Any]) -> A.BasicTransform:
        name = transform_config_item['name']
        params = transform_config_item['params'].copy()
        if 'transforms' in params: params['transforms'] = [_instantiate(c) for c in params['transforms']]
        
        instance = load(f'albumentations.{name}', **params)
        
        if name in pixel_transforms: return IsolationCompose([instance], excluded_keys=['disparity'], additional_targets=additional_targets)
        else: return instance
    
    transforms = []
    if transform_config:
        for transform_config_item in transform_config:
            transforms.append(_instantiate(transform_config_item))
    else:
        transforms.append(A.Resize(height=256, width=256))
        normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        transforms.append(IsolationCompose([normalize], excluded_keys=['disparity'], additional_targets=additional_targets))
            
    # normalize = (img - mean) / (std * max_pixel_value) = img / max_pixel_value
    disparity_normalize = A.Normalize(mean=0.0, std=1.0, max_pixel_value=config['data']['max_disparity'])
    transforms.append(IsolationCompose(
        [disparity_normalize],
        excluded_keys=['image', 'right_image', 'segmentation'],
        additional_targets=additional_targets,
        renaming={'disparity': 'image'}))
    transforms.append(ToTensorV2())

    return A.Compose(transforms, additional_targets=additional_targets)