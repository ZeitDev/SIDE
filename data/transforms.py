from typing import Any, Dict, List, Optional
from utils.helpers import load
import albumentations as A
from albumentations.pytorch import ToTensorV2

pixel_transforms = {
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
    'Normalize'
    
    # Domain Adaptation
    'FDA', 'HistogramMatching', 'PixelDistributionAdaptation', 'TemplateTransform'
}

class IsolationCompose(A.Compose):
    def __init__(self, transforms, excluded_keys: List[str], p: float = 1.0, additional_targets: dict = None):
        super().__init__(transforms, p=p, additional_targets=additional_targets)
        self.excluded_keys = excluded_keys

    def __call__(self, **data):
        held_data = {k: data.pop(k) for k in self.excluded_keys if k in data}
        data = super().__call__(**data)
        data.update(held_data)
        
        return data

def build_transforms(transform_config: Optional[List[Dict[str, Any]]]) -> A.Compose:
    additional_targets = {
        'right_image': 'image',
        'segmentation': 'mask',
        'disparity': 'image'
    }
    
    def _instantiate(config):
        name = config['name']
        params = config.get('params', {}).copy() 
        if 'transforms' in params: params['transforms'] = [_instantiate(c) for c in params['transforms']]
        
        instance = load(f'albumentations.{name}', **params)
        
        if name in pixel_transforms: return IsolationCompose([instance], excluded_keys=['disparity'])
        else: return instance
    
    transforms = []
    if not transform_config:
        transforms.append(A.Resize(height=256, width=256))
        normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        transforms.append(IsolationCompose([normalize], excluded_keys=['disparity']))
    else:
        for transform_config_item in transform_config:
            transforms.append(_instantiate(transform_config_item))
            
    transforms.append(ToTensorV2())

    return A.Compose(transforms, additional_targets=additional_targets)