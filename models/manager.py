import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

class Combiner(nn.Module):
    def __init__(self, encoder: nn.Module, decoders: Dict[str, nn.Module]):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)
        
    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        outputs = {}
        left_features = self.encoder(left_image)
        for task, decoder in self.decoders.items():
            if task == 'segmentation':
                outputs['segmentation'] = decoder(left_features)
            if task == 'disparity':
                right_features = self.encoder(right_image)
                disparity_output = decoder(left_features, right_features)
                outputs['disparity'] = disparity_output['prediction']
                outputs['disparity_intercept_features'] = disparity_output['intercept_features']
            
        return outputs
    
class AttachHead(nn.Module):
    def __init__(self, decoder_class, n_classes, encoder_channels, encoder_reductions, **kwargs):
        super().__init__()
        self.decoder = decoder_class(encoder_channels, encoder_reductions, **kwargs)
        self.head = nn.Conv2d(self.decoder.all_n_decoder_channels[-1], n_classes, kernel_size=1)
        
        self.is_stereo = kwargs.get('is_stereo', False)
        if self.is_stereo:
            intercept_at = kwargs.get('intercept_at', 4)
            intercept_channels = kwargs.get('intercept_channels', 128)
            idx = self.decoder.all_decoder_increases.index(intercept_at)
            n_output_channels = self.decoder.all_n_decoder_channels[idx]
            
            self.intercept_head = nn.Sequential(
                nn.Conv2d(n_output_channels, intercept_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(intercept_channels, intercept_channels, kernel_size=1),
            )
            
    
    def forward(self, features, feature_right=None) -> torch.Tensor:
        if not self.is_stereo:
            x = self.decoder(features)
            x = self.head(x)
            
            return x
        else:
            x, intercept_features = self.decoder(features, feature_right)
            
            x = self.head(x)
            x = F.relu(x, inplace=True)
            intercept_features = self.intercept_head(intercept_features)
            
            return {
                'prediction': x,
                'intercept_features': intercept_features
            }