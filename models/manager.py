import torch
import torch.nn as nn

from typing import Dict

class Combiner(nn.Module):
    def __init__(self, encoder, decoders):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)
        
    def forward(self, left_image, right_image=None) -> Dict[str, torch.Tensor]:
        outputs = {}
        for task, decoder in self.decoders.items():
            left_features = self.encoder(left_image)
            if task == 'segmentation':
                outputs[task] = decoder(left_features)
            if task == 'disparity':
                right_features = self.encoder(right_image)
                outputs[task] = decoder(left_features, right_features)
            
        return outputs
    
class AttachHead(nn.Module):
    def __init__(self, decoder_class, n_classes, encoder_channels, encoder_reductions, **kwargs):
        super().__init__()
        self.decoder = decoder_class(encoder_channels, encoder_reductions, **kwargs)
        self.head = nn.Conv2d(self.decoder.all_n_decoder_channels[-1], n_classes, kernel_size=1)
    
    def forward(self, features, feature_right=None) -> torch.Tensor:
        x = self.decoder(features, feature_right)
        x = self.head(x)
        
        return x