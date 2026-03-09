import torch
import torch.nn as nn
from models.decoders.dynamic_unet import DecoderBlock
    
class SegmentationDecoder(nn.Module):
    """
    Missing explanations can be found in dynamic_unet.py
    """
    def __init__(self, all_n_encoder_channels, all_encoder_reductions,  **kwargs):
        super().__init__()
        self.all_n_decoder_channels = all_n_encoder_channels[::-1]
        self.all_decoder_increases = all_encoder_reductions[::-1]
        
        self.blocks = nn.ModuleList()
        for i in range(len(all_n_encoder_channels) - 1):
            n_decoder_channels = self.all_n_decoder_channels[i]
            n_skip_channels = self.all_n_decoder_channels[i+1]
            n_output_channels = self.all_n_decoder_channels[i+1]
            
            block = DecoderBlock(n_decoder_channels, n_skip_channels, n_output_channels)
            self.blocks.append(block)
            
        current_stride = self.all_decoder_increases[-1]
        if current_stride > 1:
            upsample_layers = []
            while current_stride > 1:
                upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                last_channels = self.all_n_decoder_channels[-1]
                upsample_layers.append(nn.Conv2d(last_channels, last_channels, kernel_size=3, padding=1))
                upsample_layers.append(nn.ReLU(inplace=True))
                current_stride //= 2
            self.final_block = nn.Sequential(*upsample_layers)
        else:
            self.final_block = nn.Identity()
            
    def forward(self, left_features, right_features=None):
        reversed_left_features = left_features[::-1]
        x = reversed_left_features[0]

        for i, block in enumerate(self.blocks):
            skip_features = reversed_left_features[i + 1]
            x = block(x, skip_features)
                
        x = self.final_block(x)
        return x
         