import torch
import torch.nn as nn



class ConvNeXtBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(1e-6 * torch.ones(channels, 1, 1))
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels * 4, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=True),
        )
        
    def forward(self, x):
        return x + self.gamma * self.block(x) # Residual connection with learnable scaling factor gamma
    
class DecoderBlock(nn.Module): # A single decoder block with upsampling and skip connections
    def __init__(self, n_input_channels, n_skip_channels, n_output_channels):
        super().__init__()
        self.gamma = nn.Parameter(1e-6 * torch.ones(n_output_channels, 1, 1))
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Depthwise convolution doesn't mix channels, so we need to mix it before
        self.mix_channels = nn.Conv2d(n_input_channels + n_skip_channels, n_output_channels, kernel_size=1, bias=False) 
        
        self.convnext_block = ConvNeXtBlock(n_output_channels)
        
    def forward(self, x, skip_features=None):
        x = self.up(x)
        
        if skip_features is not None:
            x = torch.cat([x, skip_features], dim=1)
            
        x = self.mix_channels(x)
        x = self.convnext_block(x) 
        
        return x
    
class LayerNorm2d(nn.Module):
    """Official LayerNorm implementation from the ConvNeXt paper (Channels First).
    Calculates the mean and variance strictly across the channel dimension, 
    perfectly preserving the spatial structure and contrast of the image.
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x