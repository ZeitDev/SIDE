import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.decoders.dynamic_unet import DecoderBlock
from models.decoders.unext import DecoderBlock, LayerNorm2d, ConvNeXtBlock

class CorrelationVolume(nn.Module):
    def __init__(self, max_displacement: int):
        super().__init__()
        self.max_displacement = max_displacement
        
    def forward(self, left_features: torch.Tensor, right_features: torch.Tensor) -> torch.Tensor:
        # Normalize features to ensure that the correlation is based on cosine similarity rather than raw dot product, which can be influenced by feature magnitudes
        left_features = F.normalize(left_features, p=2, dim=1)
        right_features = F.normalize(right_features, p=2, dim=1)
        
        B, C, H, W = left_features.shape
        volume = []
        
        for i in range(self.max_displacement):
            if i == 0:
                # No shift at origin
                shifted_right = right_features
            else:
                # Discard the right most i columns and pad 0s on the left to maintain spatial dimensions, essentially shifting the right features to the right by i pixels
                shifted_right = F.pad(right_features[:, :, :, :-i], (i, 0))
                
            # Element-wise multiplication followed by summation over the channel dimension to get the similarity map for this displacement (B, 1, H, W)
            correlation = (left_features * shifted_right).sum(dim=1, keepdim=True)
            volume.append(correlation)
            
        # Concatenate the list of similarity maps (volume: B, 1, H, W) along the channel dimension to get the final correlation volume (B, max_displacement, H, W)    
        return torch.cat(volume, dim=1)
    
class DisparityDecoder(nn.Module):
    """
    Missing explanations can be found in dynamic_unet.py
    """
    def __init__(self, all_n_encoder_channels, all_encoder_reductions, max_disparity: int = 512, intercept_at: int = 4,  **kwargs):
        super().__init__()
        self.intercept_at = intercept_at
        
        self.all_n_decoder_channels = all_n_encoder_channels[::-1]
        self.all_decoder_increases = all_encoder_reductions[::-1]
        
        # The last encoder reduction or first decoder increase determines the max displacement at the bottleneck for the correlation volume
        max_displacement = max_disparity // self.all_decoder_increases[0] # 512 // 32 = 16
        self.correlation_volume = CorrelationVolume(max_displacement)
        
        # Equalize left features and cost volume before concatenation to prevent one from dominating the other
        self.squeeze_left = nn.Sequential(
            nn.Conv2d(self.all_n_decoder_channels[0], max_displacement, kernel_size=1, bias=False),
            LayerNorm2d(max_displacement),
            nn.GELU()
        )
        
        # Adapter reduced the [Left Features + Cost Volume] channels back to the original number of decoder channels expected by the rest of the decoder blocks
        self.cost_volume_adapter = nn.Sequential(
            nn.Conv2d(max_displacement + max_displacement, self.all_n_decoder_channels[0], kernel_size=3, padding=1, bias=False),
            LayerNorm2d(self.all_n_decoder_channels[0]),
            nn.GELU(),
        )
        
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
            last_channels = self.all_n_decoder_channels[-1]
            while current_stride > 1:
                upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                upsample_layers.append(ConvNeXtBlock(last_channels))
                current_stride //= 2
                
            self.final_block = nn.Sequential(*upsample_layers)
        else:
            self.final_block = nn.Identity()
            
    def forward(self, left_features, right_features):
        reversed_left_features = left_features[::-1]
        reversed_right_features = right_features[::-1]
        
        # Cost Correlation Volume at the bottleneck
        cost_volume = self.correlation_volume(reversed_left_features[0], reversed_right_features[0])
        squeezed_left = self.squeeze_left(reversed_left_features[0])
        
        x = torch.cat([squeezed_left, cost_volume], dim=1)
        x = self.cost_volume_adapter(x)
        
        intercept_features = None
        for i, block in enumerate(self.blocks):
            skip_features = reversed_left_features[i + 1]
            x = block(x, skip_features)
            
            if self.all_decoder_increases[i+1] == self.intercept_at:
                intercept_features = x
                
        x = self.final_block(x)
        return x, intercept_features
         