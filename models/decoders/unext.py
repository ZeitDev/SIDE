import torch
import torch.nn as nn

class DecoderBlock(nn.Module): # A single decoder block with upsampling and skip connections
    def __init__(self, n_input_channels, n_skip_channels, n_output_channels):
        super().__init__()
        
        # deviating from U-Net paper using nn.ConvTranspose2d to avoid checkerboard artifacts
        # bilinear for smooth slopes to descent on (makes optimization easier vs nearest which is steppy)
        # align_corner to prevent pixel shift
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        
        self.double_conv = nn.Sequential(
            # n_input_channels + n_skip_channels: number of channels from previous layer
            # out_channels: desired output channels after convolution
            # kernel_size=3x3 and padding=1: Maintain spatial dimensions, 2x 3x3 convs gives same receptive field than 1x 5x5 , but with less parameters
            # bias false: because of BatchNorm adding its own bias
            nn.Conv2d(n_input_channels + n_skip_channels, n_output_channels, kernel_size=3, padding=1, bias=False), 
            # prevent neural network being a drama queen (exploding/vanishing gradients)
            nn.BatchNorm2d(n_output_channels), 
            # Non-linearity
            # inplace saves memory
            nn.ReLU(inplace=True), 
            # Conv again, because U-Net paper uses double convs in each block
            # Behaves like giving the model more thinking steps
            # Retain n_output_channels as first conv already reduced channels
            nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_output_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, skip_features=None):
        x = self.up(x)
        
        if skip_features is not None: # if skip connection exists
            # Concat upsampled features with skip connection features along channel dimension
            # Expects both tensors to have same spatial dimensions -> divisible by 2 accordingly
            x = torch.cat([x, skip_features], dim=1)
            
        x = self.double_conv(x)
        
        return x