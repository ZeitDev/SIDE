import torch
import torch.nn as nn

class DecoderBlock(nn.Module): # A single decoder block with upsampling and skip concatenation
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
            # Double conv again, because U-Net paper uses double convs in each block
            # Behaves like giving the model more thinking steps
            # Retain n_output_channels as first conv already reduced channels
            nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_output_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, skip=None):
        x = self.up(x)
        
        if skip is not None: # if skip connection exists
            x = torch.cat([x, skip], dim=1) # concatenate skip features along channel dimension with upsampled features (x)
            
        x = self.double_conv(x)
        
        return x
    
class Decoder(nn.Module): # Assembles decoder blocks based on encoder channel list
    def __init__(self, all_n_encoder_channels, all_encoder_reductions):
        super().__init__()
        
        # Channels
        # Encoder: [16, 24, 40, 112, 320] (Top -> Bottom)
        # Reversed: [320, 112, 40, 24, 16] (Bottom -> Top), Important for first decoder pass and skip connection features
        self.all_n_decoder_channels = all_n_encoder_channels[::-1]
        
        # Encoder reductions reversed: [32, 16, 8, 4, 2]
        self.reversed_all_encoder_reductions = all_encoder_reductions[::-1]
        
        # 3. Build the Blocks
        self.blocks = nn.ModuleList()
        
        for i in range(len(all_n_encoder_channels) - 1):
            # Number of decoder channels for this block
            n_decoder_channels = self.all_n_decoder_channels[i] 
            
            # Skip connection channels coming from the next layer of the reversed encoder (falsely named, but kept for consistency)
            n_skip_channels = self.all_n_decoder_channels[i+1]
            
            # Expected number of output channels to prepare for the next block
            n_output_channels = self.all_n_decoder_channels[i+1]
            
            block = DecoderBlock(n_input_channels=n_decoder_channels, n_skip_channels=n_skip_channels, n_output_channels=n_output_channels)
            self.blocks.append(block)

        # We need one final upsampling to reach the original input resolution, because the encoder downsamples one more time than we have decoder blocks
        # Usually scale_factor=2 because encoder strides by 2, but depends on encoder architecture
        if self.reversed_all_encoder_reductions[-1] >= 2:
            current_stride = self.reversed_all_encoder_reductions[-1]
            
            upsample_layers = []
            while current_stride > 1:
                # Each upsampling step doubles the spatial dimensions
                upsample_layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                # Sharpens features after upsampling
                upsample_layers.append(nn.Conv2d(self.all_n_decoder_channels[-1], self.all_n_decoder_channels[-1], kernel_size=3, padding=1))
                upsample_layers.append(nn.ReLU(inplace=True))
                current_stride //= 2
                
            self.final_block = nn.Sequential(*upsample_layers)
        else:
            self.final_block = nn.Identity()

    def forward(self, encoder_features):
        reversed_encoder_features = encoder_features[::-1]
        
        x = reversed_encoder_features[0]
        
        for i, block in enumerate(self.blocks):
            skip = reversed_encoder_features[i + 1]
            x = block(x, skip)
            
        x = self.final_block(x)
        
        return x