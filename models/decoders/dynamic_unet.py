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
    
class Decoder(nn.Module): # Assembles decoder blocks based on encoder channel list
    def __init__(self, all_n_encoder_channels, all_encoder_reductions, is_stereo=False):
        super().__init__()
        self.is_stereo = is_stereo
        
        # Channels
        # Encoder: [16, 24, 40, 112, 320] (Top -> Bottom)
        # Reversed: [320, 112, 40, 24, 16] (Bottom -> Top), Important for first decoder pass and skip connection features
        self.all_n_decoder_channels = all_n_encoder_channels[::-1]
        
        # Reductions
        # Encoder: [2, 4, 8, 16, 32] (Top -> Bottom)
        # Reversed: [32, 16, 8, 4, 2] (Bottom -> Top), Important for final upsampling to original input resolution
        self.all_decoder_increases = all_encoder_reductions[::-1]
        
        # 3. Build the Blocks
        self.blocks = nn.ModuleList()
        
        for i in range(len(all_n_encoder_channels) - 1):
            # For stereo input, double input channels for the first block to accommodate right image features
            # because both left and right features are concatenated
            # For other blocks, input channels remain the same
            n_input_multiplier = 2 if self.is_stereo and i == 0 else 1
            
            # Number of decoder channels for this block 
            # Mono [i==0: 320, i==1: 112, i==2: 40, i==3: 24]
            # Stereo [i==0: 640, i==1: 112, i==2: 40, i==3: 24]
            n_decoder_channels = self.all_n_decoder_channels[i] * n_input_multiplier
            
            # For stereo, always double skip connection channels because skip features come from both left and right images
            n_skip_multiplier = 2 if self.is_stereo else 1
            
            # Skip connection channels coming from the next layer of the reversed encoder (falsely named, but kept for consistency)
            # Mono [i==0: 112, i==1: 40, i==2: 24, i==3: 16]
            # Stereo [i==0: 224, i==1: 80, i==2: 48, i==3: 32]
            n_skip_channels = self.all_n_decoder_channels[i+1] * n_skip_multiplier
            
            # Expected number of output channels to prepare for the next block
            # Mono [i==0: 112, i==1: 40, i==2: 24, i==3: 16]
            # Stereo [i==0: 112, i==1: 40, i==2: 24, i==3: 16]
            n_output_channels = self.all_n_decoder_channels[i+1]
            
            block = DecoderBlock(n_input_channels=n_decoder_channels, n_skip_channels=n_skip_channels, n_output_channels=n_output_channels)
            self.blocks.append(block)

        # We need one final upsampling to reach the original input resolution, because the encoder downsamples one more time than we have decoder blocks
        # Usually scale_factor=2 because encoder strides by 2, but depends on encoder architecture
        if self.all_decoder_increases[-1] >= 2:
            current_stride = self.all_decoder_increases[-1]
            
            upsample_layers = []
            while current_stride > 1:
                # Each upsampling step doubles the spatial dimensions
                upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                # Sharpens features after upsampling
                upsample_layers.append(nn.Conv2d(self.all_n_decoder_channels[-1], self.all_n_decoder_channels[-1], kernel_size=3, padding=1))
                upsample_layers.append(nn.ReLU(inplace=True))
                # Halve the current stride until we reach original resolution
                current_stride //= 2
                
            self.final_block = nn.Sequential(*upsample_layers)
        else:
            self.final_block = nn.Identity()

    def forward(self, encoder_features, encoder_features_right=None):
        # Reverse encoder features to start from the deepest layer
        reversed_encoder_features = encoder_features[::-1]
        if not self.is_stereo:
            # Start with the deepest features
            x = reversed_encoder_features[0]
        else:
            # stereo: concatenate left and right features along channel dimension
            reversed_encoder_features_right = encoder_features_right[::-1]
            x = torch.cat([reversed_encoder_features[0], reversed_encoder_features_right[0]], dim=1)
            
        
        # Pass through each decoder block with corresponding skip features
        for i, block in enumerate(self.blocks):
            # Skip features come from the next layer in the reversed encoder features
            skip_features = reversed_encoder_features[i + 1]
            if self.is_stereo:
                skip_features_right = reversed_encoder_features_right[i + 1]
                # Stereo: concate left and right skip features along channel dimension
                skip_features = torch.cat([skip_features, skip_features_right], dim=1)
            
            x = block(x, skip_features)
            
        # Final upsampling to original resolution    
        x = self.final_block(x)
        
        return x