# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# %% Print encoder feature shapes
model_name = 'efficientnet_b0'
encoder = timm.create_model(model_name, features_only=True, pretrained=True)
input = torch.randn(1, 3, 512, 512) # (B, C, H, W)

with torch.no_grad():
    features = encoder(input) 
    
for i, feature in enumerate(features):
    print(f'Feature {i}: shape = {feature.shape}')
    
# %%
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
    
# %%
class ModularDecoder(nn.Module): # Assembles decoder blocks based on encoder channel list
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
    
# %%
class UniversalFramework(nn.Module):
    def __init__(self, encoder_name='efficientnet_b0', num_classes=1):
        super().__init__()
        
        # features_only=True gives us access to intermediate feature maps
        # pretrained=True for transfer learning from ImageNet
        self.encoder = timm.create_model(encoder_name, features_only=True, pretrained=True)
        
        all_encoder_reductions = self.encoder.feature_info.reduction() # type: ignore
        print("Encoder reductions:", all_encoder_reductions)
        
        # Encoder: [16, 24, 40, 112, 320] (Top -> Bottom)
        all_n_encoder_channels = self.encoder.feature_info.channels() # type: ignore
        print("Encoder channels:", all_n_encoder_channels)
        
        # Decoder: [320, 112, 40, 24, 16] (Bottom -> Top)
        self.decoder = ModularDecoder(all_n_encoder_channels, all_encoder_reductions)
        print("Decoder channels:", self.decoder.all_n_decoder_channels)
        
        # We need to map the last decoder output [16] to the desired number of classes for the final output [8]
        self.head = nn.Conv2d(self.decoder.all_n_decoder_channels[-1], num_classes, kernel_size=1)
        
    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.head(x)
        return x

# %%
if __name__ == "__main__":
    print("Testing UniversalFramework...")
    encoder_name = 'convnextv2_tiny'
    image_size = 512
    num_classes = 8
    model = UniversalFramework(encoder_name=encoder_name, num_classes=num_classes)
    
    # Create dummy input (Batch Size, Channels, Height, Width)
    x = torch.randn(1, 3, image_size, image_size)
    
    # Forward pass
    y = model(x)
    
    expected_shape = (1, num_classes, image_size, image_size)
    
    print("\nModel:", encoder_name)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected output shape: {expected_shape}")
    
    # Verify output shape matches input spatial dimensions
    assert y.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {y.shape}"
    print("Test passed successfully!")

# %%
class Combiner(nn.Module):
    def __init__(self, encoder, decoders):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)
        
    def forward(self, x):
        features = self.encoder(x)
        outputs = {}
        for task, decoder in self.decoders.items():
            outputs[task] = decoder(features)
            
        return outputs
    
class AttachHead(nn.Module):
    def __init__(self, decoder_class, num_classes, encoder_channels, encoder_reductions, **kwargs):
        super().__init__()
        self.decoder = decoder_class(encoder_channels, encoder_reductions, **kwargs)
        self.head = nn.Conv2d(self.decoder.all_n_decoder_channels[-1], num_classes, kernel_size=1)
    
    def forward(self, features):
        x = self.decoder(features)
        x = self.head(x)
        return x

# File timm_encoder.py 
def TimmEncoder(encoder_name='efficientnet_b0', pretrained=True):
    encoder = timm.create_model(encoder_name, features_only=True, pretrained=pretrained)
    return encoder

# %%
encoder = TimmEncoder(encoder_name='resnet18', pretrained=True)
model = Combiner(
    encoder=encoder,
    decoders={
        'segmentation': AttachHead(
            decoder_class=ModularDecoder,
            num_classes=8,
            encoder_channels=encoder.feature_info.channels(), # type: ignore
            encoder_reductions=encoder.feature_info.reduction() # type: ignore
        ),
        'depth': AttachHead(
            decoder_class=ModularDecoder,
            num_classes=1,
            encoder_channels=encoder.feature_info.channels(), # type: ignore
            encoder_reductions=encoder.feature_info.reduction() # type: ignore
        )
    }
)

# Create dummy input (Batch Size, Channels, Height, Width)
x = torch.randn(1, 3, 512, 512)
outputs = model(x)
for task, output in outputs.items():
    print(f"Task: {task}, Output shape: {output.shape}")
    
# %%
# TODO: Do model saving, loading with mlflow
import mlflow
from mlflow.models.signature import infer_signature

dummy_input = torch.randn(1, 3, 512, 512) # Convert to numpy for MLflow

# 2. Run a forward pass to get dummy output
with torch.no_grad():
    dummy_output = model(dummy_input)
    # Convert dictionary outputs to numpy for inference
    dummy_output = {k: v.numpy() for k, v in dummy_output.items()}

# 3. Infer signature
signature = infer_signature(dummy_input, dummy_output)

with mlflow.start_run() as run:
    mlflow.pytorch.log_model( # type: ignore
        pytorch_model=model,
        name="modular_encoder_decoder_model",
        code_paths=["modular_encoder_decoder.py"],
        signature=signature
    )
    
loaded_model = mlflow.pytorch.load_model( # type: ignore
    model_uri=f"runs:/{run.info.run_id}/modular_encoder_decoder_model"
)
loaded_model.eval()
x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    outputs = loaded_model(x)
    
for task, output in outputs.items():
    print(f"Task: {task}, Output shape: {output.shape}")
    
# %%