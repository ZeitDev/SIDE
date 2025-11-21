import torch.nn as nn

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