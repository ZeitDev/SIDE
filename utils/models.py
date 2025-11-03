from torch import nn

class Combiner(nn.Module):
    """
    A modular model that combines an encoder with one or more decoders.
    """
    def __init__(self, encoder, decoders):
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, x):
        """
        Passes the input through the encoder and then through each decoder.
        Returns a dictionary of outputs, with keys corresponding to the decoder tasks.
        """
        features = self.encoder(x)
        outputs = {}
        for task, decoder in self.decoders.items(): 
            outputs[task] = decoder(features)
        
        return outputs
    
class Decombiner(nn.Module):
    """
    A wrapper model that extracts a specific output from a multi-task model.
    Needed for saving models with mlflow.
    """
    def __init__(self, model, output_task_key):
        super().__init__()
        self.model = model
        self.output_task_key = output_task_key

    def forward(self, x):
        output_dict = self.model(x)
        return output_dict[self.output_task_key]