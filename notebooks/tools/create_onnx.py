# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml
import torch
from utils import helpers
from utils.setup import setup_environment
from utils.helpers import load
from models.manager import AttachHead, Combiner

os.chdir('/data/Zeitler/code/SIDE')

# %%
EXPERIMENT = 'exp01/MT-KD'

# Load config
with open('./configs/base.yaml', 'r') as f: base_config = yaml.safe_load(f)
with open(f'./configs/{EXPERIMENT}.yaml', 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)

# %%
def load_model_from_config(config: dict, device: str = 'cpu') -> Combiner:
    # 1. Load the Encoder
    encoder_config = config['training']['encoder']
    EncoderClass = load(encoder_config['name'])
    encoder = EncoderClass(**encoder_config['params'])
    
    # 2. Iterate and load the Decoders
    decoders = {}
    tasks_config = config['training']['tasks']
    for task, task_config in tasks_config.items():
        if task_config.get('enabled', False):
            decoder_config = task_config['decoder']
            DecoderClass = load(decoder_config['name'])
            decoders[task] = AttachHead(
                decoder_class=DecoderClass,
                n_classes=config['data']['num_of_classes'].get(task, 1),
                encoder_channels=encoder.feature_info.channels(), 
                encoder_reductions=encoder.feature_info.reduction(), 
                **decoder_config['params']
            )
            
    # 3. Combine into the multi-task model
    model = Combiner(encoder=encoder, decoders=decoders).to(device)
    model.eval()
    
    return model

model = load_model_from_config(config)

# %%
# Export to ONNX
# Create dummy inputs based on the batch size and image resolution (e.g., 256x512)
dummy_left_image = torch.randn(1, 3, 1024, 1024)
dummy_right_image = torch.randn(1, 3, 1024, 1024)

# %%
# Print model summary using torchinfo
from torchinfo import summary
summary(
    model, 
    input_data=(dummy_left_image, dummy_right_image), 
    depth=6,
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"]
)

# %%
onnx_path = f"notebooks/output/{EXPERIMENT.replace('/', '_')}_model.onnx"
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

print(f"Exporting model to {onnx_path}...")
torch.onnx.export(
    model,
    (dummy_left_image, dummy_right_image),
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['left_image', 'right_image'],
    output_names=list(model(dummy_left_image, dummy_right_image).keys()),
    dynamic_axes={
        'left_image': {0: 'batch_size', 2: 'height', 3: 'width'},
        'right_image': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)
print("ONNX export completed successfully!")

# %%
