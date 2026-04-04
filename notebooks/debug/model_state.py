# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import yaml

import mlflow
import mlflow.artifacts
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from data.transforms import build_transforms

import torch

from processors.tester import Tester
from models.manager import AttachHead


from utils import helpers
from utils.helpers import load, get_model_run_id, logits2disparity

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment(skip_cuda=True)


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./configs/base.yaml', 'r') as f: base_config = yaml.safe_load(f)
with open('./configs/ConvNext/MT-KD.yaml', 'r') as f: experiment_config = yaml.safe_load(f)
config = helpers.deep_merge(experiment_config, base_config)

# %%
encoder_config = config['training']['encoder']
EncoderClass = load(encoder_config['name'])
encoder = EncoderClass(**encoder_config['params'])
        
decoders = {}
tasks_config = config['training']['tasks']
for task, task_config in tasks_config.items():
    if task_config['enabled']:
        decoder_config = task_config['decoder']
        DecoderClass = load(decoder_config['name'])
        decoders[task] = AttachHead(
            decoder_class=DecoderClass,
            n_classes=config['data']['num_of_classes'][task],
            encoder_channels=encoder.feature_info.channels(), 
            encoder_reductions=encoder.feature_info.reduction(), 
            **decoder_config['params']
        )
            
model = load(
    'models.manager.Combiner', 
    encoder=encoder, 
    decoders=decoders
).to(device)

model_state_path = os.path.join('./.temp', 'model_state.pth')
state_dict = torch.load(model_state_path)
model.load_state_dict(state_dict['model_state_dict'])
model.to('cpu')
model.eval()
# %%
train_transforms = build_transforms(config, mode='train')
dataset_class = load(config['data']['dataset'])
dataset_train = dataset_class(
    mode='train',
    config=config,
    transforms=train_transforms,
)

signature_input_example = dataset_train[0]['image'].unsqueeze(0)
signature_input_example_right = dataset_train[0]['right_image'].unsqueeze(0)
            
with torch.no_grad():
    signature_output_example = model(signature_input_example, signature_input_example_right)
    signature_output_example = {k: v.numpy() for k, v in signature_output_example.items()}
    
signature = infer_signature(signature_input_example.numpy(), signature_output_example)

# %%
mlflow.set_experiment('debug')
with mlflow.start_run(run_name='save_broken_model_state2') as run:
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name=f'best_model_',
        code_paths=['models/'],
        signature=signature
    )

# %%
tester = Tester(config, run_id='c70171f3deff4290844756fe87f73f23')
test_metrics = tester.test()
[print(f'{k}: {v}') for k, v in test_metrics.items()]

# %%
