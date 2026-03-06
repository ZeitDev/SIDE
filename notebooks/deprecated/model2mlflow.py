# %%
import datetime
import torch
from mlflow.models.signature import infer_signature
import mlflow
from monai.networks.nets import SwinUNETR

# %%
class DisparityWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, left_img, right_img):
        x = torch.cat([left_img, right_img], dim=1)
        return self.model(x)
    
# %%
mode = 'disparity' # 'segmentation' or 'disparity'

signature_input_example = torch.rand(1, 3, 512, 512)

model = SwinUNETR(
    in_channels=signature_input_example.shape[1] if mode == 'segmentation' else signature_input_example.shape[1] * 2,
    out_channels=1,
    use_checkpoint=True,
    spatial_dims=2,
    use_v2=True,
    feature_size=48,
)
if mode == 'disparity':
    model = DisparityWrapper(model)
    
model.to('cpu')
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"Model Parameters: {num_params / 1e6:.2f} M")

# %%
mlflow.set_tracking_uri('/data/Zeitler/code/SIDE/mlruns')
mlflow.set_experiment(f'teacher::{mode}')
run_datetime = datetime.datetime.now().strftime("%y%m%d:%H%M")
with mlflow.start_run(run_name=run_datetime) as run:
    with torch.no_grad():
        if mode == 'segmentation': 
            signature_output_example = model(signature_input_example)
        elif mode == 'disparity':
            signature_output_example = model(signature_input_example, signature_input_example)
                
    signature = infer_signature(signature_input_example.numpy(), signature_output_example.numpy())
    mlflow.pytorch.log_model( 
        pytorch_model=model,
        name='best_model',
        signature=signature
    )
    
# %%