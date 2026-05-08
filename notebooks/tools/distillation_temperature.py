# %%
import math
import torch
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
import pandas as pd

# %%
def entropy_confidence(logits, c_min=0, c_max=1, prob_dim=1):
    probs = F.softmax(logits, dim=prob_dim)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=prob_dim)
    num_bins = logits.shape[prob_dim]
    max_entropy = math.log2(num_bins)
    normalized_entropy = entropy / max_entropy
    base_confidence = 1.0 - normalized_entropy
    
    denominator = max(c_max - c_min, 1e-6)
    
    scaled_conf = (base_confidence - c_min) / denominator
    final_confidence = torch.clamp(scaled_conf, min=0.0, max=1.0)
    
    return final_confidence



image_records = []
#for task in ['disparity_128_256_256', 'segmentation_8_256_256', 'segmentation_2_256_256']:
for task in ['disparity_64_128_128', 'segmentation_8_128_128', 'segmentation_2_128_128']:
    for mode_name in ['train']:
        mode_path = Path(f'/data/Zeitler/SIDED/EndoVis17/processed/{mode_name}')
        if not mode_path.exists():
            continue
            
        for seq_path in mode_path.iterdir():
            if not seq_path.is_dir():
                continue
                
            seq = seq_path.name
            logits_path = seq_path / 'teacher' / task
            
            if not logits_path.exists():
                continue
                
            pt_files = list(logits_path.glob('*.pt'))
            
            for pt_file in tqdm(pt_files, desc=f"Collecting confidence {mode_name}/{seq}"):
                logits = torch.load(pt_file, map_location='cpu', weights_only=True).float()
                
                temperatures = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
                #temperatures = [2.2] if 'disparity' in task else [3.31]
                # if 'disparity' in task:
                #     temperatures = [1.66]
                # elif 'segmentation_2' in task:
                #     temperatures = [3.42]
                # elif 'segmentation_8' in task:
                #     temperatures = [3.2]
                    
                    
                for T in temperatures:
                    scaled_logits = logits / T 
    
                    if scaled_logits.dim() == 4:
                        conf = entropy_confidence(scaled_logits, prob_dim=1).squeeze(0)
                    elif scaled_logits.dim() == 3:
                        conf = entropy_confidence(scaled_logits, prob_dim=0).squeeze(0)
        
                    img_mean_conf = conf.mean().item()
                    image_records.append({
                        'Task': task,
                        'Temperature': T,
                        'Mean Confidence': img_mean_conf,
                        'Sequence': seq,
                        'Mode': mode_name
                    })
                
                
# %%
df = pd.DataFrame(image_records)
df = df[df['Mode'] == 'train']

summary = df.groupby(['Task', 'Temperature'])['Mean Confidence'].mean().reset_index()
pivot_summary = summary.pivot(index='Task', columns='Temperature', values='Mean Confidence')
print(pivot_summary)

# %%
'''
Train only
Temperature                  1.0       1.5       2.0       2.5       3.0       3.5       4.0
Task                                                                       
disparity_128_256_256   0.770954  0.715840  0.666788  0.617503  0.564569  0.507206  0.447254 
segmentation_2_256_256  0.978072  0.956652  0.911692  0.844661  0.765904  0.685089  0.608310 
segmentation_8_256_256  0.993996  0.981560  0.937401  0.853292  0.743663  0.628890  0.523107 

disparity_128_256_256 -> 1.66
segmentation_2_256_256 -> 3.41
segmentation_8_256_256 -> 3.19
___

512x512 train only
Temperature                  1.0       1.5       2.0       2.5       3.0       3.5       4.0
Task                                                                       
disparity_64_128_128    0.814454  0.763736  0.718225  0.673438  0.627571  0.579848  0.530494 
segmentation_2_128_128  0.965404  0.938410  0.890660  0.823676  0.746705  0.668383  0.594174
segmentation_8_128_128  0.988556  0.972200  0.923196  0.835277  0.724060  0.609683  0.505468

disparity_64_128_128 -> 2.20
segmentation_2_128_128 -> 3.30
segmentation_8_128_128 -> 3.11
'''

# %%
T_1 = 3.0
C_1 = 0.724060
T_2 = 3.5
C_2 = 0.609683
C_target = 0.7

T = T_1 + ((C_target - C_1) * (T_2 - T_1)) / (C_2 - C_1)

print(round(T, 2))