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
for task in ['disparity_64_128_128', 'segmentation_2_128_128']:
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
                
                temperatures = [1.0, 1.5, 2.0, 3.0, 4.0]
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
Complete dataset
Temperature                  1.0       1.5       2.0       3.0       4.0
Task                                                                    
disparity_128_256_256   0.773823  0.719746  0.671776  0.572094  0.457019
segmentation_2_256_256  0.973566  0.948038  0.904616  0.771096  0.622796
segmentation_8_256_256  0.989641  0.970135  0.911634  0.690573  0.466522
___
 
Train only
Temperature                  1.0       1.5       2.0       3.0       4.0
Task                                                                    
disparity_128_256_256   0.770954  0.715840  0.666788  0.564569  0.447254
segmentation_2_256_256  0.972196  0.945113  0.899848  0.763570  0.614543
segmentation_8_256_256  0.991238  0.972873  0.915289  0.695270  0.471549

disparity_128_256_256 -> 0.72 -> 1.5
segmentation_2_256_256 -> 0.69  -> 3.5
segmentation_8_256_256 -> 0.70 -> 3.0
___

512x512 train only
Temperature                  1.0       1.5       2.0       3.0       4.0
Task                                                                    
disparity_64_128_128    0.814454  0.763736  0.718225  0.627571  0.530494
segmentation_2_128_128  0.960634  0.928301  0.880438  0.745146  0.599935

disparity_64_128_128 -> 0.72 -> 2.0
segmentation_2_128_128 -> 0.67 -> 3.5
'''
