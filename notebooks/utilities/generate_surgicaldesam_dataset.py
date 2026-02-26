# %%
import os
import cv2

# %%
dataset_path = '/data/Zeitler/ExternalInput/processed'
output_path = '/data/Zeitler/code/Surgical-DeSAM/endovis17'
sequences = []

for mode in ['train', 'test']:
    mode_path = os.path.join(dataset_path, mode)
    for sequence in os.listdir(mode_path):
        sequence_path = os.path.join(mode_path, sequence)
        sequences.append(sequence_path)
            
# %%
for sequence in sorted(sequences):
    binary_masks_path = os.path.join(sequence, 'ground_truth', 'segmentation')
    binary_masks_output_path = os.path.join(output_path, os.path.basename(sequence), 'instrument_masks')
    os.makedirs(binary_masks_output_path, exist_ok=True)
    # load the png binary mask and save it as a jpg binary mask, replace 'image' with 'frame'
    for mask_name in os.listdir(binary_masks_path):
        mask_path = os.path.join(binary_masks_path, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        output_mask_name = mask_name.replace('image', 'frame')
        output_mask_path = os.path.join(binary_masks_output_path, output_mask_name)
        cv2.imwrite(output_mask_path, mask)
        # delete old jpg binary mask
        os.remove(output_mask_path.replace('png', 'jpg'))

    
    
# %%