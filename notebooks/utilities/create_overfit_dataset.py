# %%
from pathlib import Path

# %%
dataset_path = Path('/data/Zeitler/SIDED/EndoVis17/processed')
overfit_path = Path('/data/Zeitler/SIDED/OverfitDataset')


for mode in ['train', 'test']:
    source_subset = dataset_path / 'train' / 'instrument_dataset_1'
    target_subsets = [overfit_path / mode / f'subset_{i}' for i in [1, 2]]
    
    source_left_images_path = source_subset / 'input' / 'left_images'
    source_right_images_path = source_subset / 'input' / 'right_images'
    source_segmentation_path = source_subset / 'ground_truth' / 'segmentation'
    source_disparity_path = source_subset / 'ground_truth' / 'disparity'
    source_calibration_path = source_subset / 'calibration' / 'rectified_calibration.json'
    source_mapping_path = dataset_path / 'mapping.json'
    target_mapping_path = overfit_path / 'mapping.json'
    target_mapping_path.write_bytes(source_mapping_path.read_bytes())
        
    file_names = sorted([f for f in source_left_images_path.iterdir() if f.is_file()])
    selected_file_names = file_names[::len(file_names)//8][:8]  
    
    for target_subset in target_subsets:
        (target_subset / 'input' / 'left_images').mkdir(parents=True, exist_ok=True)
        (target_subset / 'input' / 'right_images').mkdir(parents=True, exist_ok=True)
        (target_subset / 'ground_truth' / 'segmentation').mkdir(parents=True, exist_ok=True)
        (target_subset / 'ground_truth' / 'disparity').mkdir(parents=True, exist_ok=True)
        
        target_calibration_path = target_subset / 'calibration' / 'rectified_calibration.json'
        target_calibration_path.parent.mkdir(parents=True, exist_ok=True)
        target_calibration_path.write_bytes(source_calibration_path.read_bytes())
        
        for selected_file_name in selected_file_names:
            target_left_image_path = target_subset / 'input' / 'left_images' / selected_file_name.name
            target_right_image_path = target_subset / 'input' / 'right_images' / selected_file_name.name
            target_segmentation_path = target_subset / 'ground_truth' / 'segmentation' / selected_file_name.name
            target_disparity_path = target_subset / 'ground_truth' / 'disparity' / selected_file_name.name
            
            # Copy files to the new location
            source_left_image = (source_left_images_path / selected_file_name.name).read_bytes()
            source_right_image = (source_right_images_path / selected_file_name.name).read_bytes()
            source_segmentation = (source_segmentation_path / selected_file_name.name).read_bytes()
            source_disparity = (source_disparity_path / selected_file_name.name).read_bytes()
            
            target_left_image_path.write_bytes(source_left_image)
            target_right_image_path.write_bytes(source_right_image)
            target_segmentation_path.write_bytes(source_segmentation)
            target_disparity_path.write_bytes(source_disparity)
        
        
# %%
