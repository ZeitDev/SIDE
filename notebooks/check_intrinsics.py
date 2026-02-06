# %%
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import json
from pathlib import Path
from setup import setup_environment
setup_environment()

# %%
dataset_path = Path('/data/Zeitler/SIDED/EndoVis17/processed')

original_calibrations = []
rectified_calibrations = []
foundation_stereo_calibrations = []
for mode in ['train', 'test']:
    sequences_path = dataset_path / mode
    for sequence in sorted(sequences_path.iterdir()):
        if sequence.is_dir():
            original_calibrations.append(sequence / 'original_calibration.txt')
            rectified_calibrations.append(sequence / 'input' / 'rectified_calibration.json')
            foundation_stereo_calibrations.append(sequence / 'input' / 'foundation_stereo_calibration.txt')
       
# %% Compare if diffs exist
def get_normalized_content(file_path):
    try:
        text = file_path.read_text().strip()
        # Normalize JSONs to ignore whitespace/ordering differences
        if file_path.suffix == '.json':
            return json.dumps(json.loads(text), sort_keys=True)
        # Normalize line endings for text files
        return text.replace('\r\n', '\n')
    except Exception as e:
        return f"Error: {e}"

def check_content_consistency(file_list, label):
    if not file_list:
        print(f"[{label}] No files found.")
        return

    print(f"[{label}] Checking consistency across {len(file_list)} files...")
    
    # Group files by unique content
    groups = {} 
    for file_path in file_list:
        content = get_normalized_content(file_path)
        if content not in groups:
            groups[content] = []
        groups[content].append(file_path)
        
    print(f"  Found {len(groups)} unique variations.")
    for i, (content, files) in enumerate(groups.items()):
        # Extract dataset name for display (handle direct vs input/ subdir)
        names = [f.parts[-3] if f.parts[-2] == 'input' else f.parts[-2] for f in files]
        # Print a limited list if too many files
        display_names = str(names) if len(names) < 10 else f"{names[:5]} ... and {len(names)-5} more"
        print(f"  Variation {i+1}: Shared by {len(files)} files: {display_names}")

check_content_consistency(original_calibrations, "Original")
check_content_consistency(rectified_calibrations, "Rectified")
check_content_consistency(foundation_stereo_calibrations, "Foundation")


# %%
