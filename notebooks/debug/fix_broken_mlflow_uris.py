# %%
import os
from pathlib import Path

# Set your target directory and strings
old_dir_name = 'exp06'
new_dir_name = 'exp07'

tracking_dir = Path(f'/data/Zeitler/code/SIDE/mlruns_experiments/{new_dir_name}')

# Find all meta.yaml files recursively
yaml_files = list(tracking_dir.rglob('meta.yaml'))
modified_count = 0

for yaml_path in yaml_files:
    with open(yaml_path, 'r') as file:
        content = file.read()

    # Only rewrite if the old path actually exists in this file
    if old_dir_name in content:
        # Replace the specific directory name in the file string
        new_content = content.replace(f'/{old_dir_name}/', f'/{new_dir_name}/')
        
        with open(yaml_path, 'w') as file:
            file.write(new_content)
        modified_count += 1

print(f'Update complete. Modified {modified_count} meta.yaml files.')

# %%