# %%
import os, sys
sys.path.append(os.path.dirname('/data/Zeitler/code/SIDE/'))

import glob
import subprocess
import shutil
from pathlib import Path

from utils.setup import setup_environment
os.chdir('/data/Zeitler/code/SIDE')
setup_environment()

# %%
# 1. Set your run IDs here
GREEN_RUN_ID = '9f50e6d77e3a41168260851abd2a8392'
RED_RUN_ID = 'de7b5afc26c549aaa981c240f765e374'


# %%
def find_run_dir(mlruns_path, run_id):
    paths = glob.glob(f'{mlruns_path}/*/{run_id}')
    return paths[0]

MLRUNS_DIR = './mlruns'
good_run_dir = find_run_dir(MLRUNS_DIR, RED_RUN_ID)
bad_run_dir = find_run_dir(MLRUNS_DIR, GREEN_RUN_ID)

good_artifacts = os.path.join(good_run_dir, 'artifacts')
bad_artifacts = os.path.join(bad_run_dir, 'artifacts')

temp_path = Path('./.temp')
temp_path.mkdir(exist_ok=True)

# %%
subprocess.run(['git', 'init'], cwd=temp_path, capture_output=True)
subprocess.run(['git', 'config', 'user.email', 'diff@example.com'], cwd=temp_path, capture_output=True)
subprocess.run(['git', 'config', 'user.name', 'Diff'], cwd=temp_path, capture_output=True)

shutil.copytree(good_artifacts, temp_path / 'artifacts', dirs_exist_ok=True)
subprocess.run(['git', 'add', '.'], cwd=temp_path, capture_output=True)
subprocess.run(['git', 'commit', '-m', 'good run'], cwd=temp_path, capture_output=True)

shutil.rmtree(temp_path / 'artifacts')
shutil.copytree(bad_artifacts, temp_path / 'artifacts', dirs_exist_ok=True)

subprocess.run(['git', 'add', '.'], cwd=temp_path, capture_output=True)
result = subprocess.run(['git', 'diff', '--staged'], cwd=temp_path, capture_output=True, text=True)

diff_output = result.stdout


# %%
# Display and save the diff
print(diff_output)

# Save it to a file for easier analysis with VS Code's syntax highlighting
diff_file = './.temp/run_artifacts_diff.patch'
with open(diff_file, 'w') as f:
    f.write(diff_output)


# %%
