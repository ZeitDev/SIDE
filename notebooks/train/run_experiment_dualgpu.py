# Usage
# tmux new -s zeitler
# uv run notebooks/train/run_experiment_dualgpu.py --experiment exp05 --cuda_devices 0 1
# Detach with Ctrl+B, then D. Re-attach with `tmux attach -t zeitler`

import os
import time
import yaml
import subprocess
import tempfile
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

def run_single_task(config_path, seed, device_queue, env, root_dir):
    # Grab an available GPU from the queue
    device = device_queue.get()
    
    try:
        print(f'=== Running {config_path} with seed {seed} on GPU {device} ===')
        with open(config_path, 'r') as f: 
            exp_config = yaml.safe_load(f)
            
        if 'general' not in exp_config: 
            exp_config['general'] = {}
        exp_config['general']['seed'] = seed
        
        base_filename = os.path.basename(config_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_path = os.path.join(temp_dir, base_filename)
            with open(temp_config_path, 'w') as f: 
                yaml.dump(exp_config, f)
            
            subprocess.run(
                ['python', 'main.py', '--config', temp_config_path, '--cuda_device', str(device)], 
                env=env, 
                check=True,
                cwd=root_dir
            )
    except subprocess.CalledProcessError as e:
        print(f'Run failed for {config_path} with seed {seed}. Error: {e}')
    finally:
        # Important: Return the GPU to the queue for the next task!
        device_queue.put(device)
        time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    # Changed to accept multiple devices
    parser.add_argument('--cuda_devices', type=int, nargs='+', required=True)
    args = parser.parse_args()    
    
    experiment_name = args.experiment
    configs = [f'configs/{experiment_name}/{config_file}' for config_file in os.listdir(f'configs/{experiment_name}') if config_file.endswith('.yaml')]
    seeds = [42, 4242, 424242, 42424242, 4242424242] 

    mlflow_tracking_dir = f'./mlruns_experiments/{experiment_name}'
    env = os.environ.copy()
    env['MLFLOW_TRACKING_URI'] = f'file:{mlflow_tracking_dir}'
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Load devices into a thread-safe Queue
    device_queue = Queue()
    for d in args.cuda_devices:
        device_queue.put(d)

    # Flatten tasks
    tasks = [(config, seed) for config in configs for seed in seeds]

    # Run them using a ThreadPool (one thread per GPU)
    with ThreadPoolExecutor(max_workers=len(args.cuda_devices)) as executor:
        futures = []
        for config_path, seed in tasks:
            futures.append(
                executor.submit(run_single_task, config_path, seed, device_queue, env, root_dir)
            )
        
        # Wait for all runs to finish
        for future in as_completed(futures):
            future.result()

if __name__ == '__main__':
    main()