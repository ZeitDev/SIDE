#

import os
import time
import yaml
import subprocess
import tempfile
import argparse
import queue
from concurrent.futures import ThreadPoolExecutor

def run_single_experiment(task, gpu_queue):
    exp_name, config_path, seed = task
    
    # Blocks until a GPU is available in the queue
    gpu_id = gpu_queue.get()
    
    env = os.environ.copy()
    mlflow_tracking_dir = f'./mlruns_experiments/{exp_name}'
    env['MLFLOW_TRACKING_URI'] = f'file:{mlflow_tracking_dir}'
    
    try:
        print(f'=== Running {config_path} with seed {seed} on GPU {gpu_id} ===')
        with open(config_path, 'r') as f: exp_config = yaml.safe_load(f)
        if 'general' not in exp_config: exp_config['general'] = {}
        exp_config['general']['seed'] = seed
        
        base_filename = os.path.basename(config_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_path = os.path.join(temp_dir, base_filename)
            with open(temp_config_path, 'w') as f: 
                yaml.dump(exp_config, f)
            
            subprocess.run(
                ['python', 'main.py', '--config', temp_config_path, '--cuda_device', str(gpu_id)], 
                env=env, 
                check=True,
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            )
    except subprocess.CalledProcessError as e:
        print(f'Run failed for {config_path} with seed {seed}. Error: {e}')
    finally:
        # Return the GPU to the queue for the next experiment
        gpu_queue.put(gpu_id)
        time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    
    seeds = [42, 4242, 424242, 42424242, 4242424242]
    
    # Setup thread-safe Queue populated with GPU IDs
    available_gpus = [0, 1]
    gpu_queue = queue.Queue()
    for gpu in available_gpus:
        gpu_queue.put(gpu)
    
    # Gather all tasks (config + seed combinations)
    tasks = []
    for exp_name in os.listdir('configs'):
        if 'exp' in exp_name:
            exp_dir = f'configs/{exp_name}'
            configs = [f'{exp_dir}/{f}' for f in os.listdir(exp_dir) if f.endswith('.yaml')]
            for c in configs:
                for s in seeds:
                    tasks.append((exp_name, c, s))

    # Run ThreadPool mapped to number of GPUs
    print(f'Starting {len(tasks)} tasks on {len(available_gpus)} GPUs...')
    with ThreadPoolExecutor(max_workers=len(available_gpus)) as executor:
        for task in tasks:
            executor.submit(run_single_experiment, task, gpu_queue)

if __name__ == '__main__':
    main()