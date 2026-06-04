# Usage
# tmux new -s zeitler
# uv run notebooks/train/run_experiment.py --experiment exp08 --cuda_device 0
# Detach with Ctrl+B, then D. Re-attach with `tmux attach -t zeitler`

# 06: MT-KD 424242, 4242, 42 finished | MT-KD 42424242, 4242424242 left
# 07: MT 424242, 4242, 42, finished | MT 42424242, 4242424242 left | MT-KD 42, 4242, 424242, 42424242, 4242424242 left


import os
import time
import yaml
import subprocess
import tempfile
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--cuda_device', type=int, required=True)
    args = parser.parse_args()    
    
    experiment_name = args.experiment
    
    configs = [f'configs/{experiment_name}/{config_file}' for config_file in os.listdir(f'configs/{experiment_name}') if config_file.endswith('.yaml')]

    seeds = [42, 4242, 424242, 42424242, 4242424242] # the meaning of life cant be larger than 32-bit integer overflow after all, right? :D

    # * TEMPORARY FIX, as the workstation crashed
    # completed_seeds = {
    #     'exp06': {
    #         'MT-KD': [424242, 4242, 42],
    #     },
    #     'exp07': {
    #         'MT': [424242, 4242, 42],
    #     }
    # }

    mlflow_tracking_dir = f'./mlruns_experiments/{experiment_name}'

    env = os.environ.copy()
    env['MLFLOW_TRACKING_URI'] = f'file:{mlflow_tracking_dir}'
    
    for config_path in configs:
        for seed in seeds:
            # * TEMPORARY FIX
            # if seed in completed_seeds.get(experiment_name, {}).get(os.path.basename(config_path).replace('.yaml', ''), []):
            #     print(f'Skipping {config_path} with seed {seed} as it is already completed.')
            #     continue
            
            print(f'=== Running {config_path} with seed {seed} ===')
            
            with open(config_path, 'r') as f: exp_config = yaml.safe_load(f)
            if 'general' not in exp_config: exp_config['general'] = {}
            exp_config['general']['seed'] = seed
            
            base_filename = os.path.basename(config_path)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_config_path = os.path.join(temp_dir, base_filename)
                with open(temp_config_path, 'w') as f: 
                    yaml.dump(exp_config, f)
                
                try:
                    subprocess.run(
                        ['python', 'main.py', '--config', temp_config_path, '--cuda_device', str(args.cuda_device)], 
                        env=env, 
                        check=True,
                        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                    )
                except subprocess.CalledProcessError as e:
                    print(f'Run failed for {config_path} with seed {seed}. Error: {e}')
            
            time.sleep(2)

if __name__ == '__main__':
    main()