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

    # configs = [
    #     f'configs/{experiment_name}/SEG.yaml',
    #     f'configs/{experiment_name}/DISP.yaml',
    #     f'configs/{experiment_name}/wMT.yaml',
    #     f'configs/{experiment_name}/wMT-KD.yaml'
    # ]

    seeds = [42, 43, 44, 45, 46]

    mlflow_tracking_dir = f'./mlflow/{experiment_name}'

    env = os.environ.copy()
    env['MLFLOW_TRACKING_URI'] = f'file:{mlflow_tracking_dir}'
    
    for config_path in configs:
        for seed in seeds:
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