# %%
import os, sys
sys.path.append(os.path.dirname('../../'))
os.chdir(os.path.dirname('../../'))

import glob
import pickle
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

# %%
mlflow_base_path = './mlruns_experiments'
exp_folders = sorted(glob.glob(os.path.join(mlflow_base_path, 'exp*')))

final_runs_data = []
params_runs_data = []
historic_runs_data = []

for exp_folder in exp_folders:
    tracking_uri = f'file://{os.path.abspath(exp_folder)}'
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    experiments = client.search_experiments()
    for experiment in experiments:
        exp_id = os.path.basename(exp_folder)
        if experiment.name == 'Default': continue # skip empty default experiment
        if exp_id in ['exp05_DEP', 'exp06_DEP']: continue 
        print(f'Loading from tracking URL: {exp_folder} -> experiment: {experiment.name} (ID: {experiment.experiment_id})')
            
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        for run in runs:
            run_id = run.info.run_id
            run_name = run.data.tags.get('mlflow.runName', 'Unnamed')
            if '/' not in run_name: continue # skip parent with no information
                
            # general meta data
            identifier = f'{exp_id}/{experiment.name}/{run_name}'
            run_info = {
                'identifier': identifier,
                'experiment': exp_id,
                'config': experiment.name,
                'run_name': run_name,
                'mode': run_name.split('/')[-1],
                'status': run.info.status,
            }
                
            # final metrics values
            final_info = run_info.copy()
            for m_name, m_val in run.data.metrics.items():
                if any(x in m_name for x in ['DICE', 'IoU', 'AbsRel', 'Bad3']):
                    m_val *= 100
                final_info[f'metric.{m_name}'] = m_val
            final_runs_data.append(final_info)
                
            # hyperparams 
            params_info = run_info.copy()
            for p_name, p_val in run.data.params.items(): params_info[f'param.{p_name}'] = p_val
            params_info['mlflow_experiment_id'] = experiment.experiment_id
            params_info['mlflow_run_id'] = run_id
            params_runs_data.append(params_info)
                
            # historical metrics data
            for metric_name in run.data.metrics.keys():
                history = client.get_metric_history(run_id, metric_name)
                for m in history:
                    val = m.value
                    if any(x in metric_name for x in ['DICE', 'IoU', 'AbsRel', 'Bad3']):
                        val *= 100
                    historic_runs_data.append({
                        'identifier': identifier,
                        'experiment': os.path.basename(exp_folder),
                        'config': experiment.name,
                        'run_name': run_name,
                            
                        'metric_name': metric_name,
                        'value': val,
                        'step': m.step,
                        'timestamp': m.timestamp,
                            
                        'mlflow_experiment_id': experiment.experiment_id,
                        'mlflow_run_id': run_id,
                    })
                        

df_final = pd.DataFrame(final_runs_data)
print(f'Loaded {len(df_final)} runs.')

df_params = pd.DataFrame(params_runs_data)
print(f'Loaded {len(df_params)} parameter data points.')

df_historic = pd.DataFrame(historic_runs_data)
print(f'Loaded {len(df_historic)} metric data points.')
        
data_df = {
    'final': df_final,
    'params': df_params,
    'historic': df_historic
}

# with open('./notebooks/evaluation/storage/dataframes.pkl', 'wb') as f:
#     pickle.dump(data_df, f)
    
print('Dataframes saved to ./notebooks/evaluation/storage/dataframes.pkl')

# %%

df_historic.to_csv('./notebooks/evaluation/storage/historic.csv', index=False)
print('Historic dataframe saved to ./notebooks/evaluation/storage/historic.csv')

# # %%
# import os, glob
# from mlflow.tracking import MlflowClient

# mlflow_base_path = './mlruns_experiments'
# exp_folders = sorted(glob.glob(os.path.join(mlflow_base_path, 'exp*')))

# non_finished = []

# for exp_folder in exp_folders:
#     tracking_uri = f'file://{os.path.abspath(exp_folder)}'
#     client = MlflowClient(tracking_uri)
    
#     # Loop over all experiments in this folder
#     for experiment in client.search_experiments():
#         runs = client.search_runs(experiment_ids=[experiment.experiment_id])
#         non_finished.extend([
#             {'run_id': r.info.run_id, 'experiment': experiment.name, 'status': r.info.status} 
#             for r in runs if r.info.status != 'FINISHED'
#         ])

# print(f"Found {len(non_finished)} non-completed runs.")
# if non_finished:
#     for run in non_finished:
#         print(run)

# %%
