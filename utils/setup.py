import os
import shutil
import atexit

def cleanup(local_tmp):
    if os.path.exists(local_tmp):
        try:
            shutil.rmtree(local_tmp)
            print(f'Cleaned up temp directory: {local_tmp}')
        except Exception as e:
            print('Warning: Could not clean up {local_tmp}: {e}')
            
def setup_environment(skip_cuda=False, cuda_device=1, delete_temp=False):
    if not skip_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        print(f'Restricting to GPU {os.environ.get("CUDA_VISIBLE_DEVICES")}')
        
    if cuda_device==1: cpu_cores = list(range(24))
    elif cuda_device==0: cpu_cores = list(range(24, 48))
    os.sched_setaffinity(os.getpid(), cpu_cores)
    print(f'Set CPU affinity to cores: {cpu_cores}')
    
    local_tmp = os.path.join(os.getcwd(), f'.temp_{cuda_device}')
    if delete_temp: 
        cleanup(local_tmp)
        os.makedirs(local_tmp, exist_ok=True)
    
    os.environ['TMPDIR'] = local_tmp
    os.environ['MLFLOW_TMP_DIR'] = local_tmp
    #print(f'Redirected TMPDIR to local folder: {local_tmp}')


    #atexit.register(cleanup)