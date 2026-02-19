import os
import shutil
import atexit

def setup_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print(f'Restricting to GPU {os.environ.get("CUDA_VISIBLE_DEVICES")}')
        
    cpu_cores = list(range(24))
    os.sched_setaffinity(os.getpid(), cpu_cores)
    print(f'Set CPU affinity to cores: {cpu_cores}')
    
    local_tmp = os.path.join(os.getcwd(), '.temp')
    os.makedirs(local_tmp, exist_ok=True)
    
    os.environ['TMPDIR'] = local_tmp
    os.environ['MLFLOW_TMP_DIR'] = local_tmp
    print(f'Redirected TMPDIR to local folder: {local_tmp}')

    def cleanup():
        if os.path.exists(local_tmp):
            try:
                shutil.rmtree(local_tmp)
                print(f'Cleaned up temp directory: {local_tmp}')
            except Exception as e:
                print('Warning: Could not clean up {local_tmp}: {e}')

    atexit.register(cleanup)