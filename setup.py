import os

def setup_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print(f'Restricting to GPU {os.environ.get("CUDA_VISIBLE_DEVICES")}')
        
    cpu_cores = list(range(24))
    os.sched_setaffinity(os.getpid(), cpu_cores)
    print(f'Set CPU affinity to cores: {cpu_cores}')