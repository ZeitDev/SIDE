# * Show FPS, VRAM, PARAMs for relevant experiments
# * Exp01 - Baseline
# * Exp04 - ConvNeXt Nano
# * Exp05 - Less Input Resolution
# * Teachers

# 01/04 Teacher
# NVIDIA SF = 7.14 FPS // 140.06 ms // 2114.68 MB
# NVIDIA FS = 0.49 FPS // 2040.82 ms // 7126.30 MB in half precision
# Sequence Teacher = 0.46 FPS // 2180.88 ms

# 01 Student
# SEG = 20.08 FPS // 49.80 ms // 1389.72 MB
# DISP = 13.28 FPS // 75.30 ms // 1435.92 MB
# Sequence Single Task = 7.99 FPS // 125.10 ms
# MT (convnext/wMT/260406:2037/train) = 10.52 FPS // 95.02 ms // 1.44 GB // 540.29 G // 42.41 M
# MT-KD (convnext/wMT-KD/260406:2036/train) = 10.50 FPS // 95.23 ms // 1.44 GB // 540.29 G // 42.41 M

# 04 Student
# SEG = 32.25 FPS // 30.91 ms // 119.28 MB // 1.09 GB
# DISP = 22.88 FPS // 43.72 ms // 1157.52 MB // 1.13 GB
# Sequence Single Task = 13.38 FPS // 74.74 ms 
# MT = 16.47 FPS // 60.73 ms // 1210.77 MB // 1.18 GB
# MT-KD = 16.47 FPS // 60.70 ms // 1210.77 MB // 1.18 GB

# 05 Student
# SEG = 75.46 FPS // 13.25 ms // 456.5 MB // 0.45 GB
# DISP = 49.24 FPS // 20.31 ms // 486.96 MB // 0.46 GB
# Sequence Single Task = 29.8 FPS // 33.56 ms
# MT = 38.06 FPS // 26.27 ms // 506.41 MB // 0.49 GB
# MT-KD = 38.56 FPS // 25.94 ms // 505.88 // 0.49 GB

# %%
