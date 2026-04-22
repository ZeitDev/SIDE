# %% Imports and Setup
import os
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
# Update this path to point to a valid sequence in your raw dataset
SEQ_PATH = Path('/data/Zeitler/SIDED/EndoVis17/raw/train/instrument_dataset_1')
CALIB_PATH = SEQ_PATH / 'camera_calibration.txt'

# You might need to change 'left_frames' to 'left_images' depending on the exact folder name 
# and 'frame000.png' to an existing image file in that directory.
LEFT_IMG_PATH = list((SEQ_PATH / 'left_images').glob('*.png'))[50]  # Grab a frame with instruments
RIGHT_IMG_PATH = list((SEQ_PATH / 'right_images').glob('*.png'))[50]

# %% Calibration and Rectification Helpers
def get_calibration(path):
    with open(path, 'r') as calib_txt: lines = calib_txt.readlines()
    CRITICAL_KEYS = {'Width', 'Height', 'Camera-0-F', 'Camera-0-C', 'Camera-0-Alpha', 'Camera-0-K',
                     'Camera-1-F', 'Camera-1-C', 'Camera-1-Alpha', 'Camera-1-K', 'Extrinsic-Omega', 'Extrinsic-T'}
    raw_params = {}
    for line in lines:
        line = line.split('//')[0].rstrip()
        if ':' not in line: continue
        key, value = line.split(': ', 1)
        if key.strip() in CRITICAL_KEYS:
            raw_params[key.strip()] = [float(v) for v in value.strip().split()]
            
    calib = {'size': (int(raw_params['Height'][0]), int(raw_params['Width'][0]))}
    K1, K2 = np.eye(3), np.eye(3)
    K1[0,0], K1[1,1] = raw_params['Camera-0-F']
    K1[0,2], K1[1,2] = raw_params['Camera-0-C']
    K1[0,1] = raw_params['Camera-0-Alpha'][0]
    
    K2[0,0], K2[1,1] = raw_params['Camera-1-F']
    K2[0,2], K2[1,2] = raw_params['Camera-1-C']
    K2[0,1] = raw_params['Camera-1-Alpha'][0]

    calib['K1'], calib['K2'] = K1, K2
    calib['D1'] = np.array(raw_params['Camera-0-K']).reshape(-1, 1)
    calib['D2'] = np.array(raw_params['Camera-1-K']).reshape(-1, 1)
    calib['R'], _ = cv2.Rodrigues(np.array(raw_params['Extrinsic-Omega']))
    calib['T'] = np.array(raw_params['Extrinsic-T']).reshape(3, 1)
    return calib

class Stereo_Rectifier:
    def __init__(self, calib):
        self.calib = calib.copy()
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.calib['K1'], self.calib['D1'], self.calib['K2'], self.calib['D2'], 
            self.calib['size'], self.calib['R'], self.calib['T'], alpha=0.0
        )
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(self.calib['K1'], self.calib['D1'], R1, P1, self.calib['size'][::-1], cv2.CV_32FC1)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(self.calib['K2'], self.calib['D2'], R2, P2, self.calib['size'][::-1], cv2.CV_32FC1)

    def rectify(self, left, right):
        return (cv2.remap(left, self.map1_l, self.map2_l, cv2.INTER_LINEAR),
                cv2.remap(right, self.map1_r, self.map2_r, cv2.INTER_LINEAR))

# %% Load Data and Initialize
img_l = cv2.imread(str(LEFT_IMG_PATH))[..., ::-1]  # BGR to RGB
img_r = cv2.imread(str(RIGHT_IMG_PATH))[..., ::-1]
calib = get_calibration(CALIB_PATH)
rectifier = Stereo_Rectifier(calib)

# Define pipeline functions
def process_method_A_pipeline(img):
    # 1. Exact Center Crop (1280x1024)
    # y: 28 to 1052, x: 320 to 1600
    img_c = img[28:1052, 320:1600].copy()
    # 2. Deinterlace & Resize back to 1280x1024
    img_even = img_c[::2].copy()
    img_processed = cv2.resize(img_even, (1280, 1024), interpolation=cv2.INTER_LINEAR)
    return img_processed

def process_method_B_pipeline(img):
    # 1. method_B's "remove black borders" Crop
    # Size after crop: 1010 x 1263
    img_c = img[37:1047, 328:1591].copy()
    # 2. Deinterlace & Resize stretched to full 1280x1024
    img_even = img_c[::2].copy()
    img_processed = cv2.resize(img_even, (1280, 1024), interpolation=cv2.INTER_LINEAR)
    return img_processed

# Run both pipelines
method_A_l, method_A_r = process_method_A_pipeline(img_l), process_method_A_pipeline(img_r)
method_B_l, method_B_r = process_method_B_pipeline(img_l), process_method_B_pipeline(img_r)

# Rectify
method_A_rect_l, method_A_rect_r = rectifier.rectify(method_A_l, method_A_r)
method_B_rect_l, method_B_rect_r = rectifier.rectify(method_B_l, method_B_r)

# %% Visualize Epipolar Line Alignment
def draw_epipolar_lines(img_left, img_right, title):
    # Create side-by-side composite
    h, w, _ = img_left.shape
    composite = np.hstack((img_left, img_right))
    
    plt.figure(figsize=(18, 10))
    plt.imshow(composite)
    plt.title(title, fontsize=16)
    
    # Draw horizontal epipolar lines every 50 pixels
    for y in range(0, h, 50):
        plt.axhline(y, color='red', alpha=0.5, linestyle='-', linewidth=1)
        
    # Draw vertical divider
    plt.axvline(w, color='white', linewidth=3)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# The winner is the one where an instrument feature perfectly hits the SAME red line on the left and right halves.
draw_epipolar_lines(method_A_rect_l, method_A_rect_r, "METHOD A: method_A's Center Crop Pipeline")
draw_epipolar_lines(method_B_rect_l, method_B_rect_r, "METHOD B: method_B's Border Crop & Stretch Pipeline")

# %%
# Settings
RAW_DATA_ROOT = Path('/data/Zeitler/SIDED/EndoVis17/raw/train')
OUTPUT_DIR = Path('/data/Zeitler/Visualization/rectification')

SBS_DIR = OUTPUT_DIR / 'sbs_videos'
ANAGLYPH_DIR = OUTPUT_DIR / 'anaglyph_images'

SBS_DIR.mkdir(parents=True, exist_ok=True)
ANAGLYPH_DIR.mkdir(parents=True, exist_ok=True)

FPS = 1.0

def make_anaglyph(img_l_bgr, img_r_bgr):
    """Creates a Red-Cyan Anaglyph from BGR images."""
    anaglyph = np.zeros_like(img_l_bgr)
    anaglyph[:, :, 0] = img_r_bgr[:, :, 0]  # Blue from Right
    anaglyph[:, :, 1] = img_r_bgr[:, :, 1]  # Green from Right
    anaglyph[:, :, 2] = img_l_bgr[:, :, 2]  # Red from Left
    return anaglyph

sequences = sorted([d for d in RAW_DATA_ROOT.iterdir() if d.is_dir()])

for seq in sequences:
    print(f"Processing {seq.name}...")
    
    calib_path = seq / 'camera_calibration.txt'
    if not calib_path.exists():
        print(f"No calibration found for {seq.name}. Skipping.")
        continue
        
    calib = get_calibration(calib_path)
    rectifier = Stereo_Rectifier(calib)
    
    left_paths = sorted((seq / 'left_frames').glob('*.png'))
    if not left_paths:  # Fallback in case folder is named 'left_images'
        left_paths = sorted((seq / 'left_images').glob('*.png'))
        
    right_paths = sorted((seq / 'right_frames').glob('*.png'))
    if not right_paths:
        right_paths = sorted((seq / 'right_images').glob('*.png'))
        
    if not left_paths or len(left_paths) != len(right_paths):
        print(f"Missing or mismatched images for {seq.name}. Skipping.")
        continue

    # Prepare Video Writers
    # SBS dimensions: 1280*2 = 2560 width, 1024 height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_path_A = SBS_DIR / f"{seq.name}_Method_A_SBS.mp4"
    vid_path_B = SBS_DIR / f"{seq.name}_Method_B_SBS.mp4"
    
    writer_A = cv2.VideoWriter(str(vid_path_A), fourcc, FPS, (2560, 1024))
    writer_B = cv2.VideoWriter(str(vid_path_B), fourcc, FPS, (2560, 1024))
    
    # Store anaglyphs in sequence-specific folders
    seq_anaglyph_dir = ANAGLYPH_DIR / seq.name
    seq_anaglyph_dir.mkdir(exist_ok=True)
    
    # Iterate through every frame
    for left_p, right_p in tqdm(zip(left_paths, right_paths), total=len(left_paths)):
        img_l = cv2.imread(str(left_p)) # Keep in BGR for cv2 saving
        img_r = cv2.imread(str(right_p))
        
        # Process Method A
        mA_l = process_method_A_pipeline(img_l)
        mA_r = process_method_A_pipeline(img_r)
        mA_rect_l, mA_rect_r = rectifier.rectify(mA_l, mA_r)
        
        # Process Method B
        mB_l = process_method_B_pipeline(img_l)
        mB_r = process_method_B_pipeline(img_r)
        mB_rect_l, mB_rect_r = rectifier.rectify(mB_l, mB_r)
        
        # 1. Write to SBS Videos
        sbs_A = np.hstack((mA_rect_l, mA_rect_r))
        sbs_B = np.hstack((mB_rect_l, mB_rect_r))
        writer_A.write(sbs_A)
        writer_B.write(sbs_B)
        
        # 2. Save Anaglyphs
        frame_name = left_p.stem
        ana_A = make_anaglyph(mA_rect_l, mA_rect_r)
        ana_B = make_anaglyph(mB_rect_l, mB_rect_r)
        
        cv2.imwrite(str(seq_anaglyph_dir / f"{frame_name}_Method_A.png"), ana_A)
        cv2.imwrite(str(seq_anaglyph_dir / f"{frame_name}_Method_B.png"), ana_B)
        
    writer_A.release()
    writer_B.release()

print("Generation Complete! Files saved to:", OUTPUT_DIR)

# %% Evaluate Rectification Accuracy with SIFT
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def calculate_vertical_rmse_from_arrays(imgL_bgr, imgR_bgr):
    """Calculates vertical RMSE between two rectified BGR images using SIFT."""
    imgL = cv2.cvtColor(imgL_bgr, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR_bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)

    if des1 is None or des2 is None:
        return np.nan

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    # Lowe's Ratio Test
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
         return np.nan

    y_errors_squared = []
    for match in good_matches:
        y1 = kp1[match.queryIdx].pt[1]
        y2 = kp2[match.trainIdx].pt[1]
        
        y_error = y1 - y2
        y_errors_squared.append(y_error ** 2)

    return np.sqrt(np.mean(y_errors_squared))

# Settings
RAW_DATA_ROOT = Path('/data/Zeitler/SIDED/EndoVis17/raw/train')
sequences = sorted([d for d in RAW_DATA_ROOT.iterdir() if d.is_dir()])

overall_errors_A = []
overall_errors_B = []

print("Starting SIFT Vertical RMSE Evaluation...")

for seq in sequences:
    calib_path = seq / 'camera_calibration.txt'
    if not calib_path.exists():
        continue
        
    calib = get_calibration(calib_path)
    rectifier = Stereo_Rectifier(calib)
    
    left_paths = sorted((seq / 'left_frames').glob('*.png'))
    if not left_paths:
        left_paths = sorted((seq / 'left_images').glob('*.png'))
        
    right_paths = sorted((seq / 'right_frames').glob('*.png'))
    if not right_paths:
        right_paths = sorted((seq / 'right_images').glob('*.png'))
        
    if not left_paths or len(left_paths) != len(right_paths):
        continue

    seq_errors_A = []
    seq_errors_B = []

    for left_p, right_p in tqdm(zip(left_paths, right_paths), total=len(left_paths), desc=seq.name):
        img_l = cv2.imread(str(left_p))
        img_r = cv2.imread(str(right_p))
        
        # Process and calculate Method A
        mA_l, mA_r = process_method_A_pipeline(img_l), process_method_A_pipeline(img_r)
        mA_rect_l, mA_rect_r = rectifier.rectify(mA_l, mA_r)
        err_A = calculate_vertical_rmse_from_arrays(mA_rect_l, mA_rect_r)
        if not np.isnan(err_A):
            seq_errors_A.append(err_A)
            
        # Process and calculate Method B
        mB_l, mB_r = process_method_B_pipeline(img_l), process_method_B_pipeline(img_r)
        mB_rect_l, mB_rect_r = rectifier.rectify(mB_l, mB_r)
        err_B = calculate_vertical_rmse_from_arrays(mB_rect_l, mB_rect_r)
        if not np.isnan(err_B):
            seq_errors_B.append(err_B)

    # Print Sequence Summaries
    mean_A = np.mean(seq_errors_A) if seq_errors_A else float('inf')
    mean_B = np.mean(seq_errors_B) if seq_errors_B else float('inf')
    overall_errors_A.extend(seq_errors_A)
    overall_errors_B.extend(seq_errors_B)
    
    print(f"--- {seq.name} ---")
    print(f"  Method A Avg Vertical Error: {mean_A:.4f} px")
    print(f"  Method B Avg Vertical Error: {mean_B:.4f} px")
    print(f"  Winner: {'Method A' if mean_A < mean_B else 'Method B'}")

# Print Final Statistics
print("\n" + "="*40)
print("FINAL DATASET STATISTICS")
print("="*40)
final_A = np.mean(overall_errors_A)
final_B = np.mean(overall_errors_B)

print(f"OVERALL Method A Error: {final_A:.4f} px")
print(f"OVERALL Method B Error: {final_B:.4f} px")
print(f"ABSOLUTE WINNER: {'Method A' if final_A < final_B else 'Method B'} by {abs(final_A - final_B):.4f} px")

# %%