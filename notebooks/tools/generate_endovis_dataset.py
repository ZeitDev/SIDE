# %% Import
# Import
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from pathlib import Path
import numpy as np
import cv2
from tqdm.notebook import tqdm
import plotly.express as px
import json
import shutil
import matplotlib.pyplot as plt
import math

from utils.setup import setup_environment
setup_environment()

# %% Settings
# Settings

MODE = 'test'  # 'train' or 'test'
SEGMENTATION = True
RECT_ALPHA = 0  # 0: cropped (no black bars, for training), 1: full image (with black bars), -1: minimal black bars (for testing?)

source_path = Path('/data/Zeitler/SIDED/EndoVis17/raw') / MODE
mappings_name = Path('instrument_type_mapping.json')
output_path = Path('/data/Zeitler/SIDED/EndoVis17/processed') / MODE

# %% Helpers
# Helpers
def get_calibration(path):
    with open(path, 'r') as calib_txt:
        lines = calib_txt.readlines()
        
    CRITICAL_KEYS = {
        'Width', 'Height',
        'Camera-0-F', 'Camera-0-C', 'Camera-0-Alpha', 'Camera-0-K',
        'Camera-1-F', 'Camera-1-C', 'Camera-1-Alpha', 'Camera-1-K',
        'Extrinsic-Omega', 'Extrinsic-T'
    }
    
    raw_params = {}
    for line in lines:
        line = line.split('//')[0].rstrip()
        if ':' not in line: continue
        
        key, value = line.split(': ', 1)
        key = key.strip()
        
        if key in CRITICAL_KEYS:
            try:
                values = [float(v) for v in value.strip().split()]
                raw_params[key] = values
            except ValueError:
                raise ValueError(f'Could not parse {key} in calibration file.')
        
    if len(raw_params) != len(CRITICAL_KEYS):
        missing = CRITICAL_KEYS - raw_params.keys()
        raise ValueError(f'Missing critical calibration keys: {missing}')
    
    calib = {}
    calib['size'] = (int(raw_params['Height'][0]), int(raw_params['Width'][0]))
    K1 = np.eye(3, dtype=np.double)
    K1[0,0], K1[1,1] = raw_params['Camera-0-F']
    K1[0,2], K1[1,2] = raw_params['Camera-0-C']
    K1[0,1] = raw_params['Camera-0-Alpha'][0]
    calib['K1'] = K1
    
    calib['D1'] = np.array(raw_params['Camera-0-K'], dtype=np.double).reshape(-1, 1)
    
    K2 = np.eye(3, dtype=np.double)
    K2[0,0], K2[1,1] = raw_params['Camera-1-F']
    K2[0,2], K2[1,2] = raw_params['Camera-1-C']
    K2[0,1] = raw_params['Camera-1-Alpha'][0]
    calib['K2'] = K2
    
    calib['D2'] = np.array(raw_params['Camera-1-K'], dtype=np.double).reshape(-1, 1)
    
    omega_vec = np.array(raw_params['Extrinsic-Omega'], dtype=np.double)
    R_matrix, _ = cv2.Rodrigues(omega_vec)
    calib['R'] = R_matrix
    
    calib['T'] = np.array(raw_params['Extrinsic-T'], dtype=np.double).reshape(3, 1)
    
    return calib

def deinterlace(img, out_size=None, interpolation=cv2.INTER_LINEAR):
    if out_size is None:
        out_size = img.shape
    img0 = img[::2].copy()
    img1 = img[1::2].copy()
    img0 =cv2.resize(img0, out_size[::-1], interpolation=interpolation)
    img1 =cv2.resize(img1, out_size[::-1], interpolation=interpolation)
    return img0, img1

class Stereo_Rectifier:
    def __init__(self, calib):
        self.calib = calib.copy()
        self.left_rect_map = None
        self.right_rect_map = None
        self.rect_alpha = None
    
    def _compute_rectification_parameters(self):
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.calib['K1'], 
                                                        self.calib['D1'],
                                                        self.calib['K2'],
                                                        self.calib['D2'],
                                                        self.calib['size'],
                                                        self.calib['R'],
                                                        self.calib['T'],
                                                        alpha=self.rect_alpha) 
        
        rect_calib = {'R1': R1,'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
                        'roi1': roi1, 'roi2': roi2, 'alpha':self.rect_alpha}
        self.calib.update(rect_calib)

    def rectify(self, left, right, alpha=-1, interpolation=cv2.INTER_LINEAR):
        if alpha != self.rect_alpha:
            self.left_rect_map= None
            self.calib['R1'] = None
            self.rect_alpha = alpha
            
        if (self.left_rect_map is None):
            if self.calib['R1'] is None:
                self._compute_rectification_parameters()
            self.left_rect_map = cv2.initUndistortRectifyMap(self.calib['K1'],
                                                                self.calib['D1'],
                                                                self.calib['R1'],
                                                                self.calib['P1'],
                                                                self.calib['size'][::-1],
                                                                cv2.CV_32FC1)
            self.right_rect_map = cv2.initUndistortRectifyMap(self.calib['K2'],
                                                                self.calib['D2'],
                                                                self.calib['R2'],
                                                                self.calib['P2'],
                                                                self.calib['size'][::-1],
                                                                cv2.CV_32FC1)

        left_rect = cv2.remap(left, self.left_rect_map[0],
                                self.left_rect_map[1], interpolation)
        right_rect = cv2.remap(right, self.right_rect_map[0],
                                self.right_rect_map[1], interpolation)
        return left_rect, right_rect

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# convert bgr to rgb for plotting: left_image[:,:,::-1]
# px.imshow(combined_segmentation_mask, color_continuous_scale='gray').show()

# %% Generate Dataset
# Generate Dataset
# Because the dataset is weird
# 1. Crop images to 1280, 1024 to match camera calibration resolution and ensure no stretch for disparity reconstruction
# 2. Deinterlace images into two separate frames, because they are interlaced: even rows in T, odd rows in T+1 -> solution: take only even rows
# 3. Rectify images using camera calibration, because stereo cameras are not perfectly aligned, rotates and translates them to be aligned
# 4. Recrop to remove black bars to papers values [37:1047, 328:1591]

first_crop_x = (1920 - 1280) // 2  # 320 pixels left and right
first_crop_y = (1080 - 1024) // 2  # 28 pixels top and bottom
first_crop_w = 1280
first_crop_h = 1024

# TODO: Why is the paper using these values?
# clean_crop_x_start, clean_crop_x_end = 328, 1591
# clean_crop_y_start, clean_crop_y_end = 37, 1047

with open(source_path.parent / mappings_name) as f:
    instrument_type_mapping = json.load(f)

sequences = sorted([path for path in source_path.iterdir() if path.is_dir()])
for sequence in sequences:
    output_left_images_path = output_path / sequence.name / 'input' / 'left_images'
    output_right_images_path = output_path / sequence.name / 'input' / 'right_images'
    output_ground_truth_segmentation_path = output_path / sequence.name / 'ground_truth' / 'segmentation'
    output_ground_truth_disparity_path = output_path / sequence.name / 'ground_truth' / 'disparity'
    
    output_left_images_path.mkdir(parents=True, exist_ok=True)
    output_right_images_path.mkdir(parents=True, exist_ok=True)
    if SEGMENTATION: output_ground_truth_segmentation_path.mkdir(parents=True, exist_ok=True)
    
    calibration_path = sequence / 'camera_calibration.txt'
    calibration = get_calibration(calibration_path)
    
    source_left_images_paths = sorted((sequence / 'left_images').glob('*.png'))
    source_right_images_paths = sorted((sequence / 'right_images').glob('*.png'))

    # Left and Right Images
    # 1. Crop
    # 2. Deinterlace
    # 3. Rectify
    rectifier = Stereo_Rectifier(calibration)
    for left_image_path, right_image_path in tqdm(zip(source_left_images_paths, source_right_images_paths)):
        left_image = cv2.imread(str(left_image_path))
        right_image = cv2.imread(str(right_image_path))
        
        left_image_cropped = left_image[first_crop_y:first_crop_y+first_crop_h, first_crop_x:first_crop_x+first_crop_w]
        right_image_cropped = right_image[first_crop_y:first_crop_y+first_crop_h, first_crop_x:first_crop_x+first_crop_w]
        
        left_image_0, _ = deinterlace(left_image_cropped, out_size=(first_crop_h, first_crop_w))
        right_image_0, _ = deinterlace(right_image_cropped, out_size=(first_crop_h, first_crop_w))
        
        left_image_rectified, right_image_rectified = rectifier.rectify(left_image_0, right_image_0, alpha=RECT_ALPHA)
        
        filename = left_image_path.name.replace('frame', 'image')
        assert filename == right_image_path.name.replace('frame', 'image')
        cv2.imwrite(str(output_left_images_path / filename), left_image_rectified)
        cv2.imwrite(str(output_right_images_path / filename), right_image_rectified)
    
    save_path = output_path / sequence.name / 'input' / 'rectified_calibration.json'
    with open(save_path, 'w') as f:
        json.dump(rectifier.calib, f, cls=NumpyEncoder, indent=4)

    # Segmentation Masks
    # 1. Crop
    # 2. Resize
    # 3. Rectify
    # 4. Combine into single mask with different instrument types
    if SEGMENTATION:
        source_ground_truth_segmentation_instruments = sorted([ path for path in (sequence / 'ground_truth').iterdir() if path.is_dir()])
        source_ground_truth_segmentation_instruments_paths = [sorted((sequence / 'ground_truth' / instrument).glob('*.png')) for instrument in sorted([p.name for p in source_ground_truth_segmentation_instruments])]
        for ground_truth_segmentation in tqdm(zip(*source_ground_truth_segmentation_instruments_paths)):
            ground_truth_list_with_type = []
            filename = ground_truth_segmentation[0].name.replace('frame', 'image')
            
            for segmentation_mask_path in ground_truth_segmentation:
                instrument_name = segmentation_mask_path.parent.name
                if instrument_name == 'binary': continue
                instrument_type = None
                instrument_type_id = None
                for mapping_name, mapping_id in instrument_type_mapping.items():
                    check_name = mapping_name.replace(' ', '_')
                    if check_name in instrument_name:
                        instrument_type = mapping_name
                        instrument_type_id = mapping_id
                    
                if instrument_type is None:
                    print(f'Warning: Could not find instrument type for folder {instrument_name}. Skipping.')
                    continue
                
                segmentation_mask = cv2.imread(str(segmentation_mask_path), 0)
                segmentation_mask_cropped = segmentation_mask[first_crop_y:first_crop_y+first_crop_h, first_crop_x:first_crop_x+first_crop_w]
                segmentation_mask_resized = cv2.resize(segmentation_mask_cropped, (first_crop_w, first_crop_h), interpolation=cv2.INTER_NEAREST)
                segmentation_mask_rectified, _ = rectifier.rectify(segmentation_mask_resized, segmentation_mask_resized, alpha=RECT_ALPHA, interpolation=cv2.INTER_NEAREST)
                ground_truth_list_with_type.append((segmentation_mask_rectified, instrument_type_id))
                
            first_segmentation_mask = ground_truth_list_with_type[0][0]
            combined_segmentation_mask = np.zeros(first_segmentation_mask.shape, dtype=np.uint8)    
            for segmentation_mask_rectified, instrument_type_id in ground_truth_list_with_type:
                combined_segmentation_mask[segmentation_mask_rectified > 0] = instrument_type_id
                
            cv2.imwrite(str(output_ground_truth_segmentation_path / filename), combined_segmentation_mask)

    shutil.copy(source_path / sequence.name / 'camera_calibration.txt', output_path / sequence.name / 'original_calibration.txt')
shutil.copy(source_path.parent / mappings_name, output_path / mappings_name)

# %%
