import os
import shutil
import numpy as np
import cv2
from osgeo import gdal
from spectral.io import envi
from scipy.signal import savgol_filter
from tqdm import tqdm

# ==============================================================================
#                               CONFIGURATIONS
# ==============================================================================

# Paths
DATA_SOURCE_FOLDER = r"dataset"  # Root folder containing P1, P2...
OUTPUT_FOLDER = r"processed_dataset"

# Preprocessing Flags
PERCENT_LINEAR_VALUE = 0  # Percentile for linear stretch (e.g., 2 for 2%)
PERBAND = 0               # 1: Process each band independently; 0: Process 3D cube
SNV_USED = 1              # 1: Apply Standard Normal Variate; 0: Off
OPTLIN = 1                # 1: Apply Optimized Linear Stretch; 0: Off (Uses standard percent linear)

# Spectral Parameters
BAND_MIN = 450
STEP = 5
BAND_NUM = 61
BAND_MAX = BAND_MIN + (BAND_NUM - 1) * STEP

# Default RGB Bands for visualization in ENVI header
RED_WAVELENGTH = 660
GREEN_WAVELENGTH = 540
BLUE_WAVELENGTH = 470

# Calculate band indices for header
red_band_idx = int((RED_WAVELENGTH - BAND_MIN) / STEP)
green_band_idx = int((GREEN_WAVELENGTH - BAND_MIN) / STEP)
blue_band_idx = int((BLUE_WAVELENGTH - BAND_MIN) / STEP)

# ==============================================================================
#                             HELPER FUNCTIONS
# ==============================================================================

def optimized_linear_float(arr):
    """
    Simulates ENVI's Optimized Linear Stretch but outputs 0-1 float.
    """
    arr = arr.astype(np.float32)
    a, b = np.percentile(arr, (2.5, 99))
    c = a - 0.1 * (b - a)
    d = b + 0.5 * (b - a)
    
    if d - c == 0:
        return np.zeros_like(arr, dtype=np.float32)
        
    arr = (arr - c) / (d - c)
    arr = np.clip(arr, 0.0, 1.0) # Clip to 0-1 range
    return arr.astype(np.float32)

def percent_linear_float(arr, percent=2):
    """
    Standard linear stretch based on percentiles, outputs 0-1 float.
    """
    arr = arr.astype(np.float32)
    arr_min, arr_max = np.percentile(arr, (percent, 100 - percent))
    
    if arr_max - arr_min == 0:
        return np.zeros_like(arr, dtype=np.float32)

    arr = (arr - arr_min) / (arr_max - arr_min)
    arr = np.clip(arr, 0.0, 1.0) # Clip to 0-1 range
    return arr.astype(np.float32)

def snv(input_data):
    """
    Standard Normal Variate (SNV) correction.
    """
    output_data = np.zeros_like(input_data, dtype=np.float32)
    mean = np.mean(input_data)
    std = np.std(input_data)
    
    if std == 0:
        return output_data
        
    output_data = (input_data - mean) / std
    return output_data

def check_tissue_cleanliness(img, name):
    """
    Checks if tissue label only contains 0 and 255.
    Returns: None (Just prints warning if dirty)
    """
    unique_vals = np.unique(img)
    # Allow 0 and 255.
    allowed = {0, 255}
    found = set(unique_vals)
    
    if not found.issubset(allowed):
        unexpected = list(found - allowed)
        print(f"\n[WARNING] '{name}' Tissue Label is NOT clean (expected 0/255).")
        print(f"          Found unexpected values: {unexpected}")

def process_tissue_label(path, unique_name):
    """Reads label image, checks cleanliness, converts to 0/1."""
    if not os.path.exists(path):
        return None
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return None

    # 1. Cleanliness Check
    check_tissue_cleanliness(img, unique_name)

    # 2. Process (Threshold to 0/1)
    _, thresh = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return thresh.astype(np.uint8)

def check_cell_cleanliness(img, name):
    """
    Checks if cell label only contains Black, Red, Green, Blue.
    Returns: None (Prints warning)
    """
    # Reshape to list of pixels [N, 3]
    pixels = img.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    
    # Expected Colors (BGR Format for OpenCV)
    # Black: [0,0,0], Red: [0,0,255], Green: [0,255,0], Blue: [255,0,0]
    expected_colors = np.array([
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0]
    ])
    
    dirty_colors = []
    
    # Check every unique color found
    for color in unique_colors:
        # Check if 'color' exists in 'expected_colors'
        match = np.any(np.all(expected_colors == color, axis=1))
        if not match:
            dirty_colors.append(color)
            
    if dirty_colors:
        print(f"\n[WARNING] '{name}' Cell Label is NOT clean (expected Black/Red/Green/Blue).")
        print(f"          Found {len(dirty_colors)} unexpected RGB combinations.")
        if len(dirty_colors) <= 50:
            print(f"          Values (BGR): {dirty_colors}")
        else:
            print(f"          First 50 Values (BGR): {dirty_colors[:50]} ...")

def process_cell_label(path, unique_name):
    """Reads cell label, checks cleanliness, maps colors to 0/1/2/3."""
    if not os.path.exists(path):
        return None
    
    img = cv2.imread(path) # BGR
    if img is None:
        return None

    # 1. Cleanliness Check
    check_cell_cleanliness(img, unique_name)

    # 2. Process Map
    h, w, c = img.shape
    label_map = np.zeros((h, w), dtype=np.uint8)

    # Define Masks (BGR)
    # Using ranges allows us to map even if slightly dirty, 
    # but the check above warns the user.
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([80, 80, 255])
    
    lower_green = np.array([0, 100, 0])
    upper_green = np.array([80, 255, 80])
    
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 80, 80])

    mask_red = cv2.inRange(img, lower_red, upper_red)
    mask_green = cv2.inRange(img, lower_green, upper_green)
    mask_blue = cv2.inRange(img, lower_blue, upper_blue)
    
    label_map[mask_red > 0] = 1
    label_map[mask_green > 0] = 2
    label_map[mask_blue > 0] = 3
    
    return label_map

# ==============================================================================
#                               MAIN PROGRAM
# ==============================================================================

def main():
    # 1. Setup Output Directory
    sub_dirs = ["datacubes", "labels", "cell_labels", "rgbs"]
    path_map = {}
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    for sd in sub_dirs:
        full_path = os.path.join(OUTPUT_FOLDER, sd)
        path_map[sd] = full_path
        if not os.path.exists(full_path):
            os.makedirs(full_path)

    print(f"Reading data from: {DATA_SOURCE_FOLDER}")
    print(f"Outputting to: {OUTPUT_FOLDER}")

    if not os.path.exists(DATA_SOURCE_FOLDER):
        print("Error: Data Source Folder not found.")
        return

    patient_folders = [f for f in os.listdir(DATA_SOURCE_FOLDER) if os.path.isdir(os.path.join(DATA_SOURCE_FOLDER, f))]

    for patient in tqdm(patient_folders, desc="Processing Patients"):
        patient_path = os.path.join(DATA_SOURCE_FOLDER, patient)
        roi_folders = [f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))]
        
        for roi in tqdm(roi_folders, desc=f"ROIs in {patient}", leave=False):
            roi_path = os.path.join(patient_path, roi)
            unique_name = f"{patient}_{roi}"
            
            raw_hdr_path = os.path.join(roi_path, "Raw.hdr")
            rgb_path = os.path.join(roi_path, "rgb.png")
            label_path = os.path.join(roi_path, "Label.png")
            cell_label_path = os.path.join(roi_path, "Cell-level Label.png")

            if not os.path.exists(raw_hdr_path):
                continue

            # -----------------------------------------------------------
            # A. Process Hyperspectral Data (0-1 Float Output)
            # -----------------------------------------------------------
            try:
                img_obj = envi.open(raw_hdr_path)
                data = img_obj.load() 
                h, w, d = data.shape
                spectral_cube = data.astype("float64")

                print(f"\n[DEBUG] {unique_name} Input Range: Min={np.min(spectral_cube):.4f}, Max={np.max(spectral_cube):.4f}")

                # Smoothing
                spectral_cube = savgol_filter(spectral_cube, 11, 2, axis=2)

                # Preprocessing
                if PERBAND == 1:
                    for i in range(d):
                        if OPTLIN:
                            spectral_cube[:,:,i] = optimized_linear_float(spectral_cube[:,:,i])
                        elif SNV_USED:
                            spectral_cube[:,:,i] = snv(spectral_cube[:,:,i])
                            spectral_cube[:,:,i] = percent_linear_float(spectral_cube[:,:,i], PERCENT_LINEAR_VALUE)
                        else:
                            spectral_cube[:,:,i] = percent_linear_float(spectral_cube[:,:,i], PERCENT_LINEAR_VALUE)
                else:
                    if OPTLIN:
                        for i in range(d):
                            spectral_cube[:,:,i] = optimized_linear_float(spectral_cube[:,:,i])
                    elif SNV_USED:
                        spectral_cube = snv(spectral_cube) 
                        spectral_cube = percent_linear_float(spectral_cube, PERCENT_LINEAR_VALUE)
                    else:
                        spectral_cube = percent_linear_float(spectral_cube, PERCENT_LINEAR_VALUE)
                
                print(f"[DEBUG] {unique_name} Output Range: Min={np.min(spectral_cube):.4f}, Max={np.max(spectral_cube):.4f}")

                # Save Datacube (Float32)
                output_envi_path = os.path.join(path_map["datacubes"], unique_name)
                driver = gdal.GetDriverByName("ENVI")
                output_ds = driver.Create(output_envi_path, w, h, d, gdal.GDT_Float32)
                
                for i in range(d):
                    output_ds.GetRasterBand(i + 1).WriteArray(spectral_cube[:, :, i])
                output_ds = None 

                # Update Header
                header_file = output_envi_path + '.hdr'
                metadata = envi.read_envi_header(header_file)
                metadata['default bands'] = f'{{{red_band_idx}, {green_band_idx}, {blue_band_idx}}}'
                metadata['wavelength units'] = 'nm'
                metadata['data type'] = 4 # Ensure ENVI knows it is float (4 = float32)
                metadata['wavelength'] = list(range(BAND_MIN, BAND_MAX + 1, STEP))
                envi.write_envi_header(header_file, metadata)

            except Exception as e:
                print(f"Error processing HSI for {unique_name}: {e}")

            # -----------------------------------------------------------
            # B. Process Labels and RGB (With Checks)
            # -----------------------------------------------------------
            
            if os.path.exists(rgb_path):
                shutil.copy(rgb_path, os.path.join(path_map["rgbs"], f"{unique_name}.png"))
            
            # Tissue Label
            tissue_npy = process_tissue_label(label_path, unique_name)
            if tissue_npy is not None:
                np.save(os.path.join(path_map["labels"], f"{unique_name}.npy"), tissue_npy)
                shutil.copy(label_path, os.path.join(path_map["labels"], f"{unique_name}.png"))
                
            # Cell Label
            cell_npy = process_cell_label(cell_label_path, unique_name)
            if cell_npy is not None:
                np.save(os.path.join(path_map["cell_labels"], f"{unique_name}.npy"), cell_npy)
                shutil.copy(cell_label_path, os.path.join(path_map["cell_labels"], f"{unique_name}.png"))

    print("Processing complete.")

if __name__ == "__main__":
    main()