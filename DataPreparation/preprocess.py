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
OUTPUT_FOLDER = r"processed_dataset" # Output root folder ready for deep learning pipeline

# Preprocessing Flags
PERCENT_LINEAR_VALUE = 0  # Percentile for linear stretch (e.g., 2 for 2%)
PERBAND = 0               # 1: Process each band independently; 0: Process 3D cube as a whole.
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

def load_reference_spectrum(hdr_path):
    """
    Loads an ENVI reference file (Dark or White) and calculates 
    the spatial mean to return a 1D spectral vector.
    """
    try:
        if not os.path.exists(hdr_path):
            return None
        
        ref_obj = envi.open(hdr_path)
        ref_data = ref_obj.load() # Load into memory
        
        # Calculate spatial mean (average across Height and Width)
        # Result shape: (Bands,)
        mean_spectrum = np.mean(ref_data, axis=(0, 1))
        return mean_spectrum.astype(np.float64)
    except Exception as e:
        print(f"[ERROR] Failed to load reference {hdr_path}: {e}")
        return None

def optimized_linear_float(arr):
    """Simulates ENVI's Optimized Linear Stretch but outputs 0-1 float."""
    arr = arr.astype(np.float32)
    a, b = np.percentile(arr, (2.5, 99))
    c = a - 0.1 * (b - a)
    d = b + 0.5 * (b - a)
    
    if d - c == 0:
        return np.zeros_like(arr, dtype=np.float32)
        
    arr = (arr - c) / (d - c)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)

def percent_linear_float(arr, percent=2):
    """Standard linear stretch based on percentiles, outputs 0-1 float."""
    arr = arr.astype(np.float32)
    arr_min, arr_max = np.percentile(arr, (percent, 100 - percent))
    
    if arr_max - arr_min == 0:
        return np.zeros_like(arr, dtype=np.float32)

    arr = (arr - arr_min) / (arr_max - arr_min)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)

def snv(input_data):
    """Standard Normal Variate (SNV) correction."""
    output_data = np.zeros_like(input_data, dtype=np.float32)
    mean = np.mean(input_data)
    std = np.std(input_data)
    
    if std == 0:
        return output_data
        
    output_data = (input_data - mean) / std
    return output_data

def check_tissue_cleanliness(img, name):
    unique_vals = np.unique(img)
    allowed = {0, 255}
    found = set(unique_vals)
    if not found.issubset(allowed):
        unexpected = list(found - allowed)
        print(f"\n[WARNING] '{name}' Tissue Label is NOT clean. Unexpected: {unexpected}")

def process_tissue_label(path, unique_name):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    check_tissue_cleanliness(img, unique_name)
    _, thresh = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return thresh.astype(np.uint8)

def check_cell_cleanliness(img, name):
    pixels = img.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    expected_colors = np.array([[0,0,0], [0,0,255], [0,255,0], [255,0,0]]) # BGR
    dirty_colors = []
    for color in unique_colors:
        if not np.any(np.all(expected_colors == color, axis=1)):
            dirty_colors.append(color)
    if dirty_colors:
        print(f"\n[WARNING] '{name}' Cell Label is NOT clean. Found {len(dirty_colors)} unexpected colors.")

def process_cell_label(path, unique_name):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None: return None
    check_cell_cleanliness(img, unique_name)
    h, w, c = img.shape
    label_map = np.zeros((h, w), dtype=np.uint8)
    
    # BGR Masks
    label_map[cv2.inRange(img, np.array([0,0,100]), np.array([80,80,255])) > 0] = 1 # Red
    label_map[cv2.inRange(img, np.array([0,100,0]), np.array([80,255,80])) > 0] = 2 # Green
    label_map[cv2.inRange(img, np.array([100,0,0]), np.array([255,80,80])) > 0] = 3 # Blue
    return label_map

# ==============================================================================
#                               MAIN PROGRAM
# ==============================================================================

def main():
    # Setup Paths
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
        dark_ref_path = os.path.join(patient_path, "darkReference.hdr")
        white_ref_path = os.path.join(patient_path, "whiteReference.hdr")
        dark_spectrum = load_reference_spectrum(dark_ref_path)
        white_spectrum = load_reference_spectrum(white_ref_path)
        for roi in tqdm(roi_folders, desc=f"ROIs in {patient}", leave=False):
            roi_path = os.path.join(patient_path, roi)
            unique_name = f"{patient}_{roi}"
            
            # Paths
            raw_hdr_path = os.path.join(roi_path, "Raw.hdr")

            
            rgb_path = os.path.join(roi_path, "rgb.png")
            label_path = os.path.join(roi_path, "Label.png")
            cell_label_path = os.path.join(roi_path, "Cell-level Label.png")

            if not os.path.exists(raw_hdr_path):
                continue

            # -----------------------------------------------------------
            # A. Process Hyperspectral Data
            # -----------------------------------------------------------
            try:
                # 1. Load Raw Data
                img_obj = envi.open(raw_hdr_path)
                data = img_obj.load() 
                h, w, d = data.shape
                spectral_cube = data.astype("float64")
                print(f"\n[DEBUG] {unique_name}")

                # 2. Reflectance Correction (Raw -> Reflectance)
                # Formula: (Raw - Dark) / (White - Dark)


                if dark_spectrum is not None and white_spectrum is not None:
                    # Validate dimension
                    if len(dark_spectrum) == d and len(white_spectrum) == d:
                        # Reshape to (1, 1, Bands) for broadcasting
                        dark_3d = dark_spectrum.reshape(1, 1, d)
                        white_3d = white_spectrum.reshape(1, 1, d)
                        
                        # Apply Correction with Epsilon to avoid div/0
                        numerator = spectral_cube - dark_3d
                        denominator = white_3d - dark_3d
                        spectral_cube = numerator / (denominator + 1e-6)
                        
                    else:
                        print(f"[ERROR] Band mismatch: Raw({d}) vs Dark({len(dark_spectrum)}) vs White({len(white_spectrum)})")
                else:
                    print(f"[WARNING] Missing Dark or White reference for {unique_name}. Skipping reflectance correction.")

                # 3. Smoothing (Savitzky-Golay)
                spectral_cube = savgol_filter(spectral_cube, 11, 2, axis=2)

                # 4. Preprocessing (SNV / Stretching)
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
                        
                # 5. Save Datacube
                output_envi_path = os.path.join(path_map["datacubes"], unique_name)
                driver = gdal.GetDriverByName("ENVI")
                output_ds = driver.Create(output_envi_path, w, h, d, gdal.GDT_Float32)
                for i in range(d):
                    output_ds.GetRasterBand(i + 1).WriteArray(spectral_cube[:, :, i])
                output_ds = None 

                # 6. Update Header
                header_file = output_envi_path + '.hdr'
                metadata = envi.read_envi_header(header_file)
                metadata['default bands'] = f'{{{red_band_idx}, {green_band_idx}, {blue_band_idx}}}'
                metadata['wavelength units'] = 'nm'
                metadata['data type'] = 4 
                metadata['wavelength'] = list(range(BAND_MIN, BAND_MAX + 1, STEP))
                envi.write_envi_header(header_file, metadata)

            except Exception as e:
                print(f"Error processing HSI for {unique_name}: {e}")

            # -----------------------------------------------------------
            # B. Process Labels and RGB
            # -----------------------------------------------------------
            if os.path.exists(rgb_path):
                shutil.copy(rgb_path, os.path.join(path_map["rgbs"], f"{unique_name}.png"))
            
            tissue_npy = process_tissue_label(label_path, unique_name)
            if tissue_npy is not None:
                np.save(os.path.join(path_map["labels"], f"{unique_name}.npy"), tissue_npy)
                shutil.copy(label_path, os.path.join(path_map["labels"], f"{unique_name}.png"))
                
            cell_npy = process_cell_label(cell_label_path, unique_name)
            if cell_npy is not None:
                np.save(os.path.join(path_map["cell_labels"], f"{unique_name}.npy"), cell_npy)
                shutil.copy(cell_label_path, os.path.join(path_map["cell_labels"], f"{unique_name}.png"))

    print("Processing complete.")

if __name__ == "__main__":
    main()
