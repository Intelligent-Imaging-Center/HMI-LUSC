import os
from pathlib import Path
import sys
import imageio.v2 as iio
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
ill_label_dir = "./label_generation_sample/ill"
cell_label_dir = "./label_generation_sample/cell"         # Expects .npy files
background_label_dir = "./label_generation_sample/background"
target_dir = "./label_generation_sample/final_label"      # Output folder

# --- HELPER FUNCTIONS ---

def generate_file_list(directory, extension):
    """Generates a list of files with specific extension in a directory."""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return []
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
    files.sort()
    return files

def get_file_name(path):
    return Path(path).stem

def read_tif_img(file):
    return np.array(iio.imread(file), dtype=np.uint8)

def process_ill_img(img):
    """
    Processes the 'ill' image.
    Logic: Thresholds values >= 125 to 1.
    """
    if len(img.shape) > 2:
        # Take Red channel if RGB
        r_img = img[:, :, 0]
        mask = np.where(r_img >= 125, 1, 0)
        return mask
    else:
        mask = np.where(img >= 125, 1, 0)
        return mask

def process_background_img(img):
    """
    Processes the 'background' image.
    Logic: Keeps Red channel, but removes if Green channel > 0 (removes white).
    """
    if len(img.shape) > 2:
        r_img = img[:, :, 0]
        g_img = img[:, :, 1]
        
        # Create mask where Red is present
        bg_mask = np.where(r_img != 0, 1, 0)
        
        # Remove areas where Green is present (which implies White/Yellow, not pure Red)
        # This preserves the logic: r_img[img[:,:,1]>0]=0
        bg_mask[g_img > 0] = 0
        return bg_mask
    else:
        # Fallback for grayscale background images
        mask = np.where(img >= 50, 1, 0)
        return mask

# --- MAIN EXECUTION ---

# 1. Setup Output Directory
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

# 2. Match Files across all three folders
# We base the list on the 'ill' folder, then check if 'cell' and 'background' exist.
base_files = generate_file_list(ill_label_dir, 'tif')
valid_files = []

for f in base_files:
    filename = get_file_name(f)
    cell_path = os.path.join(cell_label_dir, filename + ".npy")
    bg_path = os.path.join(background_label_dir, filename + ".tif")
    
    if os.path.exists(cell_path) and os.path.exists(bg_path):
        valid_files.append(filename)
    else:
        print(f"Skipping {filename}: Missing corresponding cell (.npy) or background (.tif) file.")

print(f"Found {len(valid_files)} complete sets of files.")

# 3. User Confirmation
if len(valid_files) > 0:
    confirm = input(f"Ready to process {len(valid_files)} file sets. Enter 1 to proceed: ")
    if int(confirm) == 1:
        for filename in valid_files:
            print(f"Processing: {filename}")
            
            # --- LOAD DATA ---
            # Load Ill (TIF)
            ill_raw = read_tif_img(os.path.join(ill_label_dir, filename + ".tif"))
            ill_mask = process_ill_img(ill_raw)
            
            # Load Cell (NPY)
            cell_mask = np.load(os.path.join(cell_label_dir, filename + ".npy"))
            
            # Load Background (TIF)
            bg_raw = read_tif_img(os.path.join(background_label_dir, filename + ".tif"))
            bg_mask = process_background_img(bg_raw)
            
            # --- COMBINE LABELS ---
            # Step A: Combine Cell (1) and Ill (2)
            # Logic: If Cell is 1, check if Ill is 1. If so -> 2. Else -> 1.
            # If Cell is 0, it remains 0 (even if Ill mask is true, because an ill marker outside a cell is invalid).
            temp_label = np.where(cell_mask == 1, cell_mask + ill_mask, 0) 
            # Note: The above creates 2 where cell=1 & ill=1. It creates 1 where cell=1 & ill=0.
            
            # Step B: Apply Background (3)
            # Logic: If Background mask is 1, overwrite everything to 3.
            final_label = np.where(bg_mask == 1, 3, temp_label)
            
            final_label = final_label.astype(np.uint8)
            print(f"  -> Unique values in output: {np.unique(final_label)}")
            
            # --- GENERATE VISUALIZATION ---
            # Create RGB image for easy viewing
            h, w = final_label.shape
            vis_img = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Channel 0 (Red) = Label 1 (Non-ill Cell)
            vis_img[:, :, 0] = np.where(final_label == 1, 255, 0)
            
            # Channel 1 (Green) = Label 2 (Ill Cell)
            vis_img[:, :, 1] = np.where(final_label == 2, 255, 0)
            
            # Channel 2 (Blue) = Label 3 (Background)
            vis_img[:, :, 2] = np.where(final_label == 3, 255, 0)
            
            # --- SAVE FILES ---
            # Save Raw Label (TIF)
            im = Image.fromarray(final_label)
            im.save(os.path.join(target_dir, filename + ".tif"))
            
            # Save Visualization (PNG)
            graphim = Image.fromarray(vis_img)
            graphim.save(os.path.join(target_dir, filename + ".png"))

    print("Processing complete.")
else:
    print("No valid file pairs found.")