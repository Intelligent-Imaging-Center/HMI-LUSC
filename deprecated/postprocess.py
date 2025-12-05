import os
import numpy as np
import csv
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from PIL import Image
import tqdm
from pathlib import Path
import imageio.v2 as iio
import cv2
import matplotlib.pyplot as plt

# ======================= CONFIGURATION =======================
# 1. Directories
TEST_OUTPUT_ROOT = "D:/Experiment/test_output"
GT_LABEL_DIR = "../processed_dataset/cell_labels"
FINAL_OUTPUT_ROOT = "../Experiment1/final_output_instance"
REPORT_FILE = os.path.join(FINAL_OUTPUT_ROOT, "performance_report_postprocessed.csv")

# 2. Models
MODELS_TO_PROCESS = ["CNN3D", "CNN2D", "RBF_SVM", "RF"]

# 3. Parameters from your original code
SOFT_PROB_THRESH = 0.5
CONNECTED_THRESH = 50
CONNECTIVITY_PARAM = 8

# 4. Classes & Colors
CLASSES = ["Non-Cell", "Normal Cell", "Ill Cell", "Background"]
COLOR_MAP = {
    0: [0, 0, 0],       # Non-Cell
    1: [255, 0, 0],     # Normal Cell
    2: [0, 255, 0],     # Ill Cell
    3: [0, 0, 255],     # Background
    4: [255, 255, 255]  # Undefined (White)
}
# =============================================================

def generate_file_list(dir_path, extension):
    if not os.path.exists(dir_path):
        return []
    files = [f for f in os.listdir(dir_path) if f.endswith(extension)]
    files.sort()
    return files

def calculate_metrics(conf_matrix):
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp
    tn = conf_matrix.sum() - tp - fp - fn

    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        dice = 2 * tp / (2 * tp + fn + fp)

    return np.nan_to_num(accuracy), np.nan_to_num(specificity), np.nan_to_num(sensitivity), np.nan_to_num(dice)

def load_ground_truth(gt_dir, file_stem):
    npy_path = os.path.join(gt_dir, f"{file_stem}.npy")
    if os.path.exists(npy_path):
        data = np.load(npy_path)
        if hasattr(data, 'get'): data = data.get()
        return data

    png_path = os.path.join(gt_dir, f"{file_stem}.png")
    if os.path.exists(png_path):
        img = np.array(iio.imread(png_path))
        if img.ndim == 3: img = img[:, :, 0]
        img[img == 255] = 1 
        return img.astype(np.uint8)
    return None

def save_visualization(prediction_class, save_path):
    h, w = prediction_class.shape
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        vis_img[prediction_class == class_idx] = color
    Image.fromarray(vis_img).save(save_path)

def apply_postprocessing_logic(prob_map, h, w, use_softmax=False, filename=""):
    """
    Optimized implementation of your original logic using Bounding Boxes (ROI)
    to avoid full-image processing in loops.
    """
    # --- 1. PREPARE PROBABILITIES ---
    prob_flat = prob_map.reshape(-1, 4)
    if use_softmax:
        prob_tensor = torch.from_numpy(prob_flat).float()
        prob_soft = F.softmax(prob_tensor, dim=1).numpy()
    else:
        prob_soft = prob_flat
    prob_img = prob_soft.reshape(h, w, 4)

    # --- 2. THRESHOLDING & FILTERING ---
    filtered_class = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Process each class type
    for type_idx in range(4):
        data = prob_img[:, :, type_idx]
        _, thresh = cv2.threshold(data, SOFT_PROB_THRESH, 1, cv2.THRESH_BINARY)
        thresh_uint = thresh.astype(np.uint8)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_uint, connectivity=CONNECTIVITY_PARAM)
        
        # Fast vector filtering using stats (no loop needed)
        # Stats cols: [x, y, w, h, area]
        valid_indices = np.where(stats[:, 4] > CONNECTED_THRESH)[0]
        # Remove background label 0 from valid indices
        valid_indices = valid_indices[valid_indices != 0]
        
        # Create mask for this class
        image_filtered = np.isin(labels, valid_indices).astype(np.uint8) * 255
        filtered_class[:, :, type_idx] = image_filtered

    # --- 3. INITIAL COMBINATION (NOISY MAP) ---
    filtered_class[filtered_class == 255] = 1
    filtered_sum = np.sum(filtered_class, axis=2)
    
    output_type = np.full((h, w), 4, dtype=np.uint8) # Default 4
    # Assign classes: 0 -> 1 -> 2 -> 3 priority (same as your code)
    output_type[filtered_class[:, :, 0] == 1] = 0
    output_type[filtered_class[:, :, 1] == 1] = 1
    output_type[filtered_class[:, :, 2] == 1] = 2
    output_type[filtered_class[:, :, 3] == 1] = 3
    # Any pixel not claimed (sum==0) is already 4

    # --- 4. GAP FILLING (UNDEFINED PIXELS) ---
    undefined_mask = (output_type == 4).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(undefined_mask, connectivity=CONNECTIVITY_PARAM)
    
    # Use tqdm to trace progress
    if num_labels > 1:
        # Loop over components (skip background 0)
        # OPTIMIZATION: Use 'stats' to get bounding box. Do not process full image.
        for i in tqdm.tqdm(range(1, num_labels), desc=f"Filling Gaps ({filename})", leave=False):
            x, y, w_box, h_box, _ = stats[i]
            
            # Add padding of 1 pixel to check neighbors
            y1 = max(0, y - 2)
            y2 = min(h, y + h_box + 2)
            x1 = max(0, x - 2)
            x2 = min(w, x + w_box + 2)
            
            # Slice ROIs
            roi_labels = labels[y1:y2, x1:x2]
            roi_output = output_type[y1:y2, x1:x2]
            
            # Create mask relative to ROI
            label_mask_roi = (roi_labels == i).astype(np.uint8)
            
            # Dilate
            kernel = np.ones((3, 3), np.uint8)
            dilated_mask_roi = cv2.dilate(label_mask_roi, kernel, iterations=1)
            
            # Find neighbors in ROI
            neighbors_roi = dilated_mask_roi - label_mask_roi
            
            # Get types from neighbors
            neighbor_types = np.unique(roi_output[neighbors_roi == 1])
            neighbor_types = neighbor_types[neighbor_types != 4] # Ignore other undefined
            neighbor_types = np.sort(neighbor_types)
            
            # Logic: If >1 type and 0 exists, pick index 1. Else index 0.
            if len(neighbor_types) > 0:
                if len(neighbor_types) > 1 and 0 in neighbor_types:
                    chosen_type = neighbor_types[1]
                else:
                    chosen_type = neighbor_types[0]
                
                # Update GLOBAL output using ROI mask
                # We need to map ROI mask back to global coordinates
                # Actually, we can just write to roi_output if it's a view? 
                # No, numpy slice writes reflect in original array.
                roi_output[label_mask_roi == 1] = chosen_type

    # --- 5. MAJORITY VOTING (ILL VS NORMAL) ---
    # Merge 1 and 2
    cell_pixel = ((output_type == 1) | (output_type == 2)).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cell_pixel, connectivity=CONNECTIVITY_PARAM)
    
    if num_labels > 1:
        for i in tqdm.tqdm(range(1, num_labels), desc=f"Majority Vote ({filename})", leave=False):
            x, y, w_box, h_box, _ = stats[i]
            
            # Slice ROI (No padding needed for voting, just reading values)
            y1, y2 = y, y + h_box
            x1, x2 = x, x + w_box
            
            roi_labels = labels[y1:y2, x1:x2]
            roi_output = output_type[y1:y2, x1:x2]
            
            label_mask_roi = (roi_labels == i)
            
            # Extract values for this cell
            cell_vals = roi_output[label_mask_roi]
            
            ill_num = np.count_nonzero(cell_vals == 2)
            non_ill_num = np.count_nonzero(cell_vals == 1)
            
            # Vote
            if ill_num >= non_ill_num:
                roi_output[label_mask_roi] = 2
            else:
                roi_output[label_mask_roi] = 1

    return output_type

def process_models():
    # Initialize CSV
    header = ['File', 'Model', 'Class', 'Accuracy', 'Specificity', 'Sensitivity', 'Dice']
    with open(REPORT_FILE, 'w', encoding='utf-8', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(header)
        
        model_total_confusion = {m: np.zeros((4, 4), dtype=np.int64) for m in MODELS_TO_PROCESS}

        for model_name in MODELS_TO_PROCESS:
            print(f"\n================ Processing Model: {model_name} ================")
            
            prob_dir = os.path.join(TEST_OUTPUT_ROOT, model_name, "output")
            model_dest_dir = os.path.join(FINAL_OUTPUT_ROOT, model_name)
            os.makedirs(model_dest_dir, exist_ok=True)
            
            use_softmax = "CNN" in model_name # Heuristic for logits vs probs

            file_list = generate_file_list(prob_dir, '.npy')
            if not file_list:
                print(f"Warning: No files in {prob_dir}")
                continue

            # Main File Loop
            for filename in tqdm.tqdm(file_list, desc=f"Total Progress {model_name}"):
                file_stem = Path(filename).stem
                
                # 1. Load Probabilities
                prob_raw = np.load(os.path.join(prob_dir, filename))
                if hasattr(prob_raw, 'get'): prob_raw = prob_raw.get()

                # 2. Load GT for Shape Info
                gt_img = load_ground_truth(GT_LABEL_DIR, file_stem)
                if gt_img is None:
                    print(f"Skipping {filename}: No GT found (needed for dimensions)")
                    continue
                
                H, W = gt_img.shape
                
                # 3. Reshape/Validate
                if prob_raw.ndim == 2:
                    if prob_raw.shape[0] == H * W:
                        prob_map = prob_raw.reshape(H, W, 4)
                    else:
                        print(f"Size mismatch {file_stem}: GT {H}x{W} vs Data {prob_raw.shape}")
                        continue
                else:
                    prob_map = prob_raw

                # 4. RUN OPTIMIZED LOGIC
                final_class_map = apply_postprocessing_logic(prob_map, H, W, use_softmax, file_stem)

                # 5. Save
                np.save(os.path.join(model_dest_dir, f"{file_stem}.npy"), final_class_map)
                save_visualization(final_class_map, os.path.join(model_dest_dir, f"{file_stem}.tif"))

                # 6. Metrics
                y_true = gt_img.flatten()
                y_pred = final_class_map.flatten()
                
                matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
                model_total_confusion[model_name] += matrix
                
                acc, spec, sens, dice = calculate_metrics(matrix)
                for i, cls_name in enumerate(CLASSES):
                    writer.writerow([file_stem, model_name, cls_name, 
                                     f"{acc[i]:.4f}", f"{spec[i]:.4f}", f"{sens[i]:.4f}", f"{dice[i]:.4f}"])

            # Model Summary
            total_acc, total_spec, total_sens, total_dice = calculate_metrics(model_total_confusion[model_name])
            for i, cls_name in enumerate(CLASSES):
                 writer.writerow(["TOTAL_SUMMARY", model_name, cls_name, 
                                  f"{total_acc[i]:.4f}", f"{total_spec[i]:.4f}", f"{total_sens[i]:.4f}", f"{total_dice[i]:.4f}"])

    print(f"\nProcessing Complete. Report: {REPORT_FILE}")

if __name__ == "__main__":
    process_models()