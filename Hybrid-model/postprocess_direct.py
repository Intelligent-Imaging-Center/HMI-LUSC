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

# ======================= CONFIGURATION =======================
# 1. Input Directory: Root directory containing subfolders for each model
#    (Structure assumed: D:/Experiment/test_output/CNN3D/output/*.npy)
TEST_OUTPUT_ROOT = "D:/Experiment/test_output"

# 2. Ground Truth Directory: For reshaping and metrics
#    (Supports .npy or .png/.tif labels)
GT_LABEL_DIR = "../processed_dataset/cell_labels"

# 3. Output Destination: Where the processed files will be saved
#    Structure will be: D:/Experiment/test_output/output/CNN3D/*.{npy, tif}
FINAL_OUTPUT_ROOT = "../Experiment1/final_output"

# 4. Report File: CSV file path
REPORT_FILE = os.path.join(FINAL_OUTPUT_ROOT, "performance_report_integrated.csv")

# 5. Models to process
MODELS_TO_PROCESS = ["CNN3D", "CNN2D", "RBF_SVM", "RF"]

# 6. Class Definitions & Colors
CLASSES = ["Non-Cell", "Normal Cell", "Ill Cell", "Background"]
COLOR_MAP = {
    0: [0, 0, 0],       # Non-Cell (Black)
    1: [255, 0, 0],     # Normal Cell (Red)
    2: [0, 255, 0],     # Ill Cell (Green)
    3: [0, 0, 255]      # Background (Blue)
}
# =============================================================

def generate_file_list(dir_path, extension):
    if not os.path.exists(dir_path):
        return []
    files = [f for f in os.listdir(dir_path) if f.endswith(extension)]
    files.sort()
    return files

def calculate_metrics(conf_matrix):
    """
    Calculate metrics from multiclass confusion matrix.
    Returns: Accuracy, Specificity, Sensitivity, Dice per class
    """
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp
    tn = conf_matrix.sum() - tp - fp - fn

    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn) # Recall
        dice = 2 * tp / (2 * tp + fn + fp) # F1 Score equivalent

    # Replace NaNs with 0
    accuracy = np.nan_to_num(accuracy)
    specificity = np.nan_to_num(specificity)
    sensitivity = np.nan_to_num(sensitivity)
    dice = np.nan_to_num(dice)
    
    return accuracy, specificity, sensitivity, dice

def save_visualization(prediction_class, save_path):
    # Ensure it's on CPU and numpy before visualization
    if hasattr(prediction_class, 'get'): # Check for cupy array
        prediction_class = prediction_class.get()
    
    h, w = prediction_class.shape
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in COLOR_MAP.items():
        vis_img[prediction_class == class_idx] = color
        
    im = Image.fromarray(vis_img)
    im.save(save_path)

def load_ground_truth(gt_dir, file_stem):
    """
    Robustly load ground truth from .npy or image files (.png/.tif)
    Returns: numpy array of shape (H, W) or None if not found
    """
    # 1. Try .npy
    npy_path = os.path.join(gt_dir, f"{file_stem}.npy")
    if os.path.exists(npy_path):
        data = np.load(npy_path)
        if hasattr(data, 'get'): data = data.get()
        return data

    # 2. Try .png (Common for cell_labels)
    png_path = os.path.join(gt_dir, f"{file_stem}.png")
    if os.path.exists(png_path):
        # Read image
        img = np.array(iio.imread(png_path))
        
        # Process if 3 channels (e.g., if saved as RGB but actually a mask)
        if img.ndim == 3:
            img = img[:, :, 0] # Assume Red channel holds the label
            
        # Standardize labels (User's utils.py logic: 255 -> 1)
        img[img == 255] = 1 
        return img.astype(np.uint8)
        
    return None

def process_models():
    # Initialize CSV
    header = ['File', 'Model', 'Class', 'Accuracy', 'Specificity', 'Sensitivity', 'Dice']
    
    with open(REPORT_FILE, 'w', encoding='utf-8', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(header)
        
        model_total_confusion = {m: np.zeros((4, 4), dtype=np.int64) for m in MODELS_TO_PROCESS}

        for model_name in MODELS_TO_PROCESS:
            print(f"\n================ Processing Model: {model_name} ================")
            
            # Input Directory (Probabilities)
            prob_dir = os.path.join(TEST_OUTPUT_ROOT, model_name, "output")
            
            # Output Directory (Unified for labels and images)
            model_dest_dir = os.path.join(FINAL_OUTPUT_ROOT, model_name)
            os.makedirs(model_dest_dir, exist_ok=True)
            
            file_list = generate_file_list(prob_dir, '.npy')
            
            if not file_list:
                print(f"Warning: No .npy files found in {prob_dir}")
                continue
                
            for filename in tqdm.tqdm(file_list, desc=f"{model_name}"):
                file_stem = Path(filename).stem
                
                # 1. Load Prediction
                prob_path = os.path.join(prob_dir, filename)
                pred_raw = np.load(prob_path)
                
                # Handle Cupy
                if hasattr(pred_raw, 'get'):
                    pred_raw = pred_raw.get()
                
                # 2. Convert to Class Labels
                if pred_raw.ndim == 3:
                    pred_class = np.argmax(pred_raw, axis=2).astype(np.uint8)
                elif pred_raw.ndim == 2:
                    if pred_raw.shape[1] == 4: 
                         pred_class = np.argmax(pred_raw, axis=1).astype(np.uint8)
                    else:
                        pred_class = pred_raw.astype(np.uint8)
                else: 
                    pred_class = pred_raw.astype(np.uint8)

                # 3. Load Ground Truth for Reshaping & Metrics
                y_true_img = load_ground_truth(GT_LABEL_DIR, file_stem)
                
                if y_true_img is not None:
                    # Reshape Prediction if it was flattened (SVM/RF)
                    if y_true_img.ndim == 2:
                        H, W = y_true_img.shape
                        if pred_class.size == H * W:
                            pred_class = pred_class.reshape(H, W)
                        elif pred_class.shape != (H, W):
                            print(f"Error: Size mismatch {file_stem}. GT: {H*W}, Pred: {pred_class.size}")
                            continue
                    
                    # 4. Save processed label (.npy) to unified folder
                    np.save(os.path.join(model_dest_dir, f"{file_stem}.npy"), pred_class)
                    
                    # 5. Save Visualization (.tif) to unified folder
                    save_visualization(pred_class, os.path.join(model_dest_dir, f"{file_stem}.tif"))
                    
                    # 6. Metrics
                    y_true_flat = y_true_img.flatten()
                    y_pred_flat = pred_class.flatten()
                    
                    # Validate sizes before confusion matrix
                    if y_true_flat.shape[0] == y_pred_flat.shape[0]:
                        matrix = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1, 2, 3])
                        model_total_confusion[model_name] += matrix
                        
                        acc, spec, sens, dice = calculate_metrics(matrix)
                        for i, cls_name in enumerate(CLASSES):
                            writer.writerow([file_stem, model_name, cls_name, 
                                             f"{acc[i]:.4f}", f"{spec[i]:.4f}", f"{sens[i]:.4f}", f"{dice[i]:.4f}"])
                    else:
                         print(f"Skipping metrics for {file_stem}: Length mismatch after reshape.")
                else:
                    print(f"Ground truth not found for {filename} in {GT_LABEL_DIR}. Cannot determine shape for visualization.")

            # Write Model Summary
            total_acc, total_spec, total_sens, total_dice = calculate_metrics(model_total_confusion[model_name])
            for i, cls_name in enumerate(CLASSES):
                 writer.writerow(["TOTAL_SUMMARY", model_name, cls_name, 
                                  f"{total_acc[i]:.4f}", f"{total_spec[i]:.4f}", f"{total_sens[i]:.4f}", f"{total_dice[i]:.4f}"])

    print(f"\nProcessing Complete. Report saved to: {REPORT_FILE}")
    print(f"Processed files saved to: {FINAL_OUTPUT_ROOT}")

if __name__ == "__main__":
    process_models()