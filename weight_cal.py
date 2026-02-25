import glob
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_safe_weights(file_pattern, ignore_index=21, smoothing='sqrt'):
    files = sorted(glob.glob(file_pattern))
    all_labels = []

    print("Loading data...")
    for f_path in files:
        try:
            data = np.load(f_path)
            # Flatten to 1D
            if data.ndim > 1 and data.shape[-1] > 1:
                labels = np.argmax(data, axis=-1)
            else:
                labels = data.flatten()
            all_labels.append(labels)
        except:
            continue

    if not all_labels:
        return None

    y_all = np.concatenate(all_labels)
    
    # --- STEP 1: Filter out the Ignore Index for calculation ---
    # We only want to calculate ratios based on valid data
    valid_mask = (y_all != ignore_index)
    y_valid = y_all[valid_mask]
    
    classes = np.unique(y_valid)
    classes.sort()
    
    print(f"Valid classes (excluding {ignore_index}): {classes}")

    # --- STEP 2: Compute Balanced Weights ---
    # raw_weights = n_samples / (n_classes * np.bincount(y))
    raw_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_valid)

    # --- STEP 3: Apply Smoothing (Fixing the "Extreme" values) ---
    if smoothing == 'sqrt':
        print("Applying Square Root smoothing...")
        final_weights = np.sqrt(raw_weights)
    elif smoothing == 'log':
        print("Applying Log smoothing...")
        final_weights = np.log(1.2 + raw_weights) # 1.2 adds stability
    else:
        final_weights = raw_weights

    # --- STEP 4: Build full weight array (including missing classes/ignore index) ---
    # We need an array of size 22 (0 to 21)
    max_class_id = 21 
    full_weight_tensor = np.ones(max_class_id + 1) 
    
    # Map computed weights to their indices
    for cls_idx, w in zip(classes, final_weights):
        full_weight_tensor[int(cls_idx)] = w
        
    # Explicitly set the ignore_index weight to 0 (good practice, though Loss ignores it anyway)
    full_weight_tensor[ignore_index] = 0.0

    print("\n--- Final Weights (Sample) ---")
    print(f"Class 0 (Background?): {full_weight_tensor[0]:.4f}")
    print(f"Class 1 (Action):      {full_weight_tensor[1]:.4f}")
    print(f"Class {ignore_index} (Ignored):     {full_weight_tensor[ignore_index]:.4f}")
    
    return full_weight_tensor

# Usage
path = "/scratch/lt200353-pcllm/THUMOS14/target_perframe/video_validation_0000*"
weights = get_safe_weights(path, ignore_index=21, smoothing='sqrt')
print(weights)
