import glob
import os
import json
import numpy as np
from sklearn.model_selection import KFold

# --- Configuration ---
LABEL_DIR = "/scratch/lt200353-pcllm/location/real_colon/labels_cleaned/"
SPLITS_OUT_DIR = "/scratch/lt200353-pcllm/location/real_colon/splits/"

def main():
    os.makedirs(SPLITS_OUT_DIR, exist_ok=True)
    
    # Grab all available label CSVs
    all_label_files = np.array(sorted(glob.glob(os.path.join(LABEL_DIR, "*.csv"))))
    
    if len(all_label_files) == 0:
        raise ValueError(f"No CSVs found in {LABEL_DIR}")

    # Setup K-Fold (n_splits=5)
    # random_state=42 ensures the exact same splits if you re-run this script
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_label_files)):
        train_files = all_label_files[train_idx].tolist()
        val_files = all_label_files[val_idx].tolist()
        
        split_data = {
            "train": train_files,
            "val": val_files
        }
        
        out_path = os.path.join(SPLITS_OUT_DIR, f"fold_{fold}.json")
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=4)
            
        print(f"Saved Fold {fold} -> {len(train_files)} train, {len(val_files)} val to {out_path}")

if __name__ == "__main__":
    main()
