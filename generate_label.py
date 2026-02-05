import pandas as pd
import numpy as np
import os
import glob
import random

# --- CONFIGURATION ---
CSV_PATH = "/scratch/lt200353-pcllm/location/cas_colon/Video_Label.csv"   # Your metadata file
FEATURE_DIR = "/scratch/lt200353-pcllm/location/cas_colon/features/"         # Where your .npy files are
GT_DIR = "/scratch/lt200353-pcllm/location/cas_colon/ground_truth"           # Output for .txt labels
SPLIT_DIR = "/scratch/lt200353-pcllm/location/cas_colon/splits"       # Output for split bundles
MAPPING_FILE = "/scratch/lt200353-pcllm/location/cas_colon/mapping.txt"

# Must match extraction FPS (You used 1 fps previously)
FPS = 1.0 

# Define classes strictly in order of your CSV columns
CLASSES = [
    'Terminal_Ileum', 'Cecum', 'Ascending_Colon', 'Hepatic_Flexure',
    'Transverse_Colon', 'Splenic_Flexure', 'Descending_Colon', 
    'Sigmoid_Colon', 'Rectum', 'Anal_Canal', 'Equipment'
]

def parse_time_intervals(time_str):
    """Parses '0:00-0:09' or '0:17-0:23/ 0:33-1:06' into seconds."""
    if pd.isna(time_str) or str(time_str).strip() in ['', '-', 'nan']:
        return []
    
    intervals = []
    # Handle multiple segments separated by '/'
    segments = str(time_str).split('/')
    
    for seg in segments:
        seg = seg.strip()
        if not seg: continue
        try:
            start_s, end_s = seg.split('-')
            def to_sec(t):
                parts = t.strip().split(':')
                return int(parts[0]) * 60 + int(parts[1])
            intervals.append((to_sec(start_s), to_sec(end_s)))
        except Exception as e:
            print(f"  Warning: Could not parse '{seg}' - {e}")
            continue
    return intervals

def main():
    if not os.path.exists(GT_DIR): os.makedirs(GT_DIR)
    if not os.path.exists(SPLIT_DIR): os.makedirs(SPLIT_DIR)
    if not os.path.exists(os.path.dirname(MAPPING_FILE)): os.makedirs(os.path.dirname(MAPPING_FILE))

    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    generated_files = []

    for _, row in df.iterrows():
        vid_id = str(row['VideoID'])
        
        # 1. Load corresponding Feature file to get exact frame count
        # This ensures labels length == features length (Crucial for ASFormer)
        npy_path = os.path.join(FEATURE_DIR, f"{vid_id}.npy")
        
        if not os.path.exists(npy_path):
            print(f"Skipping Video {vid_id}: Feature file {npy_path} not found.")
            continue
            
        try:
            # Load only shape info to be fast
            features = np.load(npy_path) 
            # Check shape orientation. ASFormer features usually (Feat, Time) or (Time, Feat)
            # We assume the larger dimension is Time if it's not 2048
            if features.shape[0] == 2048:
                num_frames = features.shape[1]
            else:
                num_frames = features.shape[0]
                
            # 2. Initialize Labels
            # Default to "Unlabeled" or a specific background class
            labels = ["Unlabeled"] * num_frames
            
            # 3. Fill Labels based on CSV
            for cls in CLASSES:
                if cls not in row: continue
                intervals = parse_time_intervals(row[cls])
                
                for (start_sec, end_sec) in intervals:
                    # Convert seconds to frame indices
                    start_idx = int(start_sec * FPS)
                    end_idx = int(end_sec * FPS)
                    
                    # Clamp to video length
                    start_idx = max(0, min(start_idx, num_frames))
                    end_idx = max(0, min(end_idx, num_frames))
                    
                    # Assign Label
                    for i in range(start_idx, end_idx):
                        labels[i] = cls
                        
            # 4. Save .txt file
            gt_path = os.path.join(GT_DIR, f"{vid_id}.txt")
            with open(gt_path, 'w') as f:
                for l in labels:
                    f.write(f"{l}\n")
            
            generated_files.append(vid_id)
            print(f"Generated labels for Video {vid_id} ({num_frames} frames)")
            
        except Exception as e:
            print(f"Error processing {vid_id}: {e}")

    # --- Generate Helper Files for ASFormer ---
    
    # 1. Mapping.txt
    print(f"\nGenerating {MAPPING_FILE}...")
    with open(MAPPING_FILE, 'w') as f:
        # Add classes + Unlabeled
        all_classes = CLASSES + ['Unlabeled']
        for i, cls in enumerate(all_classes):
            f.write(f"{i} {cls}\n")
            
    # 2. Splits (Train/Test bundles)
    print("Generating Split files...")
    random.shuffle(generated_files)
    split_idx = int(len(generated_files) * 0.8)
    train_vids = generated_files[:split_idx]
    test_vids = generated_files[split_idx:]
    
    with open(os.path.join(SPLIT_DIR, "train.split1.bundle"), 'w') as f:
        f.write('\n'.join(train_vids))
        
    with open(os.path.join(SPLIT_DIR, "test.split1.bundle"), 'w') as f:
        f.write('\n'.join(test_vids))
        
    print("\nDone! Dataset is ready for training.")
    print(f"Run command: python main.py --action train --dataset Colon --split 1")

if __name__ == "__main__":
    main()
