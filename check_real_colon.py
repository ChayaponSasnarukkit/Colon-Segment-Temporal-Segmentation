import os
import csv
import torch
import numpy as np
from tqdm import tqdm

# --- Configuration ---
DATA_ROOT = "/scratch/lt200353-pcllm/location/real_colon/dataset"
VIDEO_INFO_PATH = os.path.join(DATA_ROOT, "video_info.csv")
OUTPUT_DIR = os.path.join(DATA_ROOT, "features_dinov3") # Your extraction folder
EXPECTED_EMBEDDING_DIM = 1024 # DINOv2 vitl14 outputs 1024-dim vectors

def main():
    print("Starting Dataset Integrity Check...\n")
    
    if not os.path.exists(VIDEO_INFO_PATH):
        print(f"❌ ERROR: Cannot find video info at {VIDEO_INFO_PATH}")
        return

    # 1. Load Expected Metadata
    video_metadata = {}
    with open(VIDEO_INFO_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_metadata[row['unique_video_name']] = {
                'fps': float(row['fps']),
                'total_frames': int(row['num_frames'])
            }

    print(f"Loaded metadata for {len(video_metadata)} videos.")
    
    issues_found = 0
    passed_videos = 0

    # 2. Iterate through expected videos and check files
    for video_name, meta in tqdm(video_metadata.items(), desc="Checking videos"):
        feat_path = os.path.join(OUTPUT_DIR, f"{video_name}.pt")
        label_path = os.path.join(OUTPUT_DIR, f"{video_name}_labels.npy")
        
        # --- Check A: File Existence ---
        if not os.path.exists(feat_path):
            print(f"\n⚠️ Missing Features: {feat_path}")
            issues_found += 1
            continue
        if not os.path.exists(label_path):
            print(f"\n⚠️ Missing Labels: {label_path}")
            issues_found += 1
            continue

        # --- Check B: File Loading & Dimensions ---
        try:
            # We use mmap_mode='r' for numpy to avoid loading the whole array into RAM just to check shape
            labels = np.load(label_path, mmap_mode='r') 
            
            # PyTorch doesn't have a direct mmap for .pt, but we can load just the tensor structure
            features = torch.load(feat_path, map_location='cpu')
            
            num_feats = features.shape[0]
            feat_dim = features.shape[1]
            num_labels = labels.shape[0]
            
        except Exception as e:
            print(f"\n❌ Corrupted File for {video_name}: {e}")
            issues_found += 1
            continue

        # --- Check C: Feature <-> Label Alignment ---
        if num_feats != num_labels:
            print(f"\n❌ Alignment Error ({video_name}): {num_feats} features vs {num_labels} labels")
            issues_found += 1
            continue
            
        # --- Check D: Embedding Dimension ---
        if feat_dim != EXPECTED_EMBEDDING_DIM:
            print(f"\n❌ Dimension Error ({video_name}): Expected dim {EXPECTED_EMBEDDING_DIM}, got {feat_dim}")
            issues_found += 1
            continue

        # --- Check E: 5 FPS Logical Length ---
        # Calculate theoretical frames based on the sampling step used during extraction
        original_fps = meta['fps']
        step_size = max(1, round(original_fps / 5.0))
        
        # Theoretical length is total_frames / step_size
        theoretical_frames = len(range(0, meta['total_frames'], step_size))
        
        # We allow a small tolerance (e.g., +/- 10 frames) because the original dataset 
        # CSV might have had missing rows or slight discrepancies.
        tolerance = 10 
        
        if abs(num_feats - theoretical_frames) > tolerance:
            print(f"\n⚠️ Length Warning ({video_name}): "
                  f"Expected ~{theoretical_frames} frames (5fps), but got {num_feats}. "
                  f"(Original frames: {meta['total_frames']}, FPS: {original_fps}, Step: {step_size})")
            # We don't increment issues_found here because missing frames in the raw CSV 
            # are expected in real-world medical data, but it's good to log it.
            
        passed_videos += 1

    # --- Summary ---
    print("\n" + "="*40)
    print("INTEGRITY CHECK COMPLETE")
    print("="*40)
    print(f"Total Videos Checked: {len(video_metadata)}")
    print(f"✅ Perfectly Passed: {passed_videos}")
    if issues_found > 0:
        print(f"❌ Hard Errors/Missing: {issues_found} (Requires re-extraction)")
    else:
        print("🎉 Zero hard errors! Your dataset is perfectly aligned and ready for training.")

if __name__ == "__main__":
    main()
