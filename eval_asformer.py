import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from model.ASFormer import MyTransformer as ASFormer  # Adjust import based on your folder structure

# --- CONFIGURATION ---
class Config:
    # Paths
    MODEL_PATH = "/scratch/lt200353-pcllm/location/cas_colon/split_1/epoch-31.model" # Point to your best model
    FEATURE_DIR = "/scratch/lt200353-pcllm/location/cas_colon/features/"
    GT_DIR = "/scratch/lt200353-pcllm/location/cas_colon/ground_truth/"
    CSV_PATH = "/scratch/lt200353-pcllm/location/cas_colon/Video_Label.csv"
    MAPPING_FILE = "/scratch/lt200353-pcllm/location/cas_colon/mapping.txt"
    RAW_VIDEO_DIR = "/scratch/lt200353-pcllm/location/cas_colon/videos/" # Path to actual .mp4 files (for visualization)
    OUTPUT_DIR = "./eval_outputs"
    
    # Model Params (Must match training!)
    NUM_LAYERS = 10
    NUM_F_MAPS = 64
    INPUT_DIM = 2048
    NUM_CLASSES = 10 # 11 Classes + Unlabeled? Check your mapping.txt count
    
    # Target Video for Deep Dive
    TARGET_VID = "75"  
    FPS = 1.0 # Your extraction FPS
    TEST_SPLIT_PATH = "/scratch/lt200353-pcllm/location/cas_colon/splits/test.split1.bundle"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- UTILS ---

def load_mapping(mapping_file):
    mapping = {}
    id2name = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            class_name = parts[1]
            mapping[class_name] = class_id
            id2name[class_id] = class_name
    return mapping, id2name

def get_distinct_colors(n):
    # Generate distinct colors for visualization
    cm = plt.get_cmap('tab20')
    return [cm(1. * i / n) for i in range(n)]

# --- MAIN EVALUATION ---

def main():
    if not os.path.exists(Config.OUTPUT_DIR): os.makedirs(Config.OUTPUT_DIR)
    
    # 1. Load Mapping
    action_dict, id2name = load_mapping(Config.MAPPING_FILE)
    num_classes = len(action_dict)
    print(f"Loaded {num_classes} classes.")

    # 2. Load Model
    model = ASFormer(3, 10, 2, 2, 64, 2048, num_classes, 0.3).to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 3. Load Test Video List (FIXED)
    if not os.path.exists(Config.TEST_SPLIT_PATH):
        raise FileNotFoundError(f"Could not find test split file: {Config.TEST_SPLIT_PATH}")
        
    with open(Config.TEST_SPLIT_PATH, 'r') as f:
        test_vids = [line.strip() for line in f if line.strip()]
    
    print(f"Evaluating on {len(test_vids)} test videos...")
    
    all_preds = []
    all_targets = []
    target_video_data = None 

    for vid_id in test_vids:
        # Construct paths dynamically based on ID
        feat_path = os.path.join(Config.FEATURE_DIR, f"{vid_id}.npy")
        gt_path = os.path.join(Config.GT_DIR, f"{vid_id}.txt")
        
        if not os.path.exists(feat_path):
            print(f"Warning: Feature not found for {vid_id}")
            continue
        
        # Load Features
        features = np.load(feat_path)
        if features.shape[1] == 2048: features = features.T # Ensure (2048, T)
        
        # Load GT
        if not os.path.exists(gt_path): 
            print(f"Warning: GT not found for {vid_id}")
            continue
            
        with open(gt_path, 'r') as f:
            gt_labels_raw = [line.strip() for line in f.readlines()]
        
        # Convert GT to IDs
        gt_ids = [action_dict.get(l, -100) for l in gt_labels_raw]
        
        # Prepare Input
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.ones((1, 1, features.shape[1]), dtype=torch.float32).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(features_tensor, mask)
            prediction = output[-1] 
            pred_ids = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()

        # Truncate
        min_len = min(len(gt_ids), len(pred_ids))
        gt_ids = gt_ids[:min_len]
        pred_ids = pred_ids[:min_len]

        all_preds.extend(pred_ids)
        all_targets.extend(gt_ids)

        # Store Video 75 (Only if it is in the test set!)
        if vid_id == Config.TARGET_VID:
            target_video_data = {
                'vid_id': vid_id,
                'pred': pred_ids,
                'gt': gt_ids,
                'gt_raw': gt_labels_raw[:min_len]
            }
    # --- REPORT 1: PER CLASS F1 ---
    print("\n" + "="*30)
    print("REPORT 1: Per-Class Evaluation")
    print("="*30)
    
    # Filter out -100 (Ignore index) if present
    valid_indices = [i for i, t in enumerate(all_targets) if t != -100]
    clean_targets = [all_targets[i] for i in valid_indices]
    clean_preds = [all_preds[i] for i in valid_indices]
    
    target_names = [id2name[i] for i in range(num_classes) if i in set(clean_targets)]
    print(classification_report(clean_targets, clean_preds, target_names=target_names, digits=4))

    # --- REPORT 2 & 2.5: VISUALIZATION (GT vs PRED vs CSV) ---
    if target_video_data:
        print(f"\nGeneratign Visualization for Video {Config.TARGET_VID}...")
        visualize_video_timeline(target_video_data, Config.CSV_PATH, action_dict, id2name)
        
        # --- REPORT 3: RENDER VIDEO ---
        # Note: This requires the actual mp4 file to exist
        raw_vid_path = os.path.join(Config.RAW_VIDEO_DIR, f"{Config.TARGET_VID}.mp4") # Check extension
        if os.path.exists(raw_vid_path):
            render_overlay_video(raw_vid_path, target_video_data, id2name)
        else:
            print(f"Skipping video render: Could not find {raw_vid_path}")
    else:
        print(f"Warning: Video {Config.TARGET_VID} was not found in the feature directory.")

# --- VISUALIZATION HELPERS ---

def parse_csv_intervals_for_vis(csv_path, vid_id, fps, total_frames):
    """Re-parses the CSV to create a 'Ground Truth from CSV' timeline."""
    df = pd.read_csv(csv_path)
    # Ensure VideoID is string for matching
    df['VideoID'] = df['VideoID'].astype(str)
    row = df[df['VideoID'] == str(vid_id)]
    
    if row.empty: return ["Unknown"] * total_frames
    
    row = row.iloc[0]
    csv_labels = ["Unlabeled"] * total_frames
    
    # Parse columns similar to training script
    # We iterate columns to find time intervals
    for col in df.columns:
        if col in ['VideoID', 'Total Time', 'Note']: continue
        
        time_str = row[col]
        if pd.isna(time_str) or str(time_str).strip() in ['', '-', 'nan']: continue
        
        # Parse intervals
        segments = str(time_str).split('/')
        for seg in segments:
            try:
                start_s, end_s = seg.strip().split('-')
                start_f = int((int(start_s.split(':')[0])*60 + int(start_s.split(':')[1])) * fps)
                end_f = int((int(end_s.split(':')[0])*60 + int(end_s.split(':')[1])) * fps)
                
                # IMPORTANT: Apply the +1 fix here to match our new logic
                # But also visualizing "Raw CSV" might be useful without the fix to see the gaps
                # Let's apply the fix to see if it matches the TXT
                for i in range(start_f, min(end_f + 1, total_frames)):
                    csv_labels[i] = col
            except: pass
            
    return csv_labels

def visualize_temporal(ax, sequence, title, id2name, color_map, mapping):
    # Sequence is a list of IDs or Strings
    # Convert everything to IDs for plotting
    if len(sequence) > 0 and isinstance(sequence[0], str):
        seq_ids = [mapping.get(x, -100) for x in sequence]
    else:
        seq_ids = sequence

    # Expand to image for barcode plot
    # Shape: (1, T)
    data = np.array(seq_ids).reshape(1, -1)

    # Mask -100 for visualization so it doesn't mess up color scaling
    # We map -100 to a safe value (e.g. max class ID + 1) or just let it plot
    # Ideally, we rely on the legend loop to handle the specific label lookups

    ax.imshow(data, aspect='auto', cmap='tab20', interpolation='nearest', vmin=0, vmax=len(mapping))
    ax.set_title(title)
    ax.set_yticks([])

    # Custom Legend
    unique_ids = np.unique(seq_ids)
    patches = []
    for uid in unique_ids:
        # --- FIX: Skip -100 (Ignore Index) or -1 (Unknown) ---
        if uid == -100 or uid == -1:
            continue

        # Ensure uid is an integer for lookup
        uid_int = int(uid)

        if uid_int in id2name:
            label = id2name[uid_int]
            # Map ID to color (same logic as cmap)
            color = plt.cm.tab20(uid / len(mapping))
            patches.append(mpatches.Patch(color=color, label=label))

    return patches

"""def visualize_temporal(ax, sequence, title, id2name, color_map, mapping):
    # Sequence is a list of IDs or Strings
    # Convert everything to IDs for plotting
    if isinstance(sequence[0], str):
        seq_ids = [mapping.get(x, -1) for x in sequence]
    else:
        seq_ids = sequence

    # Expand to image for barcode plot
    # Shape: (1, T)
    data = np.array(seq_ids).reshape(1, -1)
    
    ax.imshow(data, aspect='auto', cmap='tab20', interpolation='nearest', vmin=0, vmax=len(mapping))
    ax.set_title(title)
    ax.set_yticks([])
    
    # Custom Legend
    # We only show legend for classes that appear
    unique_ids = np.unique(seq_ids)
    patches = []
    for uid in unique_ids:
        if uid == -1: continue # Skip unknown
        label = id2name[uid]
        # Map ID to color (approximate logic for cmap)
        color = plt.cm.tab20(uid / len(mapping)) 
        patches.append(mpatches.Patch(color=color, label=label))
    
    return patches"""

def visualize_video_timeline(data, csv_path, mapping, id2name):
    vid_id = data['vid_id']
    pred = data['pred']
    gt_txt = data['gt']
    
    # Generate CSV-based labels for comparison (Sanity Check 2.5)
    csv_labels = parse_csv_intervals_for_vis(csv_path, vid_id, Config.FPS, len(pred))
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=True)
    
    # 1. Prediction
    patches = visualize_temporal(axes[0], pred, f"Prediction (Model) - Video {vid_id}", id2name, None, mapping)
    
    # 2. GT from TXT
    visualize_temporal(axes[1], gt_txt, f"Ground Truth (From .txt file)", id2name, None, mapping)
    
    # 3. GT from CSV (Re-parsed)
    visualize_temporal(axes[2], csv_labels, f"Ground Truth (Direct from CSV Check)", id2name, None, mapping)
    
    # Add legend to the side
    fig.legend(handles=patches, loc='center right')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Make room for legend
    
    save_path = os.path.join(Config.OUTPUT_DIR, f"vis_video_{vid_id}.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close()

def render_overlay_video(video_path, data, id2name):
    # Open Source Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    save_path = os.path.join(Config.OUTPUT_DIR, f"render_{data['vid_id']}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    pred_seq = data['pred']
    gt_seq = data['gt']
    total_frames = len(pred_seq)
    
    # ASFormer operates on downsampled FPS (1.0). The video might be 30 FPS.
    # We need to map video frame index -> label index
    # Ratio = Video_FPS / Label_FPS
    ratio = fps / Config.FPS 
    
    frame_idx = 0
    print(f"Rendering video to {save_path}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Calculate which label corresponds to this frame
        label_idx = int(frame_idx / ratio)
        
        if label_idx < total_frames:
            # Get Strings
            pred_txt = id2name[pred_seq[label_idx]]
            gt_txt = id2name[gt_seq[label_idx]]
            
            # Colors (BGR)
            # Green if match, Red if mismatch
            color = (0, 255, 0) if pred_txt == gt_txt else (0, 0, 255)
            
            # Draw overlay box
            cv2.rectangle(frame, (0, 0), (400, 100), (0, 0, 0), -1)
            
            # Put Text
            cv2.putText(frame, f"GT: {gt_txt}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Pred: {pred_txt}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Optional: Add Frame count
            cv2.putText(frame, f"Sec: {int(frame_idx/fps)}", (width-150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print("Video rendering complete.")

if __name__ == "__main__":
    main()
