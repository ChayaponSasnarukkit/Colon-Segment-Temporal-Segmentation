import torch
import numpy as np
import random
import os
import pandas as pd
from model.ASFormer import Trainer
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# --- Helper to parse time strings like "1:23" to seconds ---

def time_to_seconds(t_str):
    
    #Parses 'MM:SS' or 'HH:MM:SS' to total seconds.
    #Returns 0 on failure.
    
    try:
        parts = t_str.strip().split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, AttributeError):
        pass
    return 0

class BatchGenerator(object):
    def __init__(self, actions_dict, csv_path, features_path, target_fps=1):
        self.actions_dict = actions_dict
        self.features_path = features_path
        self.csv_path = csv_path

        # Metadata
        self.csv_fps = 60.0
        self.target_fps = target_fps

        # Load CSV and clean columns
        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]

        # OPTIMIZATION: Create a dictionary for O(1) lookups instead of O(N) filtering
        # This maps str(VideoID) -> row_data (dict)
        self.video_info = self.df.set_index(self.df['VideoID'].astype(str)).to_dict('index')

        self.list_of_examples = list(self.video_info.keys())
        self.index = 0
        self.reset()

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def next_batch(self, batch_size, *args):
        batch_ids = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        final_batch_ids = [] # To keep track of valid IDs only

        for vid in batch_ids:
            # --- 0. Retrieve Metadata (Optimized) ---
            if str(vid) not in self.video_info:
                continue
            row = self.video_info[str(vid)]
            total_frames_csv = int(row['TotalFrames'])

            # --- 1. Load Features ---
            feat_path = os.path.join(self.features_path, f"{vid}.pt")
            try:
                features = torch.load(feat_path, map_location='cpu')
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
            except FileNotFoundError:
                print(f"⚠️ Feature file not found: {feat_path}")
                continue

            # --- 2. Robust Shape Handling ---
            # We need shape (Dim, Time).
            # Heuristic: Find which dim is closer to the CSV frame count.
            shape = features.shape
            if len(shape) == 2:
                # Calculate difference between dimensions and expected time
                diff_dim0 = abs(shape[0] - total_frames_csv)
                diff_dim1 = abs(shape[1] - total_frames_csv)

                # If dim0 is closer to frame count (or if dim0 is very large vs dim1),
                # then input is likely (Time, Dim). We need to transpose.
                if diff_dim0 < diff_dim1:
                    features = features.T

            # Now features is (Dim, Time)

            # --- 3. Calculate Target Length ---
            # Duration in seconds based on CSV FPS
            duration_sec = total_frames_csv / self.csv_fps

            # New length at target_fps
            target_len = int(duration_sec * self.target_fps)
            if target_len < 1: target_len = 1

            # --- 4. Resample Features ---
            curr_feat_len = features.shape[1]
            feat_indices = np.linspace(0, curr_feat_len - 1, target_len).astype(int)
            features = features[:, feat_indices]

            # --- 5. Generate & Resample Labels ---
            full_res_labels = np.full(total_frames_csv, -100, dtype=int)

            for action_name, action_id in self.actions_dict.items():
                if action_name in row and not pd.isna(row[action_name]):
                    time_entry = str(row[action_name])
                    ranges = time_entry.split('/')

                    for rng in ranges:
                        rng = rng.strip()
                        if '-' in rng:
                            try:
                                start_str, end_str = rng.split('-')

                                start_s = time_to_seconds(start_str)
                                end_s = time_to_seconds(end_str)

                                # FIX: Add +1 to end second to cover the full duration
                                # "0:09" usually means the segment ends when the clock hits 0:10
                                start_frame = int(start_s * self.csv_fps)
                                end_frame = int((end_s + 1) * self.csv_fps)

                                start_frame = max(0, min(start_frame, total_frames_csv))
                                end_frame = max(0, min(end_frame, total_frames_csv))

                                if end_frame > start_frame:
                                    full_res_labels[start_frame:end_frame] = action_id
                            except ValueError:
                                continue

            # Resample labels to match the downsampled features
            label_indices = np.linspace(0, total_frames_csv - 1, target_len).astype(int)
            labels = full_res_labels[label_indices]

            batch_input.append(features)
            batch_target.append(labels)
            final_batch_ids.append(vid)

        if not batch_input:
            return None, None, None, []

        # --- 6. Pad Batch ---
        length_of_sequences = [len(l) for l in batch_target]
        max_len = max(length_of_sequences)
        feat_dim = batch_input[0].shape[0]

        np_batch_input = np.zeros((len(batch_input), feat_dim, max_len), dtype='float32')
        np_batch_target = np.ones((len(batch_target), max_len), dtype='int64') * -100
        mask = np.zeros((len(batch_input), 1, max_len), dtype='float32')

        for i in range(len(batch_input)):
            l = length_of_sequences[i]
            np_batch_input[i, :, :l] = batch_input[i]
            np_batch_target[i, :l] = batch_target[i]
            mask[i, :, :l] = 1

        return torch.tensor(np_batch_input), torch.tensor(np_batch_target), torch.tensor(mask), final_batch_ids

# --- Main Script ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#SEED = 19980125
SEED = 20020827 # my birthday
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
import csv
def main():
    # --- CONFIG ---
    action = 'train' 
    num_epochs = 50
    lr = 0.0005 # 1e-4 is better
    
    # !!! SET THIS TO AVOID OOM !!!
    # 1 fps = 1200 frames for 20 mins (Very safe)
    # 5 fps = 6000 frames for 20 mins (Safe on A100/V100)
    TARGET_FPS = 5 
    FOLD = 3
    print(FOLD)
    
    base_dir = "/scratch/lt200353-pcllm/location/cas_colon/"
    features_path = os.path.join(base_dir, "features_dinov3/") 
    train_split_csv = f"cv_folds_generated/fold{FOLD}_train.csv"
    test_split_csv = f"cv_folds_generated/fold{FOLD}_test.csv"
    #train_split_csv = os.path.join(base_dir, "updated_train_split.csv")
    #test_split_csv = os.path.join(base_dir, "updated_test_split.csv")
    
    save_dir = os.path.join(base_dir, f"dinov3_models_fps{TARGET_FPS}_{FOLD}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define Classes
    class_names = [
        "Terminal_Ileum", "Cecum", "Ascending_Colon", "Hepatic_Flexure", 
        "Transverse_Colon", "Splenic_Flexure", "Descending_Colon", 
        "Sigmoid_Colon", "Rectum", "Anal_Canal"
    ]
    actions_dict = {name: i for i, name in enumerate(class_names)}
    num_classes = len(actions_dict)
    id2name = {i: name for i, name in enumerate(class_names)}
    id2name[-100] = "Unknown/Unlabeled" # Handle padding/ignore index

    # Detect Dim
    try:
        sample_file = os.listdir(features_path)[0]
        sample_feat = torch.load(os.path.join(features_path, sample_file), map_location='cpu')
        input_dim = min(sample_feat.shape)
        print(f"Detected Feature Dim: {input_dim}")
    except IndexError:
        print("Error: No feature files found.")
        return

    trainer = Trainer(
        num_layers=10, 
        r1=2, 
        r2=2, 
        num_f_maps=64, 
        input_dim=input_dim, 
        num_classes=num_classes, 
        channel_masking_rate=0.3
    )
    model = trainer.model
    MODEL_WEIGHTS_PATH = os.path.join(save_dir, "epoch-27.model")
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        print(f"Successfully loaded weights from: {MODEL_WEIGHTS_PATH}")
    else:
        raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS_PATH}")
        
    model.to(DEVICE)
    model.eval()

    # --- 5. EVALUATE & SAVE CSV ---
    batch_gen_test = BatchGenerator(actions_dict, test_split_csv, features_path, target_fps=TARGET_FPS)
    csv_output_dir = save_dir
    print(f"Evaluating and saving results to: {csv_output_dir}")
    batch_gen_test.reset()
    
    all_preds = []
    all_targets = []

    while batch_gen_test.has_next():
        batch_input, batch_target, mask, batch_vid_ids = batch_gen_test.next_batch(1)
        
        if batch_input is None:
            continue
            
        vid_id = batch_vid_ids[0]
        
        batch_input = batch_input.to(DEVICE)
        mask = mask.to(DEVICE)
        
        # Forward Pass
        with torch.no_grad():
            output = model(batch_input, mask)
            prediction = output[-1] # ASFormer final stage output
            
            # shape: (1, num_classes, max_len) -> argmax -> (max_len,)
            pred_ids = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
            
        gt_ids = batch_target[0].numpy()
        
        # Crop out padding using the mask sum
        valid_length = int(mask[0, 0].sum().item())
        pred_ids = pred_ids[:valid_length]
        gt_ids = gt_ids[:valid_length]

        all_preds.extend(pred_ids)
        all_targets.extend(gt_ids)

        # Build Data for CSV
        pred_labels = [id2name[int(pid)] for pid in pred_ids]
        gt_labels = [id2name[int(gid)] for gid in gt_ids]
        
        csv_save_path = os.path.join(csv_output_dir, f"{vid_id}.csv")
        
        # Write to CSV
        with open(csv_save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frame_Idx', 'Time_Seconds', 'Ground_Truth', 'Prediction'])
            
            for frame_idx, (gt_lbl, pred_lbl) in enumerate(zip(gt_labels, pred_labels)):
                time_sec = frame_idx / TARGET_FPS
                writer.writerow([frame_idx, f"{time_sec:.2f}", gt_lbl, pred_lbl])

    # --- 6. PRINT METRICS ---
    print("\n" + "-"*50)
    print("GLOBAL CLASSIFICATION METRICS")
    print("-"*50)
    
    # Clean out the ignore index (-100)
    valid_indices = [i for i, t in enumerate(all_targets) if t != -100]
    clean_targets = [all_targets[i] for i in valid_indices]
    clean_preds = [all_preds[i] for i in valid_indices]
    
    # Only report on classes actually present in the ground truth
    present_target_ids = sorted(list(set(clean_targets)))
    present_target_names = [id2name[i] for i in present_target_ids]
    
    print(classification_report(clean_targets, clean_preds, target_names=present_target_names, digits=4))
    print(f"\nAll individual video results saved in: {csv_output_dir}")

if __name__ == "__main__":
    main()

