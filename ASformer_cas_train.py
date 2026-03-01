import torch
import numpy as np
import random
import os
import pandas as pd
from model.ASFormer import Trainer

# --- Helper to parse time strings like "1:23" to seconds ---
"""

def time_to_seconds(t_str):
    try:
        parts = t_str.strip().split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        return 0
    return 0

class BatchGenerator(object):
    def __init__(self, actions_dict, csv_path, features_path, target_fps=1):
        self.list_of_examples = list()
        self.index = 0
        self.actions_dict = actions_dict
        self.features_path = features_path
        self.csv_path = csv_path
        
        # Original video metadata is 60fps based on your CSV description
        self.csv_fps = 60.0 
        
        # The new FPS you want to train on (e.g., 1 fps or 5 fps)
        self.target_fps = target_fps
        
        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]
        self.list_of_examples = self.df['VideoID'].astype(str).tolist()
        self.reset()

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def next_batch(self, batch_size, *args):
        batch_ids = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        
        for vid in batch_ids:
            # --- 1. Load Features ---
            feat_path = os.path.join(self.features_path, f"{vid}.pt")
            try:
                features = torch.load(feat_path, map_location='cpu')
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
            except FileNotFoundError:
                print(f"⚠️ Feature file not found: {feat_path}")
                continue

            # Ensure shape is (Dim, Time)
            if features.shape[0] > features.shape[1]: 
                features = features.T
            
            # --- 2. Calculate New Target Length ---
            row = self.df[self.df['VideoID'].astype(str) == str(vid)].iloc[0]
            total_frames_csv = int(row['TotalFrames']) # Original length at 60fps
            
            # Calculate duration in seconds
            duration_sec = total_frames_csv / self.csv_fps
            
            # Calculate the new desired length at target_fps
            target_len = int(duration_sec * self.target_fps)
            
            # Safety check: avoid empty sequences
            if target_len < 1: target_len = 1

            # --- 3. Resample Features ---
            # Create indices to downsample features to target_len
            curr_feat_len = features.shape[1]
            feat_indices = np.linspace(0, curr_feat_len - 1, target_len).astype(int)
            
            # Slice the features (Dim, Time) -> (Dim, New_Time)
            features = features[:, feat_indices]

            # --- 4. Generate & Resample Labels ---
            # First, create the dense label array at full 60fps resolution
            full_res_labels = np.full(total_frames_csv, -100, dtype=int) # -100 is ignore index
            
            for action_name, action_id in self.actions_dict.items():
                if action_name in row and not pd.isna(row[action_name]):
                    time_entry = str(row[action_name])
                    ranges = time_entry.split('/')
                    for rng in ranges:
                        rng = rng.strip()
                        if '-' in rng:
                            start_str, end_str = rng.split('-')
                            # Convert time to frame indices (at 60fps)
                            start_frame = int(time_to_seconds(start_str) * self.csv_fps)
                            end_frame = int(time_to_seconds(end_str) * self.csv_fps)
                            
                            start_frame = max(0, min(start_frame, total_frames_csv))
                            end_frame = max(0, min(end_frame, total_frames_csv))
                            
                            full_res_labels[start_frame:end_frame] = action_id
            
            # Now resample labels to match the features we just downsampled
            # We map 0..total_frames_csv -> 0..target_len
            label_indices = np.linspace(0, total_frames_csv - 1, target_len).astype(int)
            labels = full_res_labels[label_indices]

            batch_input.append(features)
            batch_target.append(labels)

        if not batch_input:
            return None, None, None, []

        # --- 5. Pad Batch ---
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

        return torch.tensor(np_batch_input), torch.tensor(np_batch_target), torch.tensor(mask), batch_ids
"""


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

    if os.path.exists(train_split_csv):
        # Pass TARGET_FPS here
        batch_gen_train = BatchGenerator(actions_dict, train_split_csv, features_path, target_fps=TARGET_FPS)
    else:
        raise FileNotFoundError(f"Train CSV not found")

    if os.path.exists(test_split_csv):
        batch_gen_test = BatchGenerator(actions_dict, test_split_csv, features_path, target_fps=TARGET_FPS)
    else:
        batch_gen_test = None

    if action == 'train':
        print(f"Starting training at {TARGET_FPS} FPS...")
        trainer.train(
            save_dir=save_dir,
            batch_gen=batch_gen_train,
            num_epochs=num_epochs,
            batch_size=1, 
            learning_rate=lr,
            batch_gen_tst=batch_gen_test
        )
        
    elif action == 'predict':
        trainer.predict(
            model_dir=save_dir,
            results_dir=os.path.join(save_dir, "results"),
            features_path=features_path,
            batch_gen=batch_gen_test,
            epoch=num_epochs, 
            actions_dict=actions_dict,
            sample_rate=1 
        )

if __name__ == "__main__":
    main()
