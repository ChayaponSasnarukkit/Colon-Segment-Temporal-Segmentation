import torch
import numpy as np
import random
import os
import pandas as pd
from model.ASFormer import Trainer

# NOTE: The `time_to_seconds` helper function has been completely removed 
# because your new dataset already provides dense frame-by-frame labels!

class BatchGenerator(object):
    def __init__(self, actions_dict, label_files, labels_dir, metadata_csv, features_path, target_fps=1):
        self.label_files = sorted(label_files)
        self.actions_dict = actions_dict
        self.features_path = features_path
        self.labels_dir = labels_dir
        self.target_fps = target_fps

        # --- 1. Load Metadata (for source FPS) ---
        meta_df = pd.read_csv(metadata_csv)
        # Assuming the metadata has a column 'unique_video_name' and 'fps'
        self.fps_dict = meta_df.set_index('unique_video_name')['fps'].to_dict()

        self.video_ids = [os.path.basename(f).replace(".csv", "") for f in self.label_files]

        self.index = 0
        self.reset()

    def reset(self):
        self.index = 0
        random.shuffle(self.video_ids)

    def has_next(self):
        return self.index < len(self.video_ids)

    def next_batch(self, batch_size):
        batch_ids = self.video_ids[self.index : self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        final_batch_ids = [] # To keep track of valid IDs only

        for vid in batch_ids:
            # --- 1. Load New Dense Labels ---
            label_path = os.path.join(self.labels_dir, f"{vid}.csv")
            try:
                df_labels = pd.read_csv(label_path)
                # Map text labels (e.g., 'inside', 'outside') to integers using actions_dict
                # Unmapped/Missing values become NaN, which we fill with -100 (ignore_index)
                full_res_labels = df_labels['GT'].map(self.actions_dict).fillna(-100).values
                total_frames_csv = len(full_res_labels)
            except FileNotFoundError:
                print(f"⚠️ Label file not found: {label_path}")
                continue

            # Retrieve source FPS for this specific video
            src_fps = float(self.fps_dict.get(vid, 25.0))

            # --- 2. Load Features ---
            feat_path = os.path.join(self.features_path, f"{vid}.pt")
            try:
                features = torch.load(feat_path, map_location='cpu')
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
            except FileNotFoundError:
                print(f"⚠️ Feature file not found: {feat_path}")
                continue

            # --- 3. Robust Shape Handling ---
            # We need shape (Dim, Time) for ASFormer (1D Convs).
            shape = features.shape
            if len(shape) == 2:
                diff_dim0 = abs(shape[0] - total_frames_csv)
                diff_dim1 = abs(shape[1] - total_frames_csv)
                
                # If dim0 matches frame count better, it's (Time, Dim). Transpose to (Dim, Time).
                if diff_dim0 < diff_dim1:
                    features = features.T

            # --- 4. Calculate Target Length ---
            duration_sec = total_frames_csv / src_fps
            target_len = int(duration_sec * self.target_fps)
            if target_len < 1: target_len = 1

            # --- 5. Resample Features & Labels ---
            # Downsample/Upsample features across the Time dimension
            curr_feat_len = features.shape[1]
            feat_indices = np.linspace(0, curr_feat_len - 1, target_len).astype(int)
            features = features[:, feat_indices]

            # Downsample/Upsample labels directly (they are already an array of integers now)
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
SEED = 20020827 # your birthday
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
import json
def main():
    # --- CONFIG ---
    action = 'train' 
    num_epochs = 50
    lr = 0.0005 
    TARGET_FPS = 5 
    FOLD = 0
    print(f"Fold: {FOLD}")
    
    base_dir = "/scratch/lt200353-pcllm/location/real_colon/"
    features_path = os.path.join(base_dir, "features_dinov3/") 
    
    # NEW PATHS REQUIRED FOR THE NEW FORMAT:
    labels_dir = os.path.join(base_dir, "labels_cleaned/")     # Directory containing videoID.csv
    metadata_csv = os.path.join(base_dir, "video_info.csv")    # CSV containing unique_video_name and fps
    
    #train_split_csv = os.path.join(base_dir, f"cv_folds_generated/fold{FOLD}_train.csv")
    #test_split_csv = os.path.join(base_dir, f"cv_folds_generated/fold{FOLD}_test.csv")
    with open(f"/scratch/lt200353-pcllm/location/real_colon/splits/fold_{FOLD}.json") as f:
        splits = json.load(f)

    save_dir = os.path.join(base_dir, f"dinov3_models_fps{TARGET_FPS}_{FOLD}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- Define Classes ---
    class_names = [
        'outside', 'insertion', 'ceacum', 'ileum', 'ascending',
        'transverse', 'descending', 'sigmoid', 'rectum'
    ]
    actions_dict = {name: i for i, name in enumerate(class_names)}
    num_classes = len(actions_dict)

    # Detect Feature Dimension
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

    # --- Instantiate Batch Generators with new arguments ---
    #if os.path.exists(train_split_csv):
    batch_gen_train = BatchGenerator(
        actions_dict=actions_dict, 
        label_files=splits["train"], 
        labels_dir=labels_dir, 
        metadata_csv=metadata_csv, 
        features_path=features_path, 
        target_fps=TARGET_FPS
    )
    #else:
        #raise FileNotFoundError(f"Train CSV not found: {train_split_csv}")

    #if os.path.exists(test_split_csv):
    batch_gen_test = BatchGenerator(
        actions_dict=actions_dict, 
        label_files=splits["val"], 
        labels_dir=labels_dir, 
        metadata_csv=metadata_csv, 
        features_path=features_path, 
        target_fps=TARGET_FPS
    )
    #else:
        #batch_gen_test = None

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
