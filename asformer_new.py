import torch
import numpy as np
import random
import os
import pandas as pd
from model.ASFormer import Trainer

# =====================================================================
# 1. MS-TCN Metrics & Boundary MAE Functions
# =====================================================================
def get_labels_start_end_time(frame_wise_labels, bg_class=[-100]):
    labels, starts, ends = [], [], []
    if len(frame_wise_labels) == 0:
        return labels, starts, ends
    
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(len(frame_wise_labels))
    return labels, starts, ends

def levenstein(p, y, norm=False):
    m_row, n_col = len(p), len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)
    for i in range(m_row + 1): D[i, 0] = i
    for i in range(n_col + 1): D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    return (1 - D[-1, -1] / max(m_row, n_col)) * 100 if norm else D[-1, -1]

def edit_score(recognized, ground_truth, norm=True, bg_class=[-100]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)

def iou_f1_score(recognized, ground_truth, overlap, bg_class=[-100]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp, fp, hits = 0, 0, np.zeros(len(y_label))
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        
        if len(IoU) > 0:
            idx = np.array(IoU).argmax()
            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else: fp += 1
        else: fp += 1
            
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def compute_boundary_mae(recognized, ground_truth, bg_class=[-100]):
    _, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    _, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    p_b = np.array(list(set(p_start[1:] + p_end[:-1]))) 
    y_b = np.array(list(set(y_start[1:] + y_end[:-1])))

    if len(y_b) == 0: return 0.0 
    if len(p_b) == 0: return float(len(recognized)) 

    mae = 0.0
    for gt_b in y_b: mae += np.min(np.abs(p_b - gt_b))
    return float(mae / len(y_b))

# =====================================================================
# 2. Evaluation Runner
# =====================================================================
@torch.no_grad()
def evaluate_model(model, batch_gen, device, bg_class=[-100]):
    """Runs the validation set through the model and computes metrics."""
    if batch_gen is None: return
    
    model.eval()
    batch_gen.reset()
    
    completed_video_preds = []
    completed_video_labels = []

    while batch_gen.has_next():
        batch_input, batch_target, mask, batch_ids = batch_gen.next_batch(1)
        if batch_input is None:
            continue
            
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        mask = mask.to(device)
        
        # ASFormer forward pass expects (input, mask)
        outputs = model(batch_input, mask)
        
        # ASFormer usually returns a list of outputs for each stage. Extract the final stage.
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            final_output = outputs[-1] 
        else:
            final_output = outputs
            
        preds = torch.argmax(final_output, dim=1) # [B, Max_Len]
        
        for b in range(preds.size(0)):
            p_flat = preds[b].cpu().numpy()
            l_flat = batch_target[b].cpu().numpy()
            mask_flat = mask[b, 0].cpu().numpy()
            
            # Trim padding using the mask
            valid_length = int(np.sum(mask_flat))
            valid_preds = p_flat[:valid_length].tolist()
            valid_labels = l_flat[:valid_length].tolist()
            
            completed_video_preds.append(valid_preds)
            completed_video_labels.append(valid_labels)

    # Compute Metrics over all completed videos
    overlap = [0.1, 0.25, 0.5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    edit_total, mae_total, valid_video_count = 0.0, 0.0, 0
    
    for p_seq, l_seq in zip(completed_video_preds, completed_video_labels):
        if len(p_seq) == 0: continue
        valid_video_count += 1
        edit_total += edit_score(p_seq, l_seq, bg_class=bg_class)
        mae_total += compute_boundary_mae(p_seq, l_seq, bg_class=bg_class)
        
        for s in range(len(overlap)):
            tp1, fp1, fn1 = iou_f1_score(p_seq, l_seq, overlap[s], bg_class=bg_class)
            tp[s] += tp1; fp[s] += fp1; fn[s] += fn1

    avg_edit = edit_total / valid_video_count if valid_video_count > 0 else 0.0
    avg_mae = mae_total / valid_video_count if valid_video_count > 0 else 0.0
    
    f1s = []
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0.0
        recall = tp[s] / float(tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(np.nan_to_num(f1) * 100)

    print("\n=== Validation Metrics ===")
    print(f"Edit Score:  {avg_edit:.4f}")
    print(f"Bndry MAE:   {avg_mae:.4f} frames")
    print(f"F1@10:       {f1s[0]:.2f}")
    print(f"F1@25:       {f1s[1]:.2f}")
    print(f"F1@50:       {f1s[2]:.2f}")
    print("==========================\n")
    
    return avg_edit, f1s, avg_mae

# =====================================================================
# 3. Utilities & Data Generator
# =====================================================================
def time_to_seconds(t_str):
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

        self.csv_fps = 60.0
        self.target_fps = target_fps

        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]

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
        final_batch_ids = []

        for vid in batch_ids:
            if str(vid) not in self.video_info:
                continue
            row = self.video_info[str(vid)]
            total_frames_csv = int(row['TotalFrames'])

            feat_path = os.path.join(self.features_path, f"{vid}.pt")
            try:
                features = torch.load(feat_path, map_location='cpu')
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
            except FileNotFoundError:
                print(f"⚠️ Feature file not found: {feat_path}")
                continue

            shape = features.shape
            if len(shape) == 2:
                diff_dim0 = abs(shape[0] - total_frames_csv)
                diff_dim1 = abs(shape[1] - total_frames_csv)

                if diff_dim0 < diff_dim1:
                    features = features.T

            duration_sec = total_frames_csv / self.csv_fps
            target_len = int(duration_sec * self.target_fps)
            if target_len < 1: target_len = 1

            curr_feat_len = features.shape[1]
            feat_indices = np.linspace(0, curr_feat_len - 1, target_len).astype(int)
            features = features[:, feat_indices]

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

                                start_frame = int(start_s * self.csv_fps)
                                end_frame = int((end_s + 1) * self.csv_fps)

                                start_frame = max(0, min(start_frame, total_frames_csv))
                                end_frame = max(0, min(end_frame, total_frames_csv))

                                if end_frame > start_frame:
                                    full_res_labels[start_frame:end_frame] = action_id
                            except ValueError:
                                continue

            label_indices = np.linspace(0, total_frames_csv - 1, target_len).astype(int)
            labels = full_res_labels[label_indices]

            batch_input.append(features)
            batch_target.append(labels)
            final_batch_ids.append(vid)

        if not batch_input:
            return None, None, None, []

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

# =====================================================================
# 4. Main Script
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20020827
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def main():
    # --- CONFIG ---
    action = 'train' 
    num_epochs = 50
    lr = 0.0005 
    
    TARGET_FPS = 5 
    FOLD = 1
    print(f"Executing Fold: {FOLD}")
    
    base_dir = "/project/lt200353-pcllm/3d_report_gen/cas_colon"
    features_path = os.path.join(base_dir, "features_dinov3/") 
    train_split_csv = f"cv_folds_generated/fold{FOLD}_train.csv"
    test_split_csv = f"cv_folds_generated/fold{FOLD}_test.csv"
    
    save_dir = os.path.join(base_dir, f"dinov3_models_fps{TARGET_FPS}_{FOLD}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    class_names = [
        "Terminal_Ileum", "Cecum", "Ascending_Colon", "Hepatic_Flexure", 
        "Transverse_Colon", "Splenic_Flexure", "Descending_Colon", 
        "Sigmoid_Colon", "Rectum", "Anal_Canal"
    ]
    actions_dict = {name: i for i, name in enumerate(class_names)}
    num_classes = len(actions_dict)

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
        batch_gen_train = BatchGenerator(actions_dict, train_split_csv, features_path, target_fps=TARGET_FPS)
    else:
        raise FileNotFoundError("Train CSV not found")

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
        
        # --- NEW: Run custom metric evaluation at the end of training ---
        if batch_gen_test is not None:
            print("\n--- Running Custom Metrics Evaluation on Final Model ---")
            evaluate_model(trainer.model, batch_gen_test, DEVICE)
        
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

        # Evaluate the loaded prediction model 
        if batch_gen_test is not None:
            print("\n--- Running Custom Metrics Evaluation on Loaded Model ---")
            evaluate_model(trainer.model, batch_gen_test, DEVICE)

if __name__ == "__main__":
    main()
