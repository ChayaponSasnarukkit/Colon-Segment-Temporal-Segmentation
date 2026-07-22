import torch
import numpy as np
import random
import os
from model.ASFormer import Trainer

# =====================================================================
# 1. Real-Colon Label Mapping & Constants
# =====================================================================
LABEL_MAP = {
    "outside": 0,
    "insertion": 1,
    "ceacum": 2,
    "ileum": 3,
    "ascending": 4,
    "transverse": 5,
    "descending": 6,
    "sigmoid": 7,
    "rectum": 8,
}
# We keep 999 as the raw ignore index to match real-colon conventions,
# but we will safely map it to -100 in the generator so PyTorch's 
# default CrossEntropyLoss (used in ASFormer) doesn't crash.
RAW_IGNORE_INDEX = 999 
TORCH_IGNORE_INDEX = -100

# =====================================================================
# 2. MS-TCN/ASFormer Metrics & Boundary MAE Functions
# =====================================================================
def get_labels_start_end_time(frame_wise_labels, bg_class=[TORCH_IGNORE_INDEX]):
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

def edit_score(recognized, ground_truth, norm=True, bg_class=[TORCH_IGNORE_INDEX]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)

def iou_f1_score(recognized, ground_truth, overlap, bg_class=[TORCH_IGNORE_INDEX]):
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

def compute_boundary_mae(recognized, ground_truth, bg_class=[TORCH_IGNORE_INDEX]):
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
# 3. Evaluation Runner
# =====================================================================
@torch.no_grad()
def evaluate_model(model, batch_gen, device, bg_class=[TORCH_IGNORE_INDEX]):
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
        
        outputs = model(batch_input, mask)
        
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            final_output = outputs[-1] 
        else:
            final_output = outputs
            
        preds = torch.argmax(final_output, dim=1) # [B, Max_Len]
        
        for b in range(preds.size(0)):
            p_flat = preds[b].cpu().numpy()
            l_flat = batch_target[b].cpu().numpy()
            mask_flat = mask[b, 0].cpu().numpy()
            
            # Extract valid sequences using boolean indexing 
            # (Essential because real-colon validation uses centered padding)
            valid_indices = np.where(mask_flat == 1.0)[0]
            if len(valid_indices) == 0: continue
                
            valid_preds = p_flat[valid_indices].tolist()
            valid_labels = l_flat[valid_indices].tolist()
            
            completed_video_preds.append(valid_preds)
            completed_video_labels.append(valid_labels)

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
# 4. Real-Colon Batch Generator
# =====================================================================
"""class RealColonBatchGenerator(object):
    
    Replaces the CSV BatchGenerator. Directly reads real-colon .pt and _labels.npy files.
    Formats batches to [B, C, T] to match ASFormer requirements.
    
    def __init__(self, video_root, split_dir=None, phase='train', fold=1, 
                 temporal_augmentation=False, subsampling_factor=1):
        self.video_root = video_root
        self.phase = phase
        self.temporal_augmentation = temporal_augmentation
        self.subsampling_factor = subsampling_factor
        self.video_names = []
        
        # 1. Discover sessions based on split TXT files
        sessions = []
        if split_dir:
            if self.phase == 'train':
                split_files = [f'fold{fold}_train.txt', f'fold{fold}_valid.txt']
            else:
                split_files = [f'fold{fold}_test.txt']

            for file_name in split_files:
                file_path = os.path.join(split_dir, file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        sessions.extend(lines)
                else:
                    print(f"Warning: Split file not found: {file_path}")

        # 2. Validate paths
        pt_files = sorted([f for f in os.listdir(self.video_root) if f.endswith('.pt')])
        for pt_file in pt_files:
            vid_id = pt_file.replace('.pt', '')
            if sessions and vid_id not in sessions:
                continue
            lbl_path = os.path.join(self.video_root, f"{vid_id}_labels.npy")
            if os.path.exists(lbl_path):
                self.video_names.append(vid_id)

        print(f"Loaded {len(self.video_names)} videos for phase: {self.phase}")
        self.index = 0
        self.reset()

    def reset(self):
        self.index = 0
        if self.phase == 'train':
            random.shuffle(self.video_names)

    def has_next(self):
        return self.index < len(self.video_names)

    def next_batch(self, batch_size):
        batch_ids = self.video_names[self.index : self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []

        for vid in batch_ids:
            emb_path = os.path.join(self.video_root, f"{vid}.pt")
            lbl_path = os.path.join(self.video_root, f"{vid}_labels.npy")

            # Load Tensors (Time, Features)
            emb = torch.load(emb_path, map_location='cpu').numpy()
            raw_labels = np.load(lbl_path)

            gts = []
            for lbl in raw_labels:
                lbl_str = lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl)
                lbl_str = lbl_str.strip()
                gts.append(LABEL_MAP.get(lbl_str, RAW_IGNORE_INDEX)) 
            gts = np.array(gts)

            # Match lengths
            min_len = min(emb.shape[0], gts.shape[0])
            emb = emb[:min_len]
            gts = gts[:min_len]

            # Subsampling & Temporal Augmentation
            if self.subsampling_factor > 1:
                if self.temporal_augmentation and self.phase == 'train':
                    if random.uniform(0, 1) < 0.4:
                        frame_indices = [f for f in range(emb.shape[0]) if random.uniform(0, 1) <= (1 / self.subsampling_factor)]
                    else:
                        frame_indices = list(range(random.randint(0, self.subsampling_factor - 1), emb.shape[0], self.subsampling_factor))
                else:
                    frame_indices = list(range(0, emb.shape[0], self.subsampling_factor))

                emb = emb[np.array(frame_indices), :]
                gts = gts[np.array(frame_indices)]

            batch_input.append(emb)
            batch_target.append(gts)

        if not batch_input:
            return None, None, None, []

        # 3. Dynamic Padding
        max_len = max([b.shape[0] for b in batch_input])
        feat_dim = batch_input[0].shape[1]

        # Initialize padded tensors matching ASFormer shapes
        np_batch_input = np.zeros((len(batch_input), feat_dim, max_len), dtype='float32')
        np_batch_target = np.full((len(batch_target), max_len), TORCH_IGNORE_INDEX, dtype='int64')
        mask = np.zeros((len(batch_input), 1, max_len), dtype='float32')

        for i in range(len(batch_input)):
            curr_len = batch_input[i].shape[0]
            
            # Replicating real-colon padding logic
            if self.phase == 'train':
                start = 0 # Left-aligned for training
            else:
                start = (max_len - curr_len) // 2 # Centered for validation
                
            end = start + curr_len
            
            # ASFormer expects [Batch, Channel, Time], so transpose emb
            np_batch_input[i, :, start:end] = batch_input[i].T
            
            # Remap 999 to -100 to prevent PyTorch out-of-bounds CE Loss errors
            remapped_target = np.where(batch_target[i] == RAW_IGNORE_INDEX, TORCH_IGNORE_INDEX, batch_target[i])
            np_batch_target[i, start:end] = remapped_target
            
            mask[i, :, start:end] = 1.0

        return torch.tensor(np_batch_input), torch.tensor(np_batch_target), torch.tensor(mask), batch_ids
"""
# =====================================================================
# 4. Real-Colon Batch Generator
# =====================================================================
class RealColonBatchGenerator(object):
    """
    Replaces the CSV BatchGenerator. Directly reads real-colon .pt and _labels.npy files.
    Formats batches to [B, C, T] to match ASFormer requirements.
    """
    def __init__(self, video_root, split_dir=None, phase='train', fold=1,
                 temporal_augmentation=False, subsampling_factor=1):
        self.video_root = video_root
        self.phase = phase
        self.temporal_augmentation = temporal_augmentation
        self.subsampling_factor = subsampling_factor

        # ASFormer Trainer explicitly looks for `list_of_examples` to calculate batches
        self.list_of_examples = []

        # 1. Discover sessions based on split TXT files
        sessions = []
        if split_dir:
            if self.phase == 'train':
                split_files = [f'fold{fold}_train.txt', f'fold{fold}_valid.txt']
            else:
                split_files = [f'fold{fold}_test.txt']

            for file_name in split_files:
                file_path = os.path.join(split_dir, file_name)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        sessions.extend(lines)
                else:
                    print(f"Warning: Split file not found: {file_path}")

        # 2. Validate paths
        pt_files = sorted([f for f in os.listdir(self.video_root) if f.endswith('.pt')])
        for pt_file in pt_files:
            vid_id = pt_file.replace('.pt', '')
            if sessions and vid_id not in sessions:
                continue
            lbl_path = os.path.join(self.video_root, f"{vid_id}_labels.npy")
            if os.path.exists(lbl_path):
                self.list_of_examples.append(vid_id)

        print(f"Loaded {len(self.list_of_examples)} videos for phase: {self.phase}")
        self.index = 0
        self.reset()

    def reset(self):
        self.index = 0
        if self.phase == 'train':
            random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def next_batch(self, batch_size, *args):
        batch_ids = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []

        for vid in batch_ids:
            emb_path = os.path.join(self.video_root, f"{vid}.pt")
            lbl_path = os.path.join(self.video_root, f"{vid}_labels.npy")

            # Load Tensors (Time, Features)
            emb = torch.load(emb_path, map_location='cpu').numpy()
            raw_labels = np.load(lbl_path)

            gts = []
            for lbl in raw_labels:
                lbl_str = lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl)
                lbl_str = lbl_str.strip()
                gts.append(LABEL_MAP.get(lbl_str, RAW_IGNORE_INDEX))
            gts = np.array(gts)

            # Match lengths
            min_len = min(emb.shape[0], gts.shape[0])
            emb = emb[:min_len]
            gts = gts[:min_len]

            # Subsampling & Temporal Augmentation
            if self.subsampling_factor > 1:
                if self.temporal_augmentation and self.phase == 'train':
                    if random.uniform(0, 1) < 0.4:
                        frame_indices = [f for f in range(emb.shape[0]) if random.uniform(0, 1) <= (1 / self.subsampling_factor)]
                    else:
                        frame_indices = list(range(random.randint(0, self.subsampling_factor - 1), emb.shape[0], self.subsampling_factor))
                else:
                    frame_indices = list(range(0, emb.shape[0], self.subsampling_factor))

                emb = emb[np.array(frame_indices), :]
                gts = gts[np.array(frame_indices)]

            batch_input.append(emb)
            batch_target.append(gts)

        if not batch_input:
            return None, None, None, []

        # 3. Dynamic Padding
        max_len = max([b.shape[0] for b in batch_input])
        feat_dim = batch_input[0].shape[1]

        # Initialize padded tensors matching ASFormer shapes
        np_batch_input = np.zeros((len(batch_input), feat_dim, max_len), dtype='float32')
        np_batch_target = np.full((len(batch_target), max_len), TORCH_IGNORE_INDEX, dtype='int64')
        mask = np.zeros((len(batch_input), 1, max_len), dtype='float32')

        for i in range(len(batch_input)):
            curr_len = batch_input[i].shape[0]

            # Replicating real-colon padding logic
            if self.phase == 'train':
                start = 0 # Left-aligned for training
            else:
                start = (max_len - curr_len) // 2 # Centered for validation

            end = start + curr_len

            # ASFormer expects [Batch, Channel, Time], so transpose emb
            np_batch_input[i, :, start:end] = batch_input[i].T

            # Remap 999 to -100 to prevent PyTorch out-of-bounds CE Loss errors
            remapped_target = np.where(batch_target[i] == RAW_IGNORE_INDEX, TORCH_IGNORE_INDEX, batch_target[i])
            np_batch_target[i, start:end] = remapped_target

            mask[i, :, start:end] = 1.0

        return torch.tensor(np_batch_input), torch.tensor(np_batch_target), torch.tensor(mask), batch_ids
# =====================================================================
# 5. Main Script
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
    
    SUBSAMPLING_FACTOR = 1
    FOLD = 1
    print(f"Executing Fold: {FOLD}")
    
    # Updated Path configuration for real-colon
    VIDEO_ROOT = "/project/lt200353-pcllm/3d_report_gen/real-colon/"
    SPLIT_DIR = "/home/csasnaru/temporal_segmentation/data/dataset/RC_lists/5_fold/"
    
    save_dir = os.path.join(VIDEO_ROOT, f"asformer_models_{FOLD}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_classes = len(LABEL_MAP)

    # Detect Input Dimension
    try:
        sample_file = [f for f in os.listdir(VIDEO_ROOT) if f.endswith('.pt')][0]
        sample_feat = torch.load(os.path.join(VIDEO_ROOT, sample_file), map_location='cpu')
        input_dim = sample_feat.shape[1] # .pt is [Time, Features]
        print(f"Detected Feature Dim: {input_dim}")
    except IndexError:
        print("Error: No feature files found in directory.")
        return

    # Initialize ASFormer Trainer
    trainer = Trainer(
        num_layers=10, 
        r1=2, 
        r2=2, 
        num_f_maps=64, 
        input_dim=input_dim, 
        num_classes=num_classes, 
        channel_masking_rate=0.3
    )

    # Initialize Real-Colon Batch Generators
    batch_gen_train = RealColonBatchGenerator(
        video_root=VIDEO_ROOT, 
        split_dir=SPLIT_DIR, 
        phase='train', 
        fold=FOLD, 
        temporal_augmentation=True, 
        subsampling_factor=SUBSAMPLING_FACTOR
    )
    
    batch_gen_test = RealColonBatchGenerator(
        video_root=VIDEO_ROOT, 
        split_dir=SPLIT_DIR, 
        phase='test', 
        fold=FOLD, 
        temporal_augmentation=False, 
        subsampling_factor=SUBSAMPLING_FACTOR
    )

    if action == 'train':
        print(f"Starting ASFormer training...")
        trainer.train(
            save_dir=save_dir,
            batch_gen=batch_gen_train,
            num_epochs=num_epochs,
            batch_size=6, 
            learning_rate=lr,
            batch_gen_tst=batch_gen_test
        )
        
        if batch_gen_test is not None:
            print("\n--- Running Custom Metrics Evaluation on Final Model ---")
            evaluate_model(trainer.model, batch_gen_test, DEVICE)
            
    elif action == 'predict':
        trainer.predict(
            model_dir=save_dir,
            results_dir=os.path.join(save_dir, "results"),
            features_path=VIDEO_ROOT,
            batch_gen=batch_gen_test,
            epoch=num_epochs, 
            actions_dict=LABEL_MAP,
            sample_rate=1 
        )

        if batch_gen_test is not None:
            print("\n--- Running Custom Metrics Evaluation on Loaded Model ---")
            evaluate_model(trainer.model, batch_gen_test, DEVICE)

if __name__ == "__main__":
    main()
