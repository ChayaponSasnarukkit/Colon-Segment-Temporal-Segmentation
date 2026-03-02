import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import collections
from tqdm import tqdm
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# --- Project Imports ---
from model.CMamba import MambaTemporalSegmentation, detach_states, apply_reset_mask
from dataset.cas_locationv3 import MedicalStreamingDataset
from model.ContextMamba import ContextMambav2
from mstcn_style_metric import edit_score, iou_f1_score 

CLASS_MAP = {
    'Terminal_Ileum': 0, 'Cecum': 1, 'Ascending_Colon': 2, 'Hepatic_Flexure': 3,
    'Transverse_Colon': 4, 'Splenic_Flexure': 5, 'Descending_Colon': 6,
    'Sigmoid_Colon': 7, 'Rectum': 8, 'Anal_Canal': 9
}

IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

@dataclass
class MambaTemporalConfig:
    d_model: int = 1024           
    n_layer: int = 8             
    d_intermediate: int = 0      
    ssm_cfg: dict = field(default_factory=lambda: {
        "d_state": 16,           
        "d_conv": 4,             
        "expand": 2,             
        "dt_rank": "auto",       
        "layer": "Mamba1",       
        "use_fast_path": False,
    }) 
    rms_norm: bool = True        
    norm_epsilon: float = 1e-5
    fused_add_norm: bool = True  
    residual_in_fp32: bool = True 

# --- 1. The Paper's Post-Processing Algorithm ---
def confidence_length_filter(predictions, probabilities, theta=0.8, l_min=30):
    """
    Implements the confidence-based post-processing algorithm.
    theta: Confidence threshold (e.g., 0.8 means 80% confident).
    l_min: Minimum length threshold (e.g., 30 frames = 1 second at 30fps).
    """
    if len(predictions) == 0: 
        return []

    smoothed = []
    current_class = predictions[0]
    current_len = 1
    smoothed.append(current_class)

    for t in range(1, len(predictions)):
        raw_pred = predictions[t]
        q_t = probabilities[t] 

        if raw_pred != current_class:
            is_unreliable = q_t < theta
            is_too_short = current_len < l_min

            if is_unreliable and is_too_short:
                final_pred = current_class # Disregard, retain previous
            else:
                final_pred = raw_pred      # Accept change
        else:
            final_pred = raw_pred

        smoothed.append(final_pred)

        if final_pred == current_class:
            current_len += 1
        else:
            current_class = final_pred
            current_len = 1

    return np.array(smoothed)

# --- 2. Metric Calculation Helper ---
def calculate_all_metrics(all_gt_flat, all_pred_flat, video_gts, video_preds, bg_class=[0]):
    acc = accuracy_score(all_gt_flat, all_pred_flat) if len(all_gt_flat) > 0 else 0.0
    macro_f1 = f1_score(all_gt_flat, all_pred_flat, average='macro', zero_division=0) if len(all_gt_flat) > 0 else 0.0
    
    overlap = [0.1, 0.25, 0.5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    edit_total = 0.0
    
    for p_seq, l_seq in zip(video_preds, video_gts):
        if len(p_seq) == 0: continue
        edit_total += edit_score(p_seq, l_seq, bg_class=bg_class)
        for s in range(len(overlap)):
            tp1, fp1, fn1 = iou_f1_score(p_seq, l_seq, overlap[s], bg_class=bg_class)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    num_videos = len(video_preds)
    avg_edit = edit_total / num_videos if num_videos > 0 else 0.0
    
    f1s = []
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0.0
        recall = tp[s] / float(tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1 * 100)
        
    return acc, macro_f1, avg_edit, f1s

# --- 3. Main Evaluation Loop ---
@torch.no_grad()
def evaluate_and_export(model, dataloader, device, csv_export_path, theta=0.8, l_min=30, bg_class=[0]): 
    model.eval()
    worker_states = {}
    
    video_preds_dict = collections.defaultdict(list)
    video_confs_dict = collections.defaultdict(list) # NEW: Store probabilities
    video_labels_dict = collections.defaultdict(list)
    
    completed_video_preds = []
    completed_video_confs = []
    completed_video_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="Extracting & Filtering")):
        vision_embeddings, contexts, labels, future_labels, reset_mask, context_masks, worker_id = batch
        
        vision_embeddings = vision_embeddings.to(device)
        contexts = contexts.to(device)
        labels = labels.to(device)
        reset_mask = reset_mask.to(device)
        context_masks = context_masks.to(device)
        
        actual_K = context_masks[0].sum().int().item()
        valid_contexts = contexts[:, :actual_K, :]
        
        w_id = int(worker_id[0].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
        current_states = worker_states.get(w_id, None)
        
        if current_states is not None:
            current_states = apply_reset_mask(current_states, reset_mask)

        _, _, logits_w_future, next_states = model(
            vision_embeddings=vision_embeddings, 
            contexts=valid_contexts,
            pass_states=current_states,
            labels=labels 
        )
        worker_states[w_id] = detach_states(next_states)
        
        # --- NEW: Extract Probabilities ---
        probs = F.softmax(logits_w_future, dim=-1) 
        max_probs, preds = torch.max(probs, dim=-1) 
        
        for b in range(preds.size(0)):
            b_w_id = int(worker_id[b].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
            b_reset = bool(reset_mask[b].item()) if isinstance(reset_mask, torch.Tensor) else bool(reset_mask)
            
            if b_reset and len(video_preds_dict[b_w_id]) > 0:
                completed_video_preds.append(video_preds_dict[b_w_id])
                completed_video_confs.append(video_confs_dict[b_w_id])
                completed_video_labels.append(video_labels_dict[b_w_id])
                video_preds_dict[b_w_id] = []
                video_confs_dict[b_w_id] = []
                video_labels_dict[b_w_id] = []

            p_flat = preds[b].cpu().numpy()
            c_flat = max_probs[b].cpu().numpy()
            l_flat = labels[b].cpu().numpy()
            
            valid_indices = l_flat != -100
            valid_preds = p_flat[valid_indices].tolist()
            valid_confs = c_flat[valid_indices].tolist()
            valid_labels = l_flat[valid_indices].tolist()

            video_preds_dict[b_w_id].extend(valid_preds)
            video_confs_dict[b_w_id].extend(valid_confs)
            video_labels_dict[b_w_id].extend(valid_labels)

    # Flush remaining videos
    for w_id, p_seq in video_preds_dict.items():
        if len(p_seq) > 0:
            completed_video_preds.append(p_seq)
            completed_video_confs.append(video_confs_dict[w_id])
            completed_video_labels.append(video_labels_dict[w_id])

    # --- Apply the Filter and Compute Metrics ---
    print("\nApplying Confidence Filter and Calculating Metrics...")
    
    raw_flat_preds, smooth_flat_preds, flat_labels = [], [], []
    smoothed_video_preds = []
    
    for p_seq, c_seq, l_seq in zip(completed_video_preds, completed_video_confs, completed_video_labels):
        # Apply paper's algorithm
        s_seq = confidence_length_filter(p_seq, c_seq, theta=theta, l_min=l_min)
        smoothed_video_preds.append(s_seq.tolist())
        
        raw_flat_preds.extend(p_seq)
        smooth_flat_preds.extend(s_seq.tolist())
        flat_labels.extend(l_seq)

    raw_acc, raw_macro, raw_edit, raw_f1s = calculate_all_metrics(
        flat_labels, raw_flat_preds, completed_video_labels, completed_video_preds, bg_class
    )
    smooth_acc, smooth_macro, smooth_edit, smooth_f1s = calculate_all_metrics(
        flat_labels, smooth_flat_preds, completed_video_labels, smoothed_video_preds, bg_class
    )

    # --- Print Comparison ---
    print("\n=======================================================")
    print("                POST-PROCESSING RESULTS                ")
    print(f"            (Threshold: {theta}, Min Length: {l_min})")
    print("=======================================================")
    print(f"{'Metric':<20} | {'RAW PREDICTIONS':<15} | {'SMOOTHED':<15}")
    print("-" * 55)
    print(f"{'Frame Accuracy':<20} | {raw_acc:.4f}          | {smooth_acc:.4f}")
    print(f"{'Frame Macro F1':<20} | {raw_macro:.4f}          | {smooth_macro:.4f}")
    print(f"{'Edit Score':<20} | {raw_edit:.4f}          | {smooth_edit:.4f}")
    print(f"{'Segmental F1@0.10':<20} | {raw_f1s[0]:.4f}          | {smooth_f1s[0]:.4f}")
    print(f"{'Segmental F1@0.25':<20} | {raw_f1s[1]:.4f}          | {smooth_f1s[1]:.4f}")
    print(f"{'Segmental F1@0.50':<20} | {raw_f1s[2]:.4f}          | {smooth_f1s[2]:.4f}")
    print("=======================================================")

    # --- CSV EXPORT ---
    print(f"\nWriting sequence data to {csv_export_path}...")
    with open(csv_export_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "video_index", "frame_index", "ground_truth_id", "ground_truth_class", 
            "raw_pred_id", "raw_pred_class", "confidence", "smoothed_pred_id", "smoothed_pred_class"
        ])
        
        for vid_idx in range(len(completed_video_preds)):
            for frame_idx in range(len(completed_video_preds[vid_idx])):
                l = completed_video_labels[vid_idx][frame_idx]
                p_raw = completed_video_preds[vid_idx][frame_idx]
                c = completed_video_confs[vid_idx][frame_idx]
                p_smooth = smoothed_video_preds[vid_idx][frame_idx]
                
                gt_class = IDX_TO_CLASS.get(l, "Unknown")
                raw_class = IDX_TO_CLASS.get(p_raw, "Unknown")
                smooth_class = IDX_TO_CLASS.get(p_smooth, "Unknown")
                
                writer.writerow([vid_idx, frame_idx, l, gt_class, p_raw, raw_class, c, p_smooth, smooth_class])
                
    print("[SUCCESS] Export complete.")

# --- 4. Setup and Execution ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FOLD = 3
    
    print(f"Initializing Dataset for Fold {FOLD} on {device}...")
    val_dataset = MedicalStreamingDataset(
        f"./cv_folds_generated/fold{FOLD}_test.csv", 
        "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3", 
        1, 
        chunk_size=1800, 
        fps=60,            
        target_fps=30,     
        use_memory_bank=True,
        context_seconds=600, 
        context_fps=4,
        shuffle=False,      # <--- CRITICAL: Must be false for evaluation
        use_emb=True,
        emb_dim=1024,
        transform=None
    )
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)

    print("Initializing Model...")
    config = MambaTemporalConfig(d_model=1024, n_layer=8)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) 
    
    base_model = MambaTemporalSegmentation(
        config=config, 
        vision_dim=1024, 
        num_classes=len(CLASS_MAP), 
        device=device, 
        loss_fn=loss_fn
    )
    
    full_model = ContextMambav2(
        base_model=base_model.backbone, 
        d_model=1024, 
        num_classes=10, 
        num_future=3, 
        use_multihead=True
    ).to(device)
    
    #checkpoint_path = f"/scratch/lt200353-pcllm/location/cas_colon/full_shuffle/fold{FOLD}/v2_realjoint_opt_s_best_mamba_mdodel.pth"
    checkpoint_path = f"/scratch/lt200353-pcllm/location/cas_colon/full_shuffle/fold{FOLD}/test1_v2_joint_est_mamba_mdodel.pth"
    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    full_model.load_state_dict(state_dict, strict=False)
    
    csv_output_file = f"fold{FOLD}_test_predictions_with_confidence.csv"
    
    evaluate_and_export(
        model=full_model, 
        dataloader=val_loader, 
        device=device, 
        csv_export_path=csv_output_file,
        theta=0.9,    # Confidence threshold (80%)
        l_min=300,     # Minimum length in frames (30 frames = 1 second)
        bg_class=[]
    )

if __name__ == "__main__":
    main()
