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

# Try to import weights, fallback if not available
try:
    from train_mamba import f1_based_weights
except ImportError:
    print("Warning: Could not import f1_based_weights. Loss calculation might be slightly off, but predictions will be unaffected.")
    f1_based_weights = None

CLASS_MAP = {
    'Terminal_Ileum': 0,
    'Cecum': 1,
    'Ascending_Colon': 2,
    'Hepatic_Flexure': 3,
    'Transverse_Colon': 4,
    'Splenic_Flexure': 5,
    'Descending_Colon': 6,
    'Sigmoid_Colon': 7,
    'Rectum': 8,
    'Anal_Canal': 9
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

# --- Helper Loss Functions (Required for validation loop) ---
def compute_temporal_smoothing_loss(logits, labels, ignore_index=-100):
    probs = F.softmax(logits, dim=-1) 
    diffs = probs[:, 1:, :] - probs[:, :-1, :]
    mse = torch.sum(diffs**2, dim=-1) 
    valid_mask = (labels[:, 1:] != ignore_index) & (labels[:, :-1] != ignore_index)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return mse[valid_mask].mean()

class TransitionPenaltyLoss(nn.Module):
    def __init__(self, num_classes=10, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        indices = torch.arange(num_classes, dtype=torch.float32)
        diff_matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        penalty_matrix = torch.clamp(diff_matrix - 1.0, min=0.0) 
        self.register_buffer('penalty_matrix', penalty_matrix)

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=-1)
        p_t = probs[:, :-1, :]  
        p_t1 = probs[:, 1:, :]  
        p_t_W = torch.matmul(p_t, self.penalty_matrix)
        expected_penalty = torch.sum(p_t_W * p_t1, dim=-1) 
        valid_mask = (labels[:, 1:] != self.ignore_index) & (labels[:, :-1] != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)
        return expected_penalty[valid_mask].mean()

# --- Evaluation & Export Function ---
@torch.no_grad()
def evaluate_and_export(model, dataloader, device, transition_penalty_loss, csv_export_path, bg_class=[0]): 
    model.eval()
    worker_states = {}
    
    # Use weights if available, otherwise standard CE
    if f1_based_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=f1_based_weights.to(device), ignore_index=-100)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
    all_preds = []
    all_labels = []
    
    video_preds_dict = collections.defaultdict(list)
    video_labels_dict = collections.defaultdict(list)
    completed_video_preds = []
    completed_video_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="Extracting Predictions")):
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
        preds = torch.argmax(logits_w_future, dim=-1) 
        
        for b in range(preds.size(0)):
            b_w_id = int(worker_id[b].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
            b_reset = bool(reset_mask[b].item()) if isinstance(reset_mask, torch.Tensor) else bool(reset_mask)
            
            if b_reset and len(video_preds_dict[b_w_id]) > 0:
                completed_video_preds.append(video_preds_dict[b_w_id])
                completed_video_labels.append(video_labels_dict[b_w_id])
                video_preds_dict[b_w_id] = []
                video_labels_dict[b_w_id] = []

            p_flat = preds[b].cpu().numpy()
            l_flat = labels[b].cpu().numpy()
            
            valid_indices = l_flat != -100
            valid_preds = p_flat[valid_indices].tolist()
            valid_labels = l_flat[valid_indices].tolist()

            all_preds.extend(valid_preds)
            all_labels.extend(valid_labels)
            
            video_preds_dict[b_w_id].extend(valid_preds)
            video_labels_dict[b_w_id].extend(valid_labels)

    # Flush remaining videos
    for w_id, p_seq in video_preds_dict.items():
        if len(p_seq) > 0:
            completed_video_preds.append(p_seq)
            completed_video_labels.append(video_labels_dict[w_id])

    # --- CSV EXPORT ---
    print(f"\nWriting sequence data to {csv_export_path}...")
    with open(csv_export_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["video_index", "frame_index", "ground_truth_id", "ground_truth_class", "prediction_id", "prediction_class"])
        
        for vid_idx, (p_seq, l_seq) in enumerate(zip(completed_video_preds, completed_video_labels)):
            for frame_idx, (p, l) in enumerate(zip(p_seq, l_seq)):
                gt_class = IDX_TO_CLASS.get(l, "Unknown")
                pred_class = IDX_TO_CLASS.get(p, "Unknown")
                writer.writerow([vid_idx, frame_idx, l, gt_class, p, pred_class])
                
    print("[SUCCESS] Export complete.")

    # --- Calculate Final Metrics ---
    overlap = [0.1, 0.25, 0.5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    edit_total = 0.0
    
    for p_seq, l_seq in zip(completed_video_preds, completed_video_labels):
        if len(p_seq) == 0: continue
        edit_total += edit_score(p_seq, l_seq, bg_class=bg_class)
        for s in range(len(overlap)):
            tp1, fp1, fn1 = iou_f1_score(p_seq, l_seq, overlap[s], bg_class=bg_class)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    avg_edit = edit_total / len(completed_video_preds) if len(completed_video_preds) > 0 else 0.0
    f1s = []
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0.0
        recall = tp[s] / float(tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(np.nan_to_num(f1) * 100)

    val_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
    
    print("\n--- Final Results ---")
    print(f"Total Videos Extracted: {len(completed_video_preds)}")
    print(f"Frame Accuracy: {val_acc:.4f}")
    print(f"Frame Macro F1: {val_f1_macro:.4f}")
    print(f"Edit Score:     {avg_edit:.4f}")
    for i, thresh in enumerate(overlap):
        print(f"F1@{thresh:.2f}:       {f1s[i]:.4f}")

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
    FOLD = 1
    
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
    
    # Check the typo in your original file name if this fails to load.
    # In your previous code, you saved it as 'mdodel.pth' but might have meant 'model.pth'
    checkpoint_path = f"/scratch/lt200353-pcllm/location/cas_colon/full_shuffle/fold{FOLD}/v2_realjoint_opt_s_best_mamba_mdodel.pth"
    
    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    full_model.load_state_dict(state_dict, strict=False)
    
    transition_penalty = TransitionPenaltyLoss().to(device)
    
    csv_output_file = f"fold{FOLD}_test_predictions.csv"
    
    evaluate_and_export(
        model=full_model, 
        dataloader=val_loader, 
        device=device, 
        transition_penalty_loss=transition_penalty,
        csv_export_path=csv_output_file
    )

if __name__ == "__main__":
    main()
