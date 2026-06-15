import os
import json
import argparse
import random
import collections
import numpy as np
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# --- Import Models & CAS Dataset ---
from model.CMamba import MambaTemporalSegmentation, detach_states, apply_reset_mask
from model.ContextMamba import ContextMambav2
from dataset.cas_locationv3 import MedicalStreamingDataset

# Assuming these are available in your working directory
try:
    from train_mamba import f1_based_weights
except ImportError:
    f1_based_weights = None
    print("Warning: f1_based_weights not found in train_mamba. Defaulting to unweighted CE.")

# --- 1. CAS-COLON Class Map ---
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

# --- 2. Custom Losses ---
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

def safe_ce_loss(logits, targets, criterion):
    if (targets != -100).sum() == 0:
        return logits.sum() * 0.0 
    return criterion(logits, targets)


# --- 3. Evaluation Helpers (MS-TCN & Boundary MAE) ---
def get_labels_start_end_time(frame_wise_labels, bg_class=[-100]):
    labels, starts, ends = [], [], []
    if len(frame_wise_labels) == 0: return labels, starts, ends
    
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
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1): D[i, 0] = i
    for i in range(n_col+1): D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j]+1, D[i, j-1]+1, D[i-1, j-1]+1)
    return (1 - D[-1, -1]/max(m_row, n_col))*100 if norm else D[-1, -1]

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
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        
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


# --- 4. Training & Validation Functions ---
def train_one_epoch(model, dataloader, optimizer, device, accumulation_steps=4, 
                    lambda_smooth=0.5, lambda_jump=0.0):
    model.train()
    total_loss, steps = 0.0, 0
    worker_states = {}
    
    if f1_based_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=f1_based_weights.to(device), ignore_index=-100)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
    transition_penalty_loss = TransitionPenaltyLoss().to(device)
    optimizer.zero_grad() 

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        vision_embeddings, contexts, labels, future_labels, reset_mask, context_masks, worker_id = batch
        
        vision_embeddings = vision_embeddings.to(device)
        contexts = contexts.to(device)
        labels = labels.to(device)
        future_labels = future_labels.to(device)
        reset_mask = reset_mask.to(device)
        context_masks = context_masks.to(device)
        
        actual_K = context_masks[0].sum().int().item()
        valid_contexts = contexts[:, :actual_K, :]
        
        w_id = int(worker_id[0].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
        current_states = worker_states.get(w_id, None)
        
        if current_states is not None:
            current_states = apply_reset_mask(current_states, reset_mask)

        logits_wo_future, future_logits, logits_w_future, next_states = model(
            vision_embeddings=vision_embeddings, 
            contexts=valid_contexts,
            pass_states=current_states,
            labels=labels 
        )
        
        loss_wo = safe_ce_loss(logits_wo_future.view(-1, model.num_classes), labels.view(-1), criterion)
        loss_w  = safe_ce_loss(logits_w_future.view(-1, model.num_classes), labels.view(-1), criterion) 
        loss_future = safe_ce_loss(future_logits.view(-1, model.num_classes), future_labels.view(-1), criterion)
        
        ce_loss = (0.75*loss_wo + 1.5*loss_w + 0.75*loss_future) / 3.0
        smooth_loss = compute_temporal_smoothing_loss(logits_w_future, labels)
        jump_loss = transition_penalty_loss(logits_w_future, labels)
        
        loss = ce_loss + (lambda_smooth * smooth_loss) + (lambda_jump * jump_loss)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        worker_states[w_id] = detach_states(next_states)
        total_loss += (loss.item() * accumulation_steps)
        steps += 1
        
        if step % 50 == 0:
            print(f"  [Train] Step {step} | Total Loss: {loss.item() * accumulation_steps:.4f} "
                  f"(CE: {ce_loss.item():.4f}, Smooth: {smooth_loss.item():.4f}, Jump: {jump_loss.item():.4f})")
            
    return total_loss / (steps if steps > 0 else 1)

@torch.no_grad()
def validate(model, dataloader, device, transition_penalty_loss, 
             lambda_smooth=0.0, lambda_jump=0.0, bg_class=[-100]): 
    
    model.eval()
    total_loss, steps = 0.0, 0
    worker_states = {}
    
    if f1_based_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=f1_based_weights.to(device), ignore_index=-100)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

    all_preds, all_labels = [], []
    video_preds_dict, video_labels_dict = collections.defaultdict(list), collections.defaultdict(list)
    completed_video_preds, completed_video_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Validating")):
        vision_embeddings, contexts, labels, future_labels, reset_mask, context_masks, worker_id = batch
        vision_embeddings = vision_embeddings.to(device)
        contexts = contexts.to(device)
        labels = labels.to(device)
        future_labels = future_labels.to(device)
        reset_mask = reset_mask.to(device)
        context_masks = context_masks.to(device)
        
        actual_K = context_masks[0].sum().int().item()
        valid_contexts = contexts[:, :actual_K, :]
        
        w_id = int(worker_id[0].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
        current_states = worker_states.get(w_id, None)
        
        if current_states is not None:
            current_states = apply_reset_mask(current_states, reset_mask)

        logits_wo_future, future_logits, logits_w_future, next_states = model(
            vision_embeddings=vision_embeddings, contexts=valid_contexts,
            pass_states=current_states, labels=labels 
        )
        
        loss_wo = criterion(logits_wo_future.view(-1, model.num_classes), labels.view(-1))
        loss_w  = criterion(logits_w_future.view(-1, model.num_classes), labels.view(-1))
        loss_future = criterion(future_logits.view(-1, model.num_classes), future_labels.view(-1))
        
        ce_loss = (loss_wo + loss_w + loss_future) / 3.0
        smooth_loss = compute_temporal_smoothing_loss(logits_w_future, labels)
        jump_loss = transition_penalty_loss(logits_w_future, labels)
        
        loss = ce_loss + (lambda_smooth * smooth_loss) + (lambda_jump * jump_loss)
        total_loss += loss.item()
        steps += 1
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

            p_flat, l_flat = preds[b].cpu().numpy(), labels[b].cpu().numpy()
            valid_indices = l_flat != -100
            valid_preds, valid_labels = p_flat[valid_indices].tolist(), l_flat[valid_indices].tolist()

            all_preds.extend(valid_preds)
            all_labels.extend(valid_labels)
            video_preds_dict[b_w_id].extend(valid_preds)
            video_labels_dict[b_w_id].extend(valid_labels)

    for w_id, p_seq in video_preds_dict.items():
        if len(p_seq) > 0:
            completed_video_preds.append(p_seq)
            completed_video_labels.append(video_labels_dict[w_id])

    # MS-TCN Metrics Calculation
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

    avg_loss = total_loss / (steps if steps > 0 else 1)
    val_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
    val_f1_per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(10)), zero_division=0) if len(all_labels) > 0 else []

    return avg_loss, val_acc, val_f1_macro, val_f1_per_class, avg_edit, f1s, avg_mae

@torch.no_grad()
def cache_predictions(model, dataloader, device, save_path):
    model.eval()
    all_probs, all_labels, all_reset_masks = [], [], []
    worker_states = {}

    for step, batch in enumerate(tqdm(dataloader, desc="Caching Predictions")):
        vision_embeddings, contexts, labels, _, reset_mask, context_masks, worker_id = batch
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
            contexts=valid_contexts, pass_states=current_states, labels=labels
        )

        probs = F.softmax(logits_w_future, dim=-1)
        B, L = labels.shape
        frame_reset_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        frame_reset_mask[:, 0] = reset_mask.bool().view(-1)

        probs_flat = probs.view(-1, model.num_classes).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        reset_mask_flat = frame_reset_mask.view(-1).cpu().numpy() 

        valid_indices = labels_flat != -100

        all_probs.append(probs_flat)
        all_labels.append(labels_flat)
        all_reset_masks.append(reset_mask_flat)
        worker_states[w_id] = detach_states(next_states)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_reset_masks = np.concatenate(all_reset_masks, axis=0)

    np.savez_compressed(save_path, probs=all_probs, labels=all_labels, reset_masks=all_reset_masks)
    print(f"\n✅ Cache saved: Probs {all_probs.shape}, Labels {all_labels.shape} to: {save_path}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- 5. Main Execution ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ihdfiohsa.json", help="Path to config file")
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f: hparams = json.load(f)
    else:
        print(f"Warning: Config '{args.config}' not found. Using defaults.")
        hparams = {}

    # Structure/Config Extraction
    cfg_seed = hparams.get("seed", 411)
    cfg_fold = hparams.get("fold", 1)
    cfg_epochs = hparams.get("epochs", 50)
    cfg_chunk_size = hparams.get("chunk_size", 1800)
    cfg_base_lr = hparams.get("lr", 5e-5)
    cfg_weight_decay = hparams.get("weight_decay", 1e-3)
    cfg_patience = hparams.get("patience", 25)
    cfg_lambda_smooth = hparams.get("lambda_smooth", 0.5)
     
    # Dataset specific
    cfg_train_csv = hparams.get("train_csv", f"./cv_folds_generated/fold{cfg_fold}_train.csv")
    cfg_val_csv = hparams.get("val_csv", f"./cv_folds_generated/fold{cfg_fold}_test.csv")
    cfg_feat_dir = hparams.get("feat_dir", "/project/lt200353-pcllm/3d_report_gen/cas_colon/features_dinov3/")
    cfg_save_dir = hparams.get("save_dir", f"/project/lt200353-pcllm/3d_report_gen/cas_colon/mid_a4tune3i455_4_full_shuffle/fold{cfg_fold}/")
    
    # IMPORTANT: Temporal scale configurations to match standard CAS video
    cfg_fps = hparams.get("fps", 60)
    cfg_target_fps = hparams.get("target_fps", 30)
    cfg_context_fps = hparams.get("context_fps", 4)
    cfg_query_fps = hparams.get("query_fps", 30)
    cfg_compression_ratio = hparams.get("compression_ratio", 240.0)
    cfg_frames_per_query = hparams.get("frames_per_query", [24, 10])
    cfg_vbatch = hparams.get("vbatch", 4)

    #f1_based_weights = None

    set_seed(cfg_seed)
    g = torch.Generator()
    g.manual_seed(cfg_seed)

    os.makedirs(cfg_save_dir, exist_ok=True)
    best_model_path = "/project/lt200353-pcllm/3d_report_gen/cas_colon/mid_a4tune3i431_4_full_shuffle/fold1/lr5e5_b4.pth"
    cache_save_path = os.path.join(cfg_save_dir, f"new_predictions_fold{cfg_fold}.npz")

    # Initialize CAS Dataset
    train_dataset = MedicalStreamingDataset(
        cfg_train_csv, cfg_feat_dir, 1, chunk_size=cfg_chunk_size, 
        fps=cfg_fps, target_fps=cfg_target_fps, use_memory_bank=True,
        context_seconds=600, context_fps=cfg_context_fps, shuffle=True, use_emb=True, emb_dim=1024
    )
    val_dataset = MedicalStreamingDataset(
        cfg_val_csv, cfg_feat_dir, 1, chunk_size=cfg_chunk_size, 
        fps=cfg_fps, target_fps=cfg_target_fps, use_memory_bank=True,
        context_seconds=600, context_fps=cfg_context_fps, shuffle=False, use_emb=True, emb_dim=1024
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_action_classes = len(CLASS_MAP)
    config = MambaTemporalConfig(d_model=1024, n_layer=8)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) 
    
    model = MambaTemporalSegmentation(config=config, vision_dim=1024, num_classes=num_action_classes, device=device, loss_fn=loss_fn)
    
    # Initialize the Large RealColon Head but with CAS temporal parameters
    full_model = ContextMambav2(
        base_model=model.backbone, d_model=1024, num_classes=num_action_classes, 
        num_future=3, use_multihead=True,
        target_fps=cfg_target_fps, context_fps=cfg_context_fps, query_fps=cfg_query_fps, 
        compression_ratio=cfg_compression_ratio, frames_per_query=cfg_frames_per_query
    ).to(device)

    # Joint Optimization Setup
    for param in full_model.parameters(): param.requires_grad = True

    backbone_params, head_params = [], []
    for name, param in full_model.named_parameters():
        if 'base_model' in name: backbone_params.append(param)
        else: head_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg_base_lr},
        {'params': head_params,     'lr': cfg_base_lr} 
    ], weight_decay=cfg_weight_decay)
    
    WARMUP_EPOCHS = 10
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(cfg_epochs - WARMUP_EPOCHS), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[WARMUP_EPOCHS])

    IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)
    transition_penalty_loss = TransitionPenaltyLoss().to(device)

    print("\n--- Validation Check (Pre-Train) ---")
    val_loss, val_acc, val_f1_macro, val_f1_per_class, _, _, _ = validate(full_model, val_loader, device, transition_penalty_loss)
    print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Macro F1: {val_f1_macro:.4f}\n")
    
    best_val_loss = float('inf')
    best_val_f1 = 0
    patience_counter = 0

    # for epoch in range(cfg_epochs):
    #     print(f"\n--- Epoch {epoch+1}/{cfg_epochs} ---")
        
    #     train_dataset.set_epoch(epoch)
    #     train_loader = DataLoader(train_dataset, batch_size=None, num_workers=cfg_vbatch, worker_init_fn=seed_worker, generator=g)
    #     train_loss = train_one_epoch(full_model, train_loader, optimizer, device, lambda_smooth=cfg_lambda_smooth, accumulation_steps=cfg_vbatch)
        
    #     val_loss, val_acc, val_f1_macro, val_f1_per_class, val_edit, val_f1_overlaps, val_b_mae = validate(
    #         full_model, val_loader, device, transition_penalty_loss
    #     )
        
    #     print(f"Epoch {epoch+1} Summary:")
    #     print(f"  Train Loss:  {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    #     print(f"  Val Acc:     {val_acc:.4f} | Macro F1: {val_f1_macro:.4f}")
    #     print(f"  Edit Score:  {val_edit:.4f} | Bndry MAE: {val_b_mae:.4f}")
    #     print(f"  F1@10,25,50: {val_f1_overlaps[0]:.2f}, {val_f1_overlaps[1]:.2f}, {val_f1_overlaps[2]:.2f}")
        
    #     scheduler.step()
        
    #     if val_f1_macro > best_val_f1:
    #         best_val_f1, patience_counter = val_f1_macro, 0
    #         print(f"New best validation Macro F1 ({best_val_f1:.4f})! Saving...")
    #         torch.save(full_model.state_dict(), best_model_path)
    #     else:
    #         patience_counter += 1
    #         print(f"No improvement. Patience: {patience_counter}/{cfg_patience}")
    #         if patience_counter >= cfg_patience: 
    #             print("Early stopping triggered. Halting training.")
    #             break

    print("\n--- Starting Post-Training Analysis Cache ---")
    if os.path.exists(best_model_path):
        full_model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    cache_predictions(full_model, val_loader, device, cache_save_path)

if __name__ == "__main__":
    main()
