import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import collections
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, f1_score

# --- Import your models and the Dataset ---
from model.CMamba import MambaTemporalSegmentation
from model.ContextMamba import ContextMambav2ForRealColon, ContextMambaCmeRT, ContextMambav2TASver
from dataset.real_locationv2 import RealColonInferStreamingDataset 

# --- 1. REALCOLON Class Map ---
CLASS_MAP = {
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

@dataclass
class MambaTemporalConfig:
    d_model: int = 1024          
    n_layer: int = 8             
    d_intermediate: int = 0      
    ssm_cfg: dict = field(default_factory=lambda: {
        "d_state": 32,           
        "d_conv": 4,             
        "expand": 4,             
        "dt_rank": "auto",       
        "layer": "Mamba1",       
        "use_fast_path": False,
    }) 
    rms_norm: bool = True        
    norm_epsilon: float = 1e-5
    fused_add_norm: bool = True  
    residual_in_fp32: bool = True 

def detach_states(states):
    if states is None:
        return None

    if isinstance(states, dict):
        detached = {}
        if states.get('base_states') is not None:
            detached['base_states'] = [
                (s.detach().clone(), c.detach().clone()) if s is not None else (None, None)
                for s, c in states['base_states']
            ]
        else:
            detached['base_states'] = None

        if 'tcn_caches' in states and states['tcn_caches'] is not None:
            tcn_caches = states['tcn_caches']
            if isinstance(tcn_caches, dict):
                detached['tcn_caches'] = {
                    stage_name: [c.detach() if c is not None else None for c in caches_list]
                    for stage_name, caches_list in tcn_caches.items()
                }
            elif isinstance(tcn_caches, list):
                detached['tcn_caches'] = [
                    c.detach() if c is not None else None for c in tcn_caches
                ]
        return detached

    return [
        (s.detach().clone(), c.detach().clone()) if s is not None else (None, None)
        for s, c in states
    ]

def apply_reset_mask(states, reset_mask):
    if states is None:
        return None

    def _mask_tensor(tensor, mask):
        if tensor is None:
            return None
        broadcast_mask = mask.view(-1, *[1]*(tensor.dim() - 1))
        return tensor.masked_fill(broadcast_mask, 0.0)

    if isinstance(states, dict):
        reset_states = {}
        if states.get('base_states') is not None:
            reset_states['base_states'] = [
                (_mask_tensor(s, reset_mask), _mask_tensor(c, reset_mask))
                for s, c in states['base_states']
            ]
        else:
            reset_states['base_states'] = None

        if 'tcn_caches' in states and states['tcn_caches'] is not None:
            tcn_caches = states['tcn_caches']
            if isinstance(tcn_caches, dict):
                reset_states['tcn_caches'] = {
                    stage_name: [_mask_tensor(c, reset_mask) for c in caches_list]
                    for stage_name, caches_list in tcn_caches.items()
                }
            elif isinstance(tcn_caches, list):
                reset_states['tcn_caches'] = [
                    _mask_tensor(c, reset_mask) for c in tcn_caches
                ]
        else:
            reset_states['tcn_caches'] = None
        return reset_states

    masked_states = []
    for ssm_state, conv_state in states:
        new_ssm = _mask_tensor(ssm_state, reset_mask)
        new_conv = _mask_tensor(conv_state, reset_mask)
        masked_states.append((new_ssm, new_conv))
    return masked_states

def compute_temporal_smoothing_loss(logits, labels, ignore_index=-100):
    probs = F.softmax(logits, dim=-1) 
    diffs = probs[:, 1:, :] - probs[:, :-1, :]
    mse = torch.sum(diffs**2, dim=-1) 
    valid_mask = (labels[:, 1:] != ignore_index) & (labels[:, :-1] != ignore_index)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return mse[valid_mask].mean()

def safe_ce_loss(logits, targets, criterion):
    if (targets != -100).sum() == 0:
        return logits.sum() * 0.0 
    return criterion(logits, targets)

# --- Config Flags ---
USE_TEMPORAL_SCALE = True

@torch.no_grad()
def evaluate(model, dataloader, device, lambda_smooth=0.0, lambda_jump=0.0):
    model.eval()
    total_loss, total_loss_wo, total_loss_w, total_loss_future, total_loss_smooth = 0.0, 0.0, 0.0, 0.0, 0.0
    steps = 0
    worker_states = {}
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    all_preds, all_labels = [], []
    prediction_cache = []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        # Note: Ensure your dataloader __iter__ returns vid_ids and frame_indices as requested
        vision_embeddings, contexts, labels, future_labels, reset_mask, context_masks, worker_id, vid_ids, frame_indices = batch
        
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

        logits_wo_future, future_logits, logits_w_future, tcn_tmp_logits, tcn_logits, next_states = model(
            vision_embeddings=vision_embeddings,
            contexts=valid_contexts,
            pass_states=current_states,
            labels=labels
        )

        loss_final = safe_ce_loss(tcn_logits.view(-1, model.num_classes), labels.view(-1), criterion)
        ce_loss = loss_final
        smooth_loss = compute_temporal_smoothing_loss(tcn_logits, labels)
        jump_loss = 0

        loss = ce_loss + (lambda_smooth * smooth_loss) + (lambda_jump * jump_loss)
        total_loss += loss.item()
        steps += 1
        worker_states[w_id] = detach_states(next_states)

        probs = torch.softmax(tcn_logits, dim=-1)
        confidences, preds = torch.max(probs, dim=-1)

        for b in range(preds.size(0)):
            b_vid_id = vid_ids[b]
            b_frame_idx = frame_indices[b].cpu().numpy()
            b_conf = confidences[b].cpu().numpy()
            
            p_flat, l_flat = preds[b].cpu().numpy(), labels[b].cpu().numpy()
            valid_indices = l_flat != -100
            
            valid_preds = p_flat[valid_indices].tolist()
            valid_labels = l_flat[valid_indices].tolist()
            
            all_preds.extend(valid_preds)
            all_labels.extend(valid_labels)

            valid_frames = b_frame_idx[valid_indices].tolist()
            valid_confs = b_conf[valid_indices].tolist()

            for f_idx, pred, label, conf in zip(valid_frames, valid_preds, valid_labels, valid_confs):
                if b_vid_id != "PAD" and f_idx != -1:  
                    prediction_cache.append({
                        "video_id": b_vid_id,
                        "frame_index": f_idx,
                        "groundtruth": label,
                        "prediction": pred,
                        "confidence": conf
                    })

    avg_loss = total_loss / (steps if steps > 0 else 1)
    
    val_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
    val_f1_per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(model.num_classes)), zero_division=0) if len(all_labels) > 0 else []

    return avg_loss, val_acc, val_f1_macro, val_f1_per_class, prediction_cache

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
    parser = argparse.ArgumentParser(description="Evaluate MambaTemporalSegmentation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save evaluation results")
    VIDEO_ROOT = "/project/lt200353-pcllm/3d_report_gen/real-colon/"
    SPLIT_DIR = "/home/csasnaru/temporal_segmentation/data/dataset/RC_lists/5_fold/" 
    FOLD = 4
    args = parser.parse_args()

    set_seed(441)
    os.makedirs(args.save_dir, exist_ok=True)
    
    val_dataset = RealColonInferStreamingDataset(
        video_root=VIDEO_ROOT, 
        batch_size_per_worker=1, 
        split_dir=SPLIT_DIR,
        fold=FOLD,
        phase='test',
        chunk_size=300, 
        num_future_seconds=12,
        #frames_per_query=[10, 6],
        fps=5,            
        target_fps=5,     
        use_memory_bank=True,
        context_seconds=600, 
        context_fps=5,
        shuffle=False,
        use_emb=True,
        emb_dim=1024,
        transform=None
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_action_classes = len(CLASS_MAP)
    config = MambaTemporalConfig(d_model=1024, n_layer=8)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) 
    
    model = MambaTemporalSegmentation(
        config=config, 
        vision_dim=1024, 
        num_classes=num_action_classes, 
        device=device, 
        loss_fn=loss_fn
    )
    
    full_model = ContextMambav2TASver(
        base_model=model.backbone, d_model=1024, num_classes=num_action_classes, 
        num_future=12, use_multihead=True,
        target_fps=5, context_fps=5, query_fps=5, num_layers_per_stage=4,
        compression_ratio=300.0, frames_per_query=[30, 10], hidden_expansion=4, num_stages=2, dropout=0.5
    ).to(device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    full_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print("Model loaded successfully.")

    IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)
    
    print("\n--- Starting Evaluation ---")
    val_loss, val_acc, val_f1_macro, val_f1_per_class, pred_cache = evaluate(
        full_model, val_loader, device
    )
    
    print(f"\nEvaluation Summary:")
    print(f"  Eval Loss:    {val_loss:.4f}")
    print(f"  Eval Acc:     {val_acc:.4f}")
    print(f"  Eval Macro F1:{val_f1_macro:.4f}")
    print("  Per-Class F1:")
    for idx, f1 in enumerate(val_f1_per_class):
        class_name = IDX_TO_CLASS.get(idx, f"Class_{idx}")
        print(f"    - {class_name:<15}: {f1:.4f}")
        
    csv_save_path = os.path.join(args.save_dir, f"evaluation_predictions_fold{args.fold}.csv")
    df_cache = pd.DataFrame(pred_cache)
    df_cache.to_csv(csv_save_path, index=False)
    print(f"\n✅ Successfully saved predictions with video IDs and frame indices to: {csv_save_path}")

if __name__ == "__main__":
    main()