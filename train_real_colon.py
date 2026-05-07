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
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, f1_score

# --- Import your models and the NEW Dataset ---
from model.CMamba import MambaTemporalSegmentation, detach_states, apply_reset_mask
from model.ContextMamba import ContextMambaForRealColon, ContextMambaCmeRT

# IMPORTANT: Make sure to save the newly created dataset class in a file, e.g., dataset.real_colon
from dataset.real_locationv2 import RealColonStreamingDataset 

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
    #"uncertain": -100,
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

# --- Custom Loss Functions ---
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

# --- Config Flags ---
USE_CMERT_HEAD = False
USE_TEMPORAL_SCALE = True

def train_one_epoch(model, dataloader, optimizer, device, accumulation_steps=16, 
                    lambda_smooth=0.5, lambda_jump=0.0, with_future=True, weighted=False):
    model.train()
    total_loss = 0.0
    steps = 0
    worker_states = {}
    
    if weighted:
        # You must calculate these based on your train split, but it will look like this:
        # (Heavier weights for rare classes, smaller weights for common classes)
        #realcolon_weights = torch.tensor([1.2, 1.5, 3.0, 8.0, 2.5, 1.0, 4.0, 1.0, 5.0]).to(device)
        #realcolon_weights = torch.tensor([0.554, 0.591, 0.869, 1.025, 1.336, 0.934, 1.682, 1.102, 0.906]).to(device)
        #realcolon_weights = torch.tensor([0.28, 0.36, 0.86, 2.49, 0.89, 0.30, 1.60, 0.32, 1.47]).to(device) # damp
        realcolon_weights = torch.tensor([0.48, 0.55, 0.93, 1.84, 1.09, 0.63, 1.50, 0.72, 1.28]).to(device) #avg
        criterion = nn.CrossEntropyLoss(weight=realcolon_weights, ignore_index=-100)
    else:

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    transition_penalty_loss = TransitionPenaltyLoss().to(device)

    optimizer.zero_grad() 

    for step, batch in enumerate(tqdm(dataloader)):
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
            labels=labels,
            use_temporal_scale=True
        )
        
        if with_future:
            loss_wo = safe_ce_loss(logits_wo_future.view(-1, model.num_classes), labels.view(-1), criterion)
            loss_w  = safe_ce_loss(logits_w_future.view(-1, model.num_classes), labels.view(-1), criterion)
            
            if isinstance(model, ContextMambaCmeRT):
                loss_future = safe_ce_loss(future_logits.view(-1, model.num_classes), future_labels[:, -1, :].view(-1), criterion)
            else:
                loss_future = safe_ce_loss(future_logits.view(-1, model.num_classes), future_labels.view(-1), criterion)
        
            ce_loss = (0.75*loss_wo + 1.5*loss_w + 0.75*loss_future) / 3.0
            smooth_loss = compute_temporal_smoothing_loss(logits_w_future, labels)
            jump_loss = 0#transition_penalty_loss(logits_w_future, labels)
        else:
            ce_loss = safe_ce_loss(logits_wo_future.view(-1, model.num_classes), labels.view(-1), criterion)
            smooth_loss = compute_temporal_smoothing_loss(logits_w_future, labels)
            jump_loss = 0#transition_penalty_loss(logits_w_future, labels)
        
        loss = ce_loss + (lambda_smooth * smooth_loss) + (lambda_jump * jump_loss)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        worker_states[w_id] = detach_states(next_states)
        
        total_loss += (loss.item() * accumulation_steps)
        steps += 1
        
        if step % 10 == 0:
            print(f"  [Train] Step {step} | Total Loss: {loss.item() * accumulation_steps:.4f} "
                  f"(CE: {ce_loss.item():.4f}, Smooth: {smooth_loss.item():.4f}"), #Jump: {jump_loss.item():.4f})")
            
    return total_loss / (steps if steps > 0 else 1)

@torch.no_grad()
def validate(model, dataloader, device, transition_penalty_loss, 
             lambda_smooth=0.0, lambda_jump=0.0, with_future=True, weighted=False):
    model.eval()
    total_loss, total_loss_wo, total_loss_w, total_loss_future, total_loss_smooth, total_loss_jump = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    steps = 0
    worker_states = {}
    if weighted:
        # You must calculate these based on your train split, but it will look like this:
        # (Heavier weights for rare classes, smaller weights for common classes)
        #realcolon_weights = torch.tensor([1.2, 1.5, 3.0, 8.0, 2.5, 1.0, 4.0, 1.0, 5.0]).to(device)
        #realcolon_weights = torch.tensor([0.554, 0.591, 0.869, 1.025, 1.336, 0.934, 1.682, 1.102, 0.906]).to(device)
        #realcolon_weights = torch.tensor([0.28, 0.36, 0.86, 2.49, 0.89, 0.30, 1.60, 0.32, 1.47]).to(device) # damp
        realcolon_weights = torch.tensor([0.48, 0.55, 0.93, 1.84, 1.09, 0.63, 1.50, 0.72, 1.28]).to(device) #avg
        criterion = nn.CrossEntropyLoss(weight=realcolon_weights, ignore_index=-100)
    else:

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    all_preds, all_labels = [], []

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
            vision_embeddings=vision_embeddings, 
            contexts=valid_contexts,
            pass_states=current_states,
            labels=labels,
            use_temporal_scale=USE_TEMPORAL_SCALE
        )
        
        if with_future:
            loss_wo = safe_ce_loss(logits_wo_future.view(-1, model.num_classes), labels.view(-1), criterion)
            loss_w  = safe_ce_loss(logits_w_future.view(-1, model.num_classes), labels.view(-1), criterion)
            
            if isinstance(model, ContextMambaCmeRT):
                loss_future = safe_ce_loss(future_logits.view(-1, model.num_classes), future_labels[:, -1, :].view(-1), criterion)
            else:
                loss_future = safe_ce_loss(future_logits.view(-1, model.num_classes), future_labels.view(-1), criterion)

            ce_loss = (0.75*loss_wo + 1.5*loss_w + 0.75*loss_future) / 3.0
            smooth_loss = compute_temporal_smoothing_loss(logits_w_future, labels)
            #jump_loss = 0#transition_penalty_loss(logits_w_future, labels)
        else:
            ce_loss = safe_ce_loss(logits_wo_future.view(-1, model.num_classes), labels.view(-1), criterion)
            loss_wo = ce_loss
            smooth_loss = compute_temporal_smoothing_loss(logits_w_future, labels)
            #jump_loss = 0#transition_penalty_loss(logits_w_future, labels)
        
        loss = ce_loss + (lambda_smooth * smooth_loss) #+ (lambda_jump * jump_loss)
        
        total_loss += loss.item()
        total_loss_wo += loss_wo.item()
        if with_future:
            total_loss_w += loss_w.item()
            total_loss_future += loss_future.item()
        total_loss_smooth += smooth_loss.item()
        #total_loss_jump += jump_loss.item()
        steps += 1
        worker_states[w_id] = detach_states(next_states)
        
        if with_future:
            preds = torch.argmax(logits_w_future, dim=-1) 
        else:
            preds = torch.argmax(logits_wo_future, dim=-1)
        
        preds_flat = preds.view(-1).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        valid_indices = labels_flat != -100
        
        all_preds.extend(preds_flat[valid_indices])
        all_labels.extend(labels_flat[valid_indices])

    avg_loss = total_loss / (steps if steps > 0 else 1)
    print(f"  val_loss_wo:     {total_loss_wo / steps:.4f}")
    print(f"  val_loss_w:      {total_loss_w / steps:.4f}")
    if with_future:
        print(f"  val_loss_future: {total_loss_future / steps:.4f}")
    print(f"  val_loss_smooth: {total_loss_smooth / steps:.4f}")
    #print(f"  val_loss_jump:   {total_loss_jump / steps:.4f}")
    
    val_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
    val_f1_per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(model.num_classes)), zero_division=0) if len(all_labels) > 0 else []

    return avg_loss, val_acc, val_f1_macro, val_f1_per_class
"""
@torch.no_grad()
def cache_predictions(model, dataloader, device, save_path, with_future=True):
    
    #Runs inference on the dataset, computes raw probabilities,
    #and saves them alongside ground truth labels and reset masks.
    
    model.eval()
    all_probs = []
    all_labels = []
    all_reset_masks = [] # <-- NEW: List to hold reset masks
    worker_states = {}

    for step, batch in enumerate(tqdm(dataloader, desc="Caching Predictions")):
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

        # Forward pass
        logits_wo_future, future_logits, logits_w_future, next_states = model(
            vision_embeddings=vision_embeddings,
            contexts=valid_contexts,
            pass_states=current_states,
            labels=labels,
            use_temporal_scale=USE_TEMPORAL_SCALE
        )

        # Choose which logits to use for probabilities
        logits = logits_w_future if with_future else logits_wo_future

        # Apply Softmax to get raw probabilities
        probs = F.softmax(logits, dim=-1)

        # Flatten tensors
        probs_flat = probs.view(-1, model.num_classes).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        reset_mask_flat = reset_mask.view(-1).cpu().numpy() # <-- NEW: Flatten mask

        # Filter out ignored labels (-100)
        valid_indices = labels_flat != -100

        all_probs.append(probs_flat[valid_indices])
        all_labels.append(labels_flat[valid_indices])
        all_reset_masks.append(reset_mask_flat[valid_indices]) # <-- NEW: Store filtered mask

        # Update worker states for streaming
        worker_states[w_id] = detach_states(next_states)

    # Concatenate lists into final numpy arrays
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_reset_masks = np.concatenate(all_reset_masks, axis=0) # <-- NEW: Concatenate masks

    # Save to disk as a compressed .npz file (added reset_masks)
    np.savez_compressed(
        save_path,
        probs=all_probs,
        labels=all_labels,
        reset_masks=all_reset_masks # <-- NEW: Save to file
    )
    print(f"\n✅ Successfully saved raw probabilities {all_probs.shape}, labels {all_labels.shape}, and reset masks {all_reset_masks.shape} to: {save_path}")
"""

@torch.no_grad()
def cache_predictions(model, dataloader, device, save_path, with_future=True):
    """
    Runs inference on the dataset, computes raw probabilities,
    and saves them alongside ground truth labels and reset masks.
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_reset_masks = []
    worker_states = {}

    for step, batch in enumerate(tqdm(dataloader, desc="Caching Predictions")):
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

        # Forward pass
        logits_wo_future, future_logits, logits_w_future, next_states = model(
            vision_embeddings=vision_embeddings,
            contexts=valid_contexts,
            pass_states=current_states,
            labels=labels,
            use_temporal_scale=USE_TEMPORAL_SCALE
        )

        # Choose which logits to use for probabilities
        logits = logits_w_future if with_future else logits_wo_future
        probs = F.softmax(logits, dim=-1)

        # --- FIX: Convert chunk-level reset_mask to frame-level ---
        B, L = labels.shape
        # Create an array of False for the whole sequence length
        frame_reset_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        # Apply the reset flag ONLY to the very first frame of this chunk
        frame_reset_mask[:, 0] = reset_mask.bool().view(-1)

        # Flatten tensors
        probs_flat = probs.view(-1, model.num_classes).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        reset_mask_flat = frame_reset_mask.view(-1).cpu().numpy() # Now size 600!

        # Filter out ignored labels (-100)
        valid_indices = labels_flat != -100

        all_probs.append(probs_flat[valid_indices])
        all_labels.append(labels_flat[valid_indices])
        all_reset_masks.append(reset_mask_flat[valid_indices])

        # Update worker states for streaming
        worker_states[w_id] = detach_states(next_states)

    # Concatenate lists into final numpy arrays
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_reset_masks = np.concatenate(all_reset_masks, axis=0)

    # Save to disk as a compressed .npz file
    np.savez_compressed(
        save_path,
        probs=all_probs,
        labels=all_labels,
        reset_masks=all_reset_masks
    )
    print(f"\n✅ Successfully saved raw probabilities {all_probs.shape}, labels {all_labels.shape}, and reset masks {all_reset_masks.shape} to: {save_path}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # --- Parse Hparams from Config File ---
    parser = argparse.ArgumentParser(description="Train MambaTemporalSegmentation with Hparams Config")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        hparams = json.load(f)

    cfg_lr = hparams.get("lr", 1e-4)
    cfg_weight_decay = hparams.get("weight_decay", 1e-2)
    cfg_scheduler_choice = hparams.get("scheduler_choice", "reduceonplateau").lower()
    cfg_virtual_batch_size = hparams.get("virtual_batch_size", 16)
    cfg_freeze = hparams.get("freeze", False)
    cfg_pretrain_dir = hparams.get("pretrain_dir", "")
    cfg_weighted_loss = hparams.get("weighted_loss", False)

    print(f"Loaded Hyperparameters: {hparams}")

    set_seed(42)
    g = torch.Generator()
    g.manual_seed(42)
    
    # PATH CONFIGURATIONS
    VIDEO_ROOT = "/scratch/lt200353-pcllm/location/real_colon/dataset/features_dinov3"
    SPLIT_DIR = "/home/csasnaru/temporal_segmentation/data/dataset/RC_lists/5_fold/" 
    FOLD = 1
    
    train_dataset = RealColonStreamingDataset(
        video_root=VIDEO_ROOT, 
        batch_size_per_worker=1, 
        split_dir=SPLIT_DIR,
        fold=FOLD,
        phase='train',
        chunk_size=600, 
        fps=5,            
        target_fps=5,     
        use_memory_bank=True,
        context_seconds=600, 
        context_fps=1,
        shuffle=True,
        use_emb=True,
        emb_dim=1024,
        transform=None
    )

    val_dataset = RealColonStreamingDataset(
        video_root=VIDEO_ROOT, 
        batch_size_per_worker=1, 
        split_dir=SPLIT_DIR,
        fold=FOLD,
        phase='test',
        chunk_size=600, 
        fps=5,            
        target_fps=5,     
        use_memory_bank=True,
        context_seconds=600, 
        context_fps=1,
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
    
    # --- Pretrain & Freeze Safeguard Logic ---
    is_training_from_scratch = True
    checkpoint_path = os.path.join(cfg_pretrain_dir, "best_mamba_model.pth") if cfg_pretrain_dir else ""

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        is_training_from_scratch = False
    else:
        print(f"Checkpoint not found or not provided at: '{checkpoint_path}'. Starting from scratch.")
        is_training_from_scratch = True
        
    model.to(device)

    if cfg_freeze:
        if is_training_from_scratch:
            print("WARNING: Cannot freeze base memory when training from scratch. Forcing freeze=False.")
            cfg_freeze = False
        else:
            for param in model.parameters():
                param.requires_grad = False
            print("The base short-term memory layers are frozen!")

    if USE_CMERT_HEAD:
        full_model = ContextMambaCmeRT(base_model=model.backbone, d_model=1024, num_classes=num_action_classes, num_future=3).to(device)
    else:
        print("correct choice")
        full_model = ContextMambaForRealColon(base_model=model.backbone, d_model=1024, num_classes=num_action_classes, num_future=3).to(device)
        
    epochs = 50
    patience = int(epochs//2)  
    patience_counter = 0
    best_val_loss = float('inf')
    save_dir = f"/scratch/lt200353-pcllm/location/real_colon/checkpoints/full_shuffle/mid_smoothing_dampw_realcolon_fold{FOLD}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "large_batch16best_mamba_model.pth")

    # --- Setup Optimizer and Schedulers from Config ---
    optimizer = torch.optim.AdamW(full_model.parameters(), lr=cfg_lr, weight_decay=cfg_weight_decay)
    
    if cfg_scheduler_choice == "reduceonplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    elif cfg_scheduler_choice == "cosine_with_warmup":
        warmup_epochs = 2
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        raise ValueError(f"Unsupported scheduler choice: {cfg_scheduler_choice}")

    IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)
    transition_penalty_loss = TransitionPenaltyLoss().to(device)
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        train_dataset.set_epoch(epoch)
        # Apply virtual_batch_size to num_workers
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=cfg_virtual_batch_size, worker_init_fn=seed_worker, generator=g)
        
        # Apply virtual_batch_size to accumulation_steps
        train_loss = train_one_epoch(
            full_model, train_loader, optimizer, device, 
            accumulation_steps=2*cfg_virtual_batch_size, 
            lambda_smooth=0.10, lambda_jump=0.0, with_future=True, weighted=cfg_weighted_loss
        )
        
        val_loss, val_acc, val_f1_macro, val_f1_per_class = validate(full_model, val_loader, device, transition_penalty_loss, with_future=True, weighted=cfg_weighted_loss)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss:  {train_loss:.4f}")
        print(f"  Val Loss:    {val_loss:.4f}")
        print(f"  Val Acc:     {val_acc:.4f}")
        print(f"  Val Macro F1:{val_f1_macro:.4f}")
        
        print("  Per-Class F1:")
        for idx, f1 in enumerate(val_f1_per_class):
            class_name = IDX_TO_CLASS.get(idx, f"Class_{idx}")
            print(f"    - {class_name:<15}: {f1:.4f}")
        
        # --- Scheduler Stepping Logic ---
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            # For SequentialLR (Cosine with Warmup), step without arguments
            scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"\nNew best validation loss ({val_loss:.4f})! Saving model...")
            torch.save(full_model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"\nNo improvement. Early stopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Halting training.")
                break
    print("\n--- Starting Post-Training Analysis Cache ---")
    
    # Load the best model weights
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        full_model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model path not found. Using current model state.")

    # Define where to save the analysis data
    cache_save_path = os.path.join(save_dir, f"predictions_fold{FOLD}.npz")
    
    # Run the caching function on the validation loader
    cache_predictions(
        model=full_model, 
        dataloader=val_loader, 
        device=device, 
        save_path=cache_save_path, 
        with_future=True
    )

if __name__ == "__main__":
    main()
