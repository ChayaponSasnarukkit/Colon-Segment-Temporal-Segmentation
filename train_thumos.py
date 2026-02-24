import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import json

# --- IMPORTS FROM YOUR FILES ---
# Assuming your model files are in the path or same directory
from model.CMamba import MambaTemporalSegmentation, detach_states, apply_reset_mask
from model.ContextMamba import ContextMamba
from dataset.thumos import ThumosStreamingDataset # Assuming you saved the fixed dataset here

# --- CONFIG ---
THUMOS_CLASSES = 22 # 20 Actions + 1 Background + 1 Ambiguous (adjust based on your json)
SEED = 42

@dataclass
class MambaTemporalConfig:
    d_model: int = 2048           # UPDATED: Thumos features (RGB+Flow) are usually 2048 dim
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

# --- LOSS FUNCTIONS (Kept yours, they are good) ---
def compute_temporal_smoothing_loss(logits, labels, ignore_index=-100):
    probs = F.softmax(logits, dim=-1)
    diffs = probs[:, 1:, :] - probs[:, :-1, :]
    mse = torch.sum(diffs**2, dim=-1)
    valid_mask = (labels[:, 1:] != ignore_index) & (labels[:, :-1] != ignore_index)
    if valid_mask.sum() == 0: return torch.tensor(0.0, device=logits.device)
    return mse[valid_mask].mean()

class TransitionPenaltyLoss(nn.Module):
    def __init__(self, num_classes=22, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        # Simple transition matrix: penalize switching classes too often
        # (You might want to tune this matrix for Thumos specifically)
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
        if not valid_mask.any(): return torch.tensor(0.0, device=logits.device)
        return expected_penalty[valid_mask].mean()

def safe_ce_loss(logits, targets, criterion):
    if (targets != -100).sum() == 0:
        return logits.sum() * 0.0 
    return criterion(logits, targets)

# --- TRAINING LOOP ---
def train_one_epoch(model, dataloader, optimizer, device, accumulation_steps=4, 
                    lambda_smooth=0.5, lambda_jump=0.0):
    model.train()
    total_loss = 0.0
    steps = 0
    worker_states = {}
    
    # Standard CE for Thumos (You can calculate class weights if needed)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    transition_penalty_loss = TransitionPenaltyLoss(num_classes=model.num_classes).to(device)

    optimizer.zero_grad() 

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Unpack batch (Matches the ThumosStreamingDataset return)
        vision_embeddings, contexts, labels, future_labels, reset_mask, context_masks, worker_id = batch
        
        vision_embeddings = vision_embeddings.to(device)
        contexts = contexts.to(device) if contexts is not None else None
        labels = labels.to(device)
        future_labels = future_labels.to(device)
        reset_mask = reset_mask.to(device)
        context_masks = context_masks.to(device) if context_masks is not None else None
        
        # Handle Context Masking (if using memory bank)
        valid_contexts = None
        if contexts is not None:
            # We take the max valid context length in the batch to save compute
            actual_K = context_masks.sum(dim=1).max().int().item()
            if actual_K > 0:
                valid_contexts = contexts[:, :actual_K, :]
            else:
                # Fallback if context is all empty
                valid_contexts = torch.zeros((contexts.shape[0], 1, contexts.shape[2]), device=device)

        # State Management
        w_id = int(worker_id[0].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
        current_states = worker_states.get(w_id, None)
        if current_states is not None:
            current_states = apply_reset_mask(current_states, reset_mask)

        # Forward Pass
        logits_wo_future, future_logits, logits_w_future, next_states = model(
            vision_embeddings=vision_embeddings, 
            contexts=valid_contexts,
            pass_states=current_states,
            labels=labels 
        )
        
        # Loss Calculation
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
            
    return total_loss / (steps if steps > 0 else 1)

# --- VALIDATION WITH MAP ---
@torch.no_grad()
def validate_map(model, dataloader, device, num_classes):
    """
    Computes Frame-level mAP (Mean Average Precision).
    This is a strong proxy for the official Thumos Instance-mAP during training.
    """
    model.eval()
    worker_states = {}
    
    # Store probability scores and one-hot labels for mAP calculation
    all_scores = []
    all_targets = []
    
    total_loss = 0.0
    steps = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for step, batch in enumerate(tqdm(dataloader, desc="Validating (mAP)")):
        vision_embeddings, contexts, labels, future_labels, reset_mask, context_masks, worker_id = batch
        
        vision_embeddings = vision_embeddings.to(device)
        labels = labels.to(device)
        # ... (Repeat context/state logic from training) ...
        contexts = contexts.to(device) if contexts is not None else None
        context_masks = context_masks.to(device) if context_masks is not None else None
        reset_mask = reset_mask.to(device)
        
        valid_contexts = None
        if contexts is not None:
            actual_K = context_masks.sum(dim=1).max().int().item()
            valid_contexts = contexts[:, :actual_K, :] if actual_K > 0 else torch.zeros((contexts.shape[0], 1, contexts.shape[2]), device=device)

        w_id = int(worker_id[0].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
        current_states = worker_states.get(w_id, None)
        if current_states is not None:
            current_states = apply_reset_mask(current_states, reset_mask)

        # Forward
        _, _, logits_w_future, next_states = model(
            vision_embeddings=vision_embeddings, 
            contexts=valid_contexts,
            pass_states=current_states,
            labels=labels 
        )
        
        loss = safe_ce_loss(logits_w_future.view(-1, num_classes), labels.view(-1), criterion)
        total_loss += loss.item()
        steps += 1
        worker_states[w_id] = detach_states(next_states)

        # --- DATA GATHERING FOR MAP ---
        probs = F.softmax(logits_w_future, dim=-1) # [B, M, C]
        
        # Flatten
        probs_flat = probs.view(-1, num_classes).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        
        # Filter padding
        valid_idx = labels_flat != -100
        
        if valid_idx.sum() > 0:
            valid_probs = probs_flat[valid_idx]
            valid_labels = labels_flat[valid_idx]
            
            # Convert targets to One-Hot for mAP calculation
            # We assume labels are 0..C-1. 
            # Note: For Thumos, often Class 0 or Class 21 is background. 
            # mAP is usually calculated on Action classes only.
            
            # One-hot encoding manual
            targets_one_hot = np.zeros((valid_labels.size, num_classes))
            targets_one_hot[np.arange(valid_labels.size), valid_labels] = 1
            
            all_scores.append(valid_probs)
            all_targets.append(targets_one_hot)

    # Concatenate all batches
    if len(all_scores) == 0: return 0.0, 0.0
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute mAP (Macro Average)
    # Note: Thumos usually cares about classes 1-20 (if 0 is BG). 
    # If your classes are 0-19 (actions) and 20 (BG), slice accordingly.
    # Here we compute for ALL classes first.
    map_per_class = average_precision_score(all_targets, all_scores, average=None)
    
    # Simple Macro mAP
    mAP_macro = np.mean(map_per_class)
    
    return total_loss / steps, mAP_macro

# --- UTILS ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- MAIN ---
def main():
    set_seed(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    
    # 1. Load JSON (contains session definitions)
    with open("thumos_annotations.json", 'r') as f:
        json_data = json.load(f)

    # 2. Datasets (Using the Fixed ThumosStreamingDataset)
    train_dataset = ThumosStreamingDataset(
        data_root="/path/to/thumos/features",
        json_data=json_data,
        batch_size_per_worker=1,
        
        chunk_size=240,
        fps=4.0, target_fps=4.0,
        
        # Matching Medical Logic
        use_memory_bank=True,
        context_seconds=600,
        context_fps=4,
        
        phase='train', shuffle=True
    )

    val_dataset = ThumosStreamingDataset(
        data_root="/path/to/thumos/features",
        json_data=json_data,
        batch_size_per_worker=1, # Val usually safer with 1
        
        chunk_size=240,
        fps=4.0, target_fps=4.0,
        
        use_memory_bank=True,
        context_seconds=600,
        context_fps=4,
        
        phase='test', shuffle=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Model Setup
    # Thumos features (concatenated RGB+Flow) are usually 2048 dims (1024+1024)
    # Check your feature files! If purely RGB, change to 1024.
    VISION_DIM = 4096 
    
    config = MambaTemporalConfig(d_model=VISION_DIM, n_layer=8)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=json_data.get('ignore_index', -100))
    
    model = MambaTemporalSegmentation(
        config=config, 
        vision_dim=VISION_DIM, 
        num_classes=THUMOS_CLASSES, 
        device=device, 
        loss_fn=loss_fn
    )
    
    full_model = ContextMamba(
        base_model=model.backbone, 
        d_model=VISION_DIM, 
        num_classes=THUMOS_CLASSES, 
        target_fps=4.0,
        context_fps=4.0,
        query_fps=4.0,
        num_future=3
    ).to(device)

    # 4. Training
    optimizer = torch.optim.AdamW(full_model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3) # Monitor mAP (max)

    epochs = 50
    best_map = 0.0
    
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Train
        train_dataset.set_epoch(epoch)
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=2, worker_init_fn=seed_worker, generator=g)
        train_loss = train_one_epoch(full_model, train_loader, optimizer, device)
        
        # Validate (mAP)
        val_loss, val_map = validate_map(full_model, val_loader, device, THUMOS_CLASSES)
        
        print(f"Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val mAP:    {val_map:.4f}")
        
        # Scheduler Step (Monitor mAP, so 'max')
        scheduler.step(val_map)
        
        if val_map > best_map:
            best_map = val_map
            print(f"New Best mAP! Saving...")
            torch.save(full_model.state_dict(), "best_thumos_mamba.pth")

if __name__ == "__main__":
    main()