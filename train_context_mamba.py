from dataclasses import dataclass, field
from model.CMamba import MambaTemporalSegmentation, detach_states, apply_reset_mask
from dataset.cas_locationv3 import MedicalStreamingDataset, CLASS_MAP
from model.ContextMamba import ContextMamba
@dataclass
class MambaTemporalConfig:
    # --- Architecture Scale ---
    d_model: int = 1024           # Projection dimension for your vision features
    n_layer: int = 8             # Number of Mamba layers (8 to 16 is usually plenty for temporal segmentation)
    d_intermediate: int = 0      # 0 means no interleaved MLPs (pure Mamba). Change to e.g. 1024 if using Jamba-style hybrid.
    
    # --- SSM Specifics ---
    ssm_cfg: dict = field(default_factory=lambda: {
        "d_state": 16,           # Increased from NLP default (16) to capture complex visual motions
        "d_conv": 4,             # Temporal convolution window (4 frames)
        "expand": 2,             # Internal feature expansion factor
        "dt_rank": "auto",       # Will default to math.ceil(d_model / 16)
        "layer": "Mamba1",       # Mamba1 is heavily tested for TBPTT. Switch to Mamba2 if you need extreme speed.
        "use_fast_path": False,
    }) 
    # --- Hardware & Stability ---
    rms_norm: bool = True        # Faster than standard LayerNorm
    norm_epsilon: float = 1e-5
    fused_add_norm: bool = True  # Keep True to use the fast Triton kernels we fixed!
    residual_in_fp32: bool = True # Crucial for deep video models to prevent gradient overflow

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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

# --- 1. Custom Loss Functions ---

def compute_temporal_smoothing_loss(logits, labels, ignore_index=-100):
    """Encourages consecutive predictions to have similar probability distributions."""
    probs = F.softmax(logits, dim=-1) # [B, M, C]
    
    # Calculate MSE between consecutive time steps
    diffs = probs[:, 1:, :] - probs[:, :-1, :]
    mse = torch.sum(diffs**2, dim=-1) # [B, M-1]
    
    # Mask out padding based on labels
    valid_mask = (labels[:, 1:] != ignore_index) & (labels[:, :-1] != ignore_index)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
        
    return mse[valid_mask].mean()

class TransitionPenaltyLoss(nn.Module):
    def __init__(self, num_classes=10, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        
        # Create and cache the penalty matrix W
        indices = torch.arange(num_classes, dtype=torch.float32)
        diff_matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        penalty_matrix = torch.clamp(diff_matrix - 1.0, min=0.0) ** 2 
        
        # register_buffer automatically moves this to the correct device with the model
        self.register_buffer('penalty_matrix', penalty_matrix)

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=-1) # [B, M, C]
        
        p_t = probs[:, :-1, :]  
        p_t1 = probs[:, 1:, :]  
        
        # Use the cached matrix
        p_t_W = torch.matmul(p_t, self.penalty_matrix)
        expected_penalty = torch.sum(p_t_W * p_t1, dim=-1) 
        
        valid_mask = (labels[:, 1:] != self.ignore_index) & (labels[:, :-1] != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)
            
        return expected_penalty[valid_mask].mean()

# --- 2. Training Loop ---

def train_one_epoch(model, dataloader, optimizer, device, accumulation_steps=4, 
                    lambda_smooth=0.5, lambda_jump=0.5):
    model.train()
    total_loss = 0.0
    steps = 0
    
    worker_states = {}
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
            labels=labels 
        )
        
        # --- Multi-Objective CrossEntropy Losses ---
        loss_wo = criterion(logits_wo_future.view(-1, model.num_classes), labels.view(-1))
        loss_w  = criterion(logits_w_future.view(-1, model.num_classes), labels.view(-1))
        loss_future = criterion(future_logits.view(-1, model.num_classes), future_labels.view(-1))
        
        ce_loss = (loss_wo + loss_w + loss_future) / 3.0

        # --- Custom Temporal Constraints ---
        # Apply constraints to the final, most refined predictions (logits_w_future)
        smooth_loss = compute_temporal_smoothing_loss(logits_w_future, labels)
        jump_loss = transition_penalty_loss(logits_w_future, labels)
        
        # Combine all objectives
        loss = ce_loss + (lambda_smooth * smooth_loss) + (lambda_jump * jump_loss)
        
        # Normalize the loss for gradient accumulation
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
                  f"(CE: {ce_loss.item():.4f}, Smooth: {smooth_loss.item():.4f}, Jump: {jump_loss.item():.4f})")
            
    return total_loss / (steps if steps > 0 else 1)

from sklearn.metrics import accuracy_score, f1_score
@torch.no_grad()
def validate(model, dataloader, device, transition_penalty_loss, 
             lambda_smooth=0.5, lambda_jump=0.5):
    model.eval()
    total_loss = 0.0
    steps = 0
    
    worker_states = {}
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    all_preds = []
    all_labels = []
    total_loss_wo = 0.0
    total_loss_w = 0.0
    total_loss_future = 0.0
    total_loss_smooth = 0.0
    total_loss_jump = 0.0

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
            labels=labels 
        )
        
        # Losses
        loss_wo = criterion(logits_wo_future.view(-1, model.num_classes), labels.view(-1))
        loss_w  = criterion(logits_w_future.view(-1, model.num_classes), labels.view(-1))
        loss_future = criterion(future_logits.view(-1, model.num_classes), future_labels.view(-1))
        
        ce_loss = (loss_wo + loss_w + loss_future) / 3.0
        smooth_loss = compute_temporal_smoothing_loss(logits_w_future, labels)
        jump_loss = transition_penalty_loss(logits_w_future, labels)
        
        loss = ce_loss + (lambda_smooth * smooth_loss) + (lambda_jump * jump_loss)
        
        total_loss += loss.item()
        total_loss_wo += loss_wo.item()
        total_loss_w += loss_w.item()
        total_loss_future += loss_future.item()
        total_loss_smooth += smooth_loss.item()
        total_loss_jump += jump_loss.item()
        steps += 1
        worker_states[w_id] = detach_states(next_states)
        
        # --- Metrics Collection ---
        # We care about the final prediction: logits_w_future
        preds = torch.argmax(logits_w_future, dim=-1) # [B, M]
        
        preds_flat = preds.view(-1).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        
        # Filter out padded areas (-100)
        valid_indices = labels_flat != -100
        
        all_preds.extend(preds_flat[valid_indices])
        all_labels.extend(labels_flat[valid_indices])

    avg_loss = total_loss / (steps if steps > 0 else 1)
    print(f"  val_loss_wo:    {total_loss_wo / steps:.4f}")
    print(f"  val_loss_w:    {total_loss_w / steps:.4f}")
    print(f"  val_loss_future:    {total_loss_future / steps:.4f}")
    print(f"  val_loss_smooth:    {total_loss_smooth / steps:.4f}")
    print(f"  val_loss_jump:    {total_loss_jump / steps:.4f}")
    # Calculate Sklearn Metrics
    val_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0.0
    val_f1_per_class = f1_score(all_labels, all_preds, average=None, labels=list(range(model.num_classes)), zero_division=0) if len(all_labels) > 0 else []

    return avg_loss, val_acc, val_f1_macro, val_f1_per_class

def main():
    freeze = True
    train_dataset = MedicalStreamingDataset(
        "/scratch/lt200353-pcllm/location/cas_colon/updated_train_split.csv", 
        "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3", 
        1, 
        chunk_size=1800, # 1 minute so we dont need to deal with edge case where context need to be recalculate
        
        # FPS Configuration
        fps=60,            # Source Video FPS
        target_fps=30,     # Desired Training FPS (New Argument)
        
        # Context / Memory Bank Config
        use_memory_bank=True,
        context_seconds=600, # whole video, average total of 10 minutes = 10*7 420*30
        context_fps=4,
        shuffle = True,

        use_emb=True,
        emb_dim=1024,
        transform=None)

    val_dataset = MedicalStreamingDataset(
        "/scratch/lt200353-pcllm/location/cas_colon/updated_test_split.csv", 
        "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3", 
        1, 
        chunk_size=1800, 
        
        # FPS Configuration
        fps=60,            # Source Video FPS
        target_fps=30,     # Desired Training FPS (New Argument)
        
        # Context / Memory Bank Config
        use_memory_bank=True,
        context_seconds=600, 
        context_fps=4,
        shuffle = False,

        use_emb=True,
        emb_dim=1024,
        transform=None)
    
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
    
    # 4. Load the Model Weights
    checkpoint_path = "checkpoints/base_shuffle_focal/best_small_mamba_model_4096.pth"
    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.to(device)
    if freeze==True:
        for param in model.parameters():
            param.requires_grad = False
    
    full_model = ContextMamba(base_model=model.backbone, d_model=1024, num_classes=10, num_future=3).to(device)
    # --- Training Configuration ---
    epochs = 50
    patience = 12  # How many epochs to wait for improvement before stopping
    patience_counter = 0
    best_val_loss = float('inf')
    save_dir = "./checkpoints/full_shuffle"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_mamba_model.pth")

    # Optimizer & Scheduler
    # AdamW is highly recommended for SSMs/Transformers
    optimizer = torch.optim.AdamW(full_model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Reduce learning rate by half if validation loss stops improving for 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # --- Main Training Loop ---
    IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)
    transition_penalty_loss = TransitionPenaltyLoss().to(device)
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # 1. Train
        train_dataset.set_epoch(epoch)
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=2)
        train_loss = train_one_epoch(full_model, train_loader, optimizer, device)
        
        # 2. Validate (Now receiving metrics)
        val_loss, val_acc, val_f1_macro, val_f1_per_class = validate(full_model, val_loader, device, transition_penalty_loss)
        
        # 3. Print Summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss:  {train_loss:.4f}")
        print(f"  Val Loss:    {val_loss:.4f}")
        print(f"  Val Acc:     {val_acc:.4f}")
        print(f"  Val Macro F1:{val_f1_macro:.4f}")
        
        print("  Per-Class F1:")
        # Iterate through the returned per-class F1 array
        for idx, f1 in enumerate(val_f1_per_class):
            # Gracefully handle if a class index isn't in the map for some reason
            class_name = IDX_TO_CLASS.get(idx, f"Class_{idx}")
            print(f"    - {class_name:<15}: {f1:.4f}")
        
        # 4. Step the scheduler based on validation loss
        scheduler.step(val_loss)
        
        # 5. Save Model & Early Stopping Logic
        # (You might want to change early stopping to track val_f1_macro instead of val_loss!)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"\nNew best validation loss ({best_val_loss:.4f})! Saving model...")
            torch.save(full_model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"\nNo improvement. Early stopping patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered. Halting training.")
                break

if __name__ == "__main__":
    main()
