from dataclasses import dataclass, field
from model.CMamba import MambaTemporalSegmentation, detach_states, apply_reset_mask
from dataset.cas_locationv3 import MedicalStreamingDataset, CLASS_MAP
@dataclass
class MambaTemporalConfig:
    # --- Architecture Scale ---
    d_model: int = 1024           # Projection dimension for your vision features
    n_layer: int = 8             # Number of Mamba layers (8 to 16 is usually plenty for temporal segmentation)
    d_intermediate: int = 0      # 0 means no interleaved MLPs (pure Mamba). Change to e.g. 1024 if using Jamba-style hybrid.
    
    # --- SSM Specifics ---
    ssm_cfg: dict = field(default_factory=lambda: {
        "d_state": 32,           # Increased from NLP default (16) to capture complex visual motions
        "d_conv": 4,             # Temporal convolution window (4 frames)
        "expand": 2,             # Internal feature expansion factor
        "dt_rank": "auto",       # Will default to math.ceil(d_model / 16)
        "layer": "Mamba1",       # Mamba1 is heavily tested for TBPTT. Switch to Mamba2 if you need extreme speed.
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

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    steps = 0
    current_states = None 
    
    for step, (vision_embeddings, reset_mask, labels) in enumerate(dataloader):
        vision_embeddings = vision_embeddings.to(device)
        labels = labels.to(device)
        
        # Wipe states if a new video starts
        if reset_mask.any() and current_states is not None:
            current_states = apply_reset_mask(current_states, reset_mask)

        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(
            vision_embeddings=vision_embeddings, 
            pass_states=current_states,
            labels=labels
        )
        
        # Backpropagate
        outputs.loss.backward()
        optimizer.step()
        
        # Detach states to prevent OOM
        current_states = detach_states(outputs.next_states)
        
        total_loss += outputs.loss.item()
        steps += 1
        
        if step % 50 == 0:
            print(f"  [Train] Step {step} | Loss: {outputs.loss.item():.4f}")
            
    return total_loss / (steps if steps > 0 else 1)

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def validate(model, dataloader, device, ignore_index=-100):
    model.eval()
    total_loss = 0.0
    steps = 0
    current_states = None
    
    # Lists to store flattened predictions and labels for the whole epoch
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for step, (vision_embeddings, reset_mask, labels) in enumerate(dataloader):
            vision_embeddings = vision_embeddings.to(device)
            labels = labels.to(device)
            
            if reset_mask.any() and current_states is not None:
                current_states = apply_reset_mask(current_states, reset_mask)

            outputs = model(
                vision_embeddings=vision_embeddings,
                pass_states=current_states,
                labels=labels
            )
            
            # Assuming your model returns logits in outputs.logits 
            # Shape: (Batch, Time, Num_Classes) or (Time, Num_Classes)
            logits = outputs.logits 
            preds = torch.argmax(logits, dim=-1)
            
            # Flatten to 1D arrays
            preds_flat = preds.cpu().numpy().flatten()
            labels_flat = labels.cpu().numpy().flatten()
            
            # Filter out ignored indices (like padding)
            valid_mask = labels_flat != ignore_index
            
            all_preds.extend(preds_flat[valid_mask])
            all_labels.extend(labels_flat[valid_mask])

            current_states = detach_states(outputs.next_states)
            total_loss += outputs.loss.item()
            steps += 1
            
    # Compute overall metrics
    val_loss = total_loss / (steps if steps > 0 else 1)
    
    # Calculate Accuracy and F1
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Calculate Per-Class F1 (Returns an array of F1 scores matching the class indices)
    # We pass labels=range(num_classes) to ensure the array aligns perfectly with your CLASS_MAP
    num_classes = len(np.unique(all_labels)) # Or pass len(CLASS_MAP) into the function
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
            
    return val_loss, acc, f1_macro, f1_per_class

def main():
    train_dataset = MedicalStreamingDataset(
        "/scratch/lt200353-pcllm/location/cas_colon/updated_train_split.csv", 
        "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3", 
        2, 
        chunk_size=4096, 
        
        # FPS Configuration
        fps=60,            # Source Video FPS
        target_fps=30,     # Desired Training FPS (New Argument)
        
        # Context / Memory Bank Config
        use_memory_bank=False,
        context_seconds=300, 
        context_fps=1,
        shuffle = True,

        use_emb=True,
        emb_dim=1024,
        transform=None)

    val_dataset = MedicalStreamingDataset(
        "/scratch/lt200353-pcllm/location/cas_colon/updated_val_split.csv", 
        "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3", 
        2, 
        chunk_size=4096, 
        
        # FPS Configuration
        fps=60,            # Source Video FPS
        target_fps=30,     # Desired Training FPS (New Argument)
        
        # Context / Memory Bank Config
        use_memory_bank=False,
        context_seconds=300, 
        context_fps=1,
        shuffle = False,

        use_emb=True,
        emb_dim=1024,
        transform=None)
    
    vision_feature_dim = 1024
    num_action_classes = len(CLASS_MAP)

    config = MambaTemporalConfig(d_model=1024, n_layer=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaTemporalSegmentation(
        config=config, 
        vision_dim=vision_feature_dim, 
        num_classes=num_action_classes, 
        device=device, 
        # dtype=torch.bfloat16 # bfloat16 is highly recommended for Mamba training
    )
    # --- Training Configuration ---
    epochs = 50
    patience = 5  # How many epochs to wait for improvement before stopping
    patience_counter = 0
    best_val_loss = float('inf')
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_mamba_model.pth")

    # Optimizer & Scheduler
    # AdamW is highly recommended for SSMs/Transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Reduce learning rate by half if validation loss stops improving for 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # --- Main Training Loop ---
    IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=2)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # 1. Train
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=2)
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # 2. Validate (Now receiving metrics)
        val_loss, val_acc, val_f1_macro, val_f1_per_class = validate(model, val_loader, device)
        
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
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"\nNo improvement. Early stopping patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered. Halting training.")
                break

if __name__ == "__main__":
    main()