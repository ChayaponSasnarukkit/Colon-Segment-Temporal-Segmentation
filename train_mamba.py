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
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    steps = 0
    
    # Dictionary to isolate Mamba states for each dataloader worker process
    worker_states = {}

    for step, (vision_embeddings, contexts, labels, reset_mask, context_masks, worker_id) in enumerate(tqdm(dataloader)):
        vision_embeddings = vision_embeddings.to(device)
        labels = labels.to(device)
        reset_mask = reset_mask.to(device)
        
        # Extract the integer ID for the current worker. 
        # (Dataloader collates this into a tensor, so we pull the first item)
        w_id = int(worker_id[0].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
        
        # Retrieve the correct state history for this specific worker
        current_states = worker_states.get(w_id, None)
        
        # Wipe states if a new video starts within this worker's batch
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
        # Save the detached states back to this specific worker's dictionary key
        worker_states[w_id] = detach_states(outputs.next_states)
        
        total_loss += outputs.loss.item()
        steps += 1
        
        if step % 5 == 0:
            print(f"  [Train] Step {step} | Loss: {outputs.loss.item():.4f}")
            
    return total_loss / (steps if steps > 0 else 1)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def validate(model, dataloader, device, ignore_index=-100):
    model.eval()
    
    # 1. State Isolation: Track Mamba's memory per worker
    worker_states = {}
    
    # 2. Prediction Isolation: Track predictions per unique stream
    # Key: (worker_id, stream_index_within_batch) -> Value: {'preds': [], 'labels': []}
    stream_buffers = {}
    
    all_preds_global = []
    all_labels_global = []
    video_count = 0
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for step, (vision_embeddings, context, labels, reset_mask, context_mask, worker_id) in enumerate(tqdm(dataloader)):
            vision_embeddings = vision_embeddings.to(device)
            labels = labels.to(device)
            reset_mask = reset_mask.to(device)
            
            # Extract the integer ID for the current worker
            w_id = int(worker_id[0].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
            
            # Retrieve the correct state history for this specific worker
            current_states = worker_states.get(w_id, None)
            
            # Mamba states usually have batch as the first dimension. 
            # apply_reset_mask will zero out states for the specific streams where reset_mask is True.
            if reset_mask.any() and current_states is not None:
                current_states = apply_reset_mask(current_states, reset_mask)

            outputs = model(
                vision_embeddings=vision_embeddings,
                pass_states=current_states,
                labels=labels
            )
            
            # Save the detached states back to this specific worker's dictionary key
            worker_states[w_id] = detach_states(outputs.next_states)
            
            total_loss += outputs.loss.item()
            steps += 1
            
            logits = outputs.logits 
            preds = torch.argmax(logits, dim=-1)
            
            batch_size = preds.shape[0]
            
            # 3. Untangle the interleaved batches
            for b in range(batch_size):
                stream_key = (w_id, b)
                if stream_key not in stream_buffers:
                    stream_buffers[stream_key] = {'preds': [], 'labels': []}
                
                # If reset_mask is True AND we have data, the PREVIOUS video on this stream just finished
                if reset_mask[b].item() and len(stream_buffers[stream_key]['preds']) > 0:
                    vid_preds = np.concatenate(stream_buffers[stream_key]['preds'])
                    vid_labels = np.concatenate(stream_buffers[stream_key]['labels'])
                        
                    video_count += 1
                    
                    # Accumulate to global metrics (filtering ignore_index)
                    valid_mask = vid_labels != ignore_index
                    all_preds_global.extend(vid_preds[valid_mask])
                    all_labels_global.extend(vid_labels[valid_mask])
                    
                    # Reset the buffer for the new video starting on this stream
                    stream_buffers[stream_key] = {'preds': [], 'labels': []}
                
                # Add the current chunk to the stream buffer
                stream_buffers[stream_key]['preds'].append(preds[b].cpu().numpy())
                stream_buffers[stream_key]['labels'].append(labels[b].cpu().numpy())

    # 4. Process any remaining videos left in the buffers after the loop ends
    for stream_key, buffer in stream_buffers.items():
        if len(buffer['preds']) > 0:
            vid_preds = np.concatenate(buffer['preds'])
            vid_labels = np.concatenate(buffer['labels'])
                
            video_count += 1
            
            valid_mask = vid_labels != ignore_index
            all_preds_global.extend(vid_preds[valid_mask])
            all_labels_global.extend(vid_labels[valid_mask])

    # 5. Compute overall metrics
    val_loss = total_loss / (steps if steps > 0 else 1)
    acc = accuracy_score(all_labels_global, all_preds_global)
    f1_macro = f1_score(all_labels_global, all_preds_global, average='macro', zero_division=0)
    
    # Calculate Per-Class F1 (Returns an array of F1 scores matching the class indices)
    f1_per_class = f1_score(all_labels_global, all_preds_global, average=None, zero_division=0)
            
    return val_loss, acc, f1_macro, f1_per_class

NUM_CLASSES = len(CLASS_MAP)
def parse_intervals(time_str):
    intervals = []
    if pd.isna(time_str) or str(time_str).strip() in ['', '-', 'nan']: return []
    segments = str(time_str).split('/')
    for seg in segments:
        try:
            if '-' not in seg: continue
            s, e = seg.strip().split('-')
            def to_s(x): return int(x.split(':')[0])*60 + int(x.split(':')[1])
            intervals.append((to_s(s), to_s(e)))
        except: pass
    return intervals

def compute_class_weights(csv_path, fps=60):
    """
    Parses the dataset CSV to compute inverse frequency class weights 
    for Weighted Cross-Entropy Loss.
    """
    df = pd.read_csv(csv_path)
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    # 1. Accumulate frame counts for each class
    for _, row in df.iterrows():
        for cls_name, cls_idx in CLASS_MAP.items():
            if cls_name not in row: 
                continue
                
            intervals = parse_intervals(row[cls_name])
            for (s, e) in intervals:
                start_f = int(s * fps)
                end_f = int(e * fps)
                # Adding +1 to match your build_dense_label logic: [start_f:end_f+1]
                frame_count = max(0, (end_f - start_f) + 1)
                class_counts[cls_idx] += frame_count

    # 2. Compute the total number of annotated frames
    total_annotated_frames = np.sum(class_counts)
    
    # 3. Compute Inverse Frequency Weights
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i in range(NUM_CLASSES):
        if class_counts[i] > 0:
            # Standard inverse frequency formula
            weights[i] = total_annotated_frames / (NUM_CLASSES * class_counts[i])
        else:
            # Fallback if a class has absolutely 0 frames in the dataset
            # Setting it to 0 means the network won't penalize it if it somehow guesses it
            weights[i] = 0.0 
            
    print(f"Class Counts: {class_counts}")
    print(f"Computed Weights: {weights}")
    
    # Return as a PyTorch Tensor
    return torch.tensor(weights, dtype=torch.float32)

# previous F1 score used as pseudo difficulty
f1_scores = torch.tensor([
    0.8577, # 0: Terminal_Ileum (Easy)
    0.7468, # 1: Cecum
    0.6207, # 2: Ascending_Colon
    0.2579, # 3: Hepatic_Flexure (Hard)
    0.6506, # 4: Transverse_Colon
    0.0219, # 5: Splenic_Flexure (Very Hard)
    0.4189, # 6: Descending_Colon
    0.5832, # 7: Sigmoid_Colon
    0.4574, # 8: Rectum
    0.7186  # 9: Anal_Canal (Easy)
])

f1_based_weights = torch.tensor([
    0.3050,  # 0: Terminal_Ileum (F1: 0.8577) -> Lowest weight
    0.5426,  # 1: Cecum
    0.8129,  # 2: Ascending_Colon
    1.5903,  # 3: Hepatic_Flexure (F1: 0.2579) -> High weight
    0.7488,  # 4: Transverse_Colon
    2.0961,  # 5: Splenic_Flexure (F1: 0.0219) -> Highest weight
    1.2453,  # 6: Descending_Colon
    0.8932,  # 7: Sigmoid_Colon
    1.1628,  # 8: Rectum
    0.6030   # 9: Anal_Canal
], dtype=torch.float32)
from utils import FocalLoss
def main():
    train_dataset = MedicalStreamingDataset(
        "/scratch/lt200353-pcllm/location/cas_colon/updated_train_split.csv", 
        "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3", 
        1, 
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
        "/scratch/lt200353-pcllm/location/cas_colon/updated_test_split.csv", 
        "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3", 
        1, 
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_feature_dim = 1024
    num_action_classes = len(CLASS_MAP)
    weight_tensor = compute_class_weights("/scratch/lt200353-pcllm/location/cas_colon/updated_train_split.csv").to(device)
    loss_fn=torch.nn.CrossEntropyLoss(weight=f1_based_weights.to(device), ignore_index=-100)
    #loss_fn=torch.nn.CrossEntropyLoss(weight=weight_tensor.to(device), ignore_index=-100)
    #loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-100)
    #print(weight_tensor)
    #class_weights = (1.0 - f1_scores) + 0.05
    #loss_fn = FocalLoss(weight=class_weights.to(device), gamma=2.0, ignore_index=-100)
    config = MambaTemporalConfig(d_model=1024, n_layer=8)
    model = MambaTemporalSegmentation(
        config=config, 
        vision_dim=vision_feature_dim, 
        num_classes=num_action_classes, 
        device=device, 
        loss_fn=loss_fn
        # dtype=torch.bfloat16 # bfloat16 is highly recommended for Mamba training
    )
    # --- Training Configuration ---
    epochs = 50
    patience = 12  # How many epochs to wait for improvement before stopping
    patience_counter = 0
    best_val_loss = float('inf')
    save_dir = "./checkpoints/base_shuffle_focal"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_mamba_model_4096_60fps.pth")

    # Optimizer & Scheduler
    # AdamW is highly recommended for SSMs/Transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Reduce learning rate by half if validation loss stops improving for 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # --- Main Training Loop ---
    IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # 1. Train
        train_dataset.set_epoch(epoch)
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=1)
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
