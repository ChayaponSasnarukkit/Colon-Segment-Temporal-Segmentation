import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your existing modules here
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

def plot_temporal_diagram(preds, labels, video_idx, save_dir, idx_to_class):
    """
    Creates a temporal 'barcode' plot comparing Ground Truth vs. Prediction.
    """
    # Filter out ignored indices (e.g., -100) for the plot
    valid_indices = labels != -100
    if not np.any(valid_indices):
        return  # Skip if the whole chunk is ignored
    
    clean_labels = labels[valid_indices]
    clean_preds = preds[valid_indices]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 4), sharex=True)
    
    # Create colormap based on number of classes
    num_classes = len(idx_to_class)
    cmap = plt.get_cmap('tab20', num_classes)
    
    # Plot Ground Truth
    ax1.imshow(clean_labels[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=num_classes-1)
    ax1.set_title(f'Video {video_idx}: Ground Truth')
    ax1.set_yticks([])
    
    # Plot Predictions
    ax2.imshow(clean_preds[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=num_classes-1)
    ax2.set_title('Prediction')
    ax2.set_yticks([])
    ax2.set_xlabel('Time (frames)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'temporal_video_{video_idx}.png'), dpi=150)
    plt.close()

def plot_confusion_matrix(all_labels, all_preds, save_dir, idx_to_class):
    """
    Generates and saves a normalized confusion matrix.
    """
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(idx_to_class)))
    
    # Normalize by row (ground truth)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) # Handle division by zero
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def evaluate_and_visualize(model, dataloader, device, save_dir, idx_to_class, ignore_index=-100):
    model.eval()
    current_states = None
    
    all_preds_global = []
    all_labels_global = []
    
    # To track per-video metrics
    current_video_preds = []
    current_video_labels = []
    video_count = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("Starting evaluation...")
    with torch.no_grad():
        for step, (vision_embeddings, _, labels, reset_mask, _, _) in enumerate(tqdm(dataloader)):
            vision_embeddings = vision_embeddings.to(device)
            labels = labels.to(device)
            reset_mask = reset_mask.to(device)
            
            # If a new video starts, plot the previous one and reset accumulators
            if reset_mask.any() and len(current_video_preds) > 0:
                plot_temporal_diagram(
                    np.concatenate(current_video_preds), 
                    np.concatenate(current_video_labels), 
                    video_count, 
                    save_dir, 
                    idx_to_class
                )
                video_count += 1
                current_video_preds = []
                current_video_labels = []

            if reset_mask.any() and current_states is not None:
                current_states = apply_reset_mask(current_states, reset_mask)

            outputs = model(
                vision_embeddings=vision_embeddings,
                pass_states=current_states,
                labels=labels
            )
            
            logits = outputs.logits 
            preds = torch.argmax(logits, dim=-1)
            
            # Flatten to 1D arrays
            preds_flat = preds.cpu().numpy().flatten()
            labels_flat = labels.cpu().numpy().flatten()
            
            # Accumulate for the current video (keeping ignored indices for temporal alignment, 
            # we filter them out inside the plotting function)
            current_video_preds.append(preds_flat)
            current_video_labels.append(labels_flat)
            
            # Accumulate for global metrics (filtering out ignored indices immediately)
            valid_mask = labels_flat != ignore_index
            all_preds_global.extend(preds_flat[valid_mask])
            all_labels_global.extend(labels_flat[valid_mask])

            current_states = detach_states(outputs.next_states)
            
    # Plot the very last video after the loop finishes
    if len(current_video_preds) > 0:
        plot_temporal_diagram(
            np.concatenate(current_video_preds), 
            np.concatenate(current_video_labels), 
            video_count, 
            save_dir, 
            idx_to_class
        )
        
    print(f"Generated temporal diagrams for {video_count + 1} videos.")
    
    # Plot Global Confusion Matrix
    plot_confusion_matrix(all_labels_global, all_preds_global, save_dir, idx_to_class)
    print("Generated confusion matrix.")
    
    # Calculate and print final metrics
    acc = accuracy_score(all_labels_global, all_preds_global)
    f1_macro = f1_score(all_labels_global, all_preds_global, average='macro', zero_division=0)
    print(f"\nFinal Test Accuracy: {acc:.4f}")
    print(f"Final Test Macro F1: {f1_macro:.4f}")


def main_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Reverse CLASS_MAP for visualization labels
    IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
    num_action_classes = len(CLASS_MAP)
    
    # 2. Setup Dataset & DataLoader (Identical to your test set)
    test_dataset = MedicalStreamingDataset(
        "/scratch/lt200353-pcllm/location/cas_colon/updated_test_split.csv", 
        "/scratch/lt200353-pcllm/location/cas_colon/features_dinov3", 
        2, 
        chunk_size=2048, 
        fps=60, 
        target_fps=30, 
        use_memory_bank=False,
        context_seconds=300, 
        context_fps=1,
        shuffle=False,  # CRITICAL: Must be False for valid temporal evaluation
        use_emb=True,
        emb_dim=1024,
        transform=None
    )
    test_loader = DataLoader(test_dataset, batch_size=None, num_workers=2)
    
    # 3. Setup Model Architecture
    config = MambaTemporalConfig(d_model=1024, n_layer=8)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) # Loss weighting doesn't matter for pure inference
    
    model = MambaTemporalSegmentation(
        config=config, 
        vision_dim=1024, 
        num_classes=num_action_classes, 
        device=device, 
        loss_fn=loss_fn
    )
    
    # 4. Load the Model Weights
    checkpoint_path = "./checkpoints/base_shuffle/best_mamba_model.pth"
    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    # 5. Run Evaluation
    save_directory = "./evaluation_results"
    evaluate_and_visualize(
        model=model, 
        dataloader=test_loader, 
        device=device, 
        save_dir=save_directory, 
        idx_to_class=IDX_TO_CLASS
    )

if __name__ == "__main__":
    main_eval()
