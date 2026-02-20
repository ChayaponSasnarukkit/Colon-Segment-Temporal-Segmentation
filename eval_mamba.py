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

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

def plot_temporal_diagram(preds, labels, video_idx, save_dir, idx_to_class, model_fps=30):
    """
    Creates a temporal 'barcode' plot comparing Ground Truth vs. Prediction.
    Downsamples the sequence to 1 FPS and includes a class legend.
    """
    # 1. Downsample from model FPS (e.g., 30) to 1 FPS
    # By slicing every 30th frame, we align with the 1 FPS annotations
    preds_1fps = preds[::model_fps]
    labels_1fps = labels[::model_fps]
    
    num_classes = len(idx_to_class)
    
    # 2. Map ignore_index (-100) to a new index (num_classes) for plotting purposes
    # This prevents the timeline from squishing while handling unlabeled gaps
    labels_mapped = np.where(labels_1fps == -100, num_classes, labels_1fps)
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 4), sharex=True)
    
    # 3. Create a custom colormap
    # Use tab20 for the actual classes, and add light grey for the 'Unlabeled' (-100) areas
    base_cmap = plt.get_cmap('tab20', num_classes)
    colors = [base_cmap(i) for i in range(num_classes)]
    colors.append((0.9, 0.9, 0.9, 1.0))  # Light Grey for unlabeled/ignore_index
    cmap = mcolors.ListedColormap(colors)
    
    # 4. Plot Ground Truth (Top)
    ax1.imshow(labels_mapped[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=num_classes)
    ax1.set_title(f'Video {video_idx}: Ground Truth (1 FPS)')
    ax1.set_yticks([])
    
    # 5. Plot Predictions (Bottom)
    # Model predictions won't have -100, so they will map normally to 0 -> num_classes-1
    ax2.imshow(preds_1fps[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=num_classes)
    ax2.set_title('Prediction (1 FPS)')
    ax2.set_yticks([])
    ax2.set_xlabel('Time (Seconds / Frames at 1 FPS)')
    
    # 6. Create the Legend
    patches = [mpatches.Patch(color=colors[i], label=idx_to_class[i]) for i in range(num_classes)]
    patches.append(mpatches.Patch(color=colors[num_classes], label='Unlabeled / Ignored'))
    
    # Place legend outside the plot to the right
    fig.legend(handles=patches, loc='center right', bbox_to_anchor=(1.18, 0.5), fontsize=10)
    
    # 7. Save cleanly
    # bbox_inches='tight' ensures the external legend isn't cut off in the saved image
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'temporal_video_{video_idx}.png'), dpi=150, bbox_inches='tight')
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

    # 1. Dictionaries to handle Multi-Worker & Multi-Batch streams
    worker_states = {}
    stream_buffers = {}

    all_preds_global = []
    all_labels_global = []
    video_count = 0

    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting evaluation (Saving results to {save_dir})...")

    with torch.no_grad():
        for step, (vision_embeddings, _, labels, reset_mask, _, worker_id) in enumerate(tqdm(dataloader)):
            vision_embeddings = vision_embeddings.to(device)
            labels = labels.to(device)
            reset_mask = reset_mask.to(device)

            # Identify which worker process yielded this batch
            w_id = int(worker_id[0].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)

            # Fetch the isolated Mamba memory for this specific worker
            current_states = worker_states.get(w_id, None)

            # Reset states for streams where a new video is starting
            if reset_mask.any() and current_states is not None:
                current_states = apply_reset_mask(current_states, reset_mask)

            # Forward pass
            outputs = model(
                vision_embeddings=vision_embeddings,
                pass_states=current_states,
                labels=labels
            )

            # Update the worker's state memory
            worker_states[w_id] = detach_states(outputs.next_states)

            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            batch_size = preds.shape[0]

            # 2. Untangle the streams and reconstruct the videos
            for b in range(batch_size):
                stream_key = (w_id, b)

                # Initialize the buffer if this stream hasn't been seen yet
                if stream_key not in stream_buffers:
                    stream_buffers[stream_key] = {'preds': [], 'labels': []}

                # If a new video started on this specific stream AND we have old data, plot the old video
                if reset_mask[b].item() and len(stream_buffers[stream_key]['preds']) > 0:
                    vid_preds = np.concatenate(stream_buffers[stream_key]['preds'])
                    vid_labels = np.concatenate(stream_buffers[stream_key]['labels'])

                    # 3. Generate Temporal Chart for this completed video
                    plot_temporal_diagram(vid_preds, vid_labels, video_count, save_dir, idx_to_class)
                    video_count += 1

                    # Accumulate to global metrics (filtering ignore_index)
                    valid_mask = vid_labels != ignore_index
                    all_preds_global.extend(vid_preds[valid_mask])
                    all_labels_global.extend(vid_labels[valid_mask])

                    # Clear the buffer for the new video
                    stream_buffers[stream_key] = {'preds': [], 'labels': []}

                # Append the current chunk to this stream's ongoing video
                stream_buffers[stream_key]['preds'].append(preds[b].cpu().numpy().flatten())
                stream_buffers[stream_key]['labels'].append(labels[b].cpu().numpy().flatten())

    # 4. Flush any remaining videos left in the buffers after the dataloader finishes
    print("Flushing final buffers...")
    for stream_key, buffer in stream_buffers.items():
        if len(buffer['preds']) > 0:
            vid_preds = np.concatenate(buffer['preds'])
            vid_labels = np.concatenate(buffer['labels'])

            plot_temporal_diagram(vid_preds, vid_labels, video_count, save_dir, idx_to_class)
            video_count += 1

            valid_mask = vid_labels != ignore_index
            all_preds_global.extend(vid_preds[valid_mask])
            all_labels_global.extend(vid_labels[valid_mask])

    print(f"Generated temporal diagrams for {video_count} videos.")

    # 5. Generate the Global Confusion Matrix
    plot_confusion_matrix(all_labels_global, all_preds_global, save_dir, idx_to_class)
    print("Generated normalized confusion matrix.")

    # 6. Calculate and print final global metrics
    acc = accuracy_score(all_labels_global, all_preds_global)
    f1_macro = f1_score(all_labels_global, all_preds_global, average='macro', zero_division=0)

    print("\n--- Final Evaluation Metrics ---")
    print(f"Total Valid Frames Evaluated: {len(all_labels_global)}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1_macro:.4f}")

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
