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
from dataclasses import dataclass, field
# --- IMPORTS FROM YOUR FILES ---
# Assuming your model files are in the path or same directory
from model.CMamba import MambaTemporalSegmentation, detach_states, apply_reset_mask
from model.ContextMamba import ContextMambaForThumos, ContextMambaCmeRT, ContextMambav2
from metrics import perframe_average_precision
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

class_names = [
      "Background",
      "BaseballPitch",
      "BasketballDunk",
      "Billiards",
      "CleanAndJerk",
      "CliffDiving",
      "CricketBowling",
      "CricketShot",
      "Diving",
      "FrisbeeCatch",
      "GolfSwing",
      "HammerThrow",
      "HighJump",
      "JavelinThrow",
      "LongJump",
      "PoleVault",
      "Shotput",
      "SoccerPenalty",
      "TennisSwing",
      "ThrowDiscus",
      "VolleyballSpiking",
      "Ambiguous"
    ]

from weight_cal import weights

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

def safe_ce_loss(logits, targets, criterion, ignore_index=-100):
    if (targets != ignore_index).sum() == 0:
        return logits.sum() * 0.0 
    return criterion(logits, targets)

# --- TRAINING LOOP ---
def train_one_epoch(model, dataloader, optimizer, device, accumulation_steps=16, 
                    lambda_smooth=0.5, lambda_jump=0.0, with_future=True, ignore_index=-100):
    model.train()
    total_loss = 0.0
    steps = 0
    worker_states = {}
    
    # Standard CE for Thumos (You can calculate class weights if needed)
    print("ignore index", ignore_index)
    #criterion = nn.CrossEntropyLoss(torch.tensor(weights, dtype=torch.float32, device=device), ignore_index=ignore_index)
    transition_penalty_loss = TransitionPenaltyLoss(num_classes=model.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
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
        
        if with_future:

            # --- Multi-Objective CrossEntropy Losses ---
            loss_wo = safe_ce_loss(logits_wo_future.view(-1, model.num_classes), labels.view(-1), criterion, ignore_index=ignore_index)
            loss_w  = safe_ce_loss(logits_w_future.view(-1, model.num_classes), labels.view(-1), criterion, ignore_index=ignore_index) #criterion(logits_w_future.view(-1, model.num_classes), labels.view(-1))

            if isinstance(model, ContextMambaCmeRT):
                # only use the last one
                loss_future = safe_ce_loss(future_logits.view(-1, model.num_classes), future_labels[:, -1, :].view(-1), criterion, ignore_index=ignore_index)
            else:
                loss_future = safe_ce_loss(future_logits.view(-1, model.num_classes), future_labels.view(-1), criterion, ignore_index=ignore_index)

            ce_loss = (0.15*loss_wo + loss_w + 0.4*loss_future)

            # --- Custom Temporal Constraints ---
            # Apply constraints to the final, most refined predictions (logits_w_future)
            smooth_loss = compute_temporal_smoothing_loss(logits_w_future, labels)
            jump_loss = transition_penalty_loss(logits_w_future, labels)
        else:
            # If no future, just ignore future and with_future logits.
            # Optimize using only logits_wo_future
            ce_loss = safe_ce_loss(logits_wo_future.view(-1, model.num_classes), labels.view(-1), criterion, ignore_index=ignore_index)
            #loss_wo = ce_loss
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

        if step % 10 == 0:
            print(f"  [Train] Step {step} | Total Loss: {loss.item() * accumulation_steps:.4f} "
                  f"(CE: {ce_loss.item():.4f}, Smooth: {smooth_loss.item():.4f}, Jump: {jump_loss.item():.4f})")
            
    return total_loss / (steps if steps > 0 else 1)

def thumos_postprocessing(ground_truth, prediction, smooth=False, switch=False):
    """
    We follow (Shou et al., 2017) and adopt their perframe postprocessing method on THUMOS'14 datset.
    Source: https://bitbucket.org/columbiadvmm/cdc/src/master/THUMOS14/eval/PreFrameLabeling/compute_framelevel_mAP.m
    """

    # Simple temporal smoothing via NMS of 5-frames window
    if smooth:
        prob = np.copy(prediction)
        prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
        prob2 = np.append(prob[0, :].reshape(1, -1), prob[0: -1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob4 = np.append(prob[0: 2, :], prob[0: -2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
        prediction = np.copy(probsmooth)

    # Assign cliff diving (5) as diving (8)
    if switch:
        switch_index = np.where(prediction[:, 5] > prediction[:, 8])[0]
        prediction[switch_index, 8] = prediction[switch_index, 5]

    # Remove ambiguous (21)
    valid_index = np.where(ground_truth[:, 21] != 1)[0]

    return ground_truth[valid_index], prediction[valid_index]
# --- VALIDATION WITH MAP ---
@torch.no_grad()
def validate_map(model, dataloader, device, num_classes, with_future=True, ignore_index=-100):
    """
    Computes Frame-level mAP (Mean Average Precision).
    This is a strong proxy for the official Thumos Instance-mAP during training."""
    class_names = [
      "Background",
      "BaseballPitch",
      "BasketballDunk",
      "Billiards",
      "CleanAndJerk",
      "CliffDiving",
      "CricketBowling",
      "CricketShot",
      "Diving",
      "FrisbeeCatch",
      "GolfSwing",
      "HammerThrow",
      "HighJump",
      "JavelinThrow",
      "LongJump",
      "PoleVault",
      "Shotput",
      "SoccerPenalty",
      "TennisSwing",
      "ThrowDiscus",
      "VolleyballSpiking",
      "Ambiguous"
    ]


    model.eval()
    worker_states = {}
    print("ignore_index", ignore_index) 
    # Store probability scores and one-hot labels for mAP calculation
    all_scores = []
    all_targets = []
    
    total_loss = 0.0
    steps = 0
    #criterion = nn.CrossEntropyLoss(torch.tensor(weights, dtype=torch.float32, device=device), ignore_index=ignore_index)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
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
        logits_wo_future, _, logits_w_future, next_states = model(
            vision_embeddings=vision_embeddings, 
            contexts=valid_contexts,
            pass_states=current_states,
            labels=labels 
        )
        if with_future:
            loss = safe_ce_loss(logits_w_future.view(-1, num_classes), labels.view(-1), criterion, ignore_index=ignore_index)
        else:
            loss = safe_ce_loss(logits_wo_future.view(-1, num_classes), labels.view(-1), criterion, ignore_index=ignore_index)
        total_loss += loss.item()
        steps += 1
        worker_states[w_id] = detach_states(next_states)

        # --- DATA GATHERING FOR MAP ---
        if with_future:
            probs = F.softmax(logits_w_future, dim=-1) # [B, M, C]
        else:
            probs = F.softmax(logits_wo_future, dim=-1)
        
        # Flatten
        probs_flat = probs.view(-1, num_classes).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        
        # Filter padding
        valid_mask = (labels_flat != ignore_index) & (labels_flat < num_classes) & (labels_flat >= 0)
        
        if valid_mask.sum() > 0:
            valid_probs = probs_flat[valid_mask]
            valid_labels = labels_flat[valid_mask]
            
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
    
    ground_truth = np.concatenate(all_targets, axis=0)
    prediction = np.concatenate(all_scores, axis=0)
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # 4. Call the SOTA Metric Function
    # We pass ignore_index=0 to ignore "Background" in the mAP averaging (standard Thumos protocol).
    # Note: We do NOT pass 21 here, because we already filtered 21 out of the rows above.
    results = perframe_average_precision(
        ground_truth=ground_truth,
        prediction=prediction,
        class_names=class_names,
        ignore_index=ignore_index,  # Ignore Background class column
        metrics='AP',
        postprocessing=thumos_postprocessing 
    )
    
    return total_loss / steps, results['mean_AP']

# --- UTILS ---
def set_seed(seed=42):
    print("SEED", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#SEED = random.randint(0, 9876)
SEED = 42
# --- MAIN ---
def main():
    freeze = False
    set_seed(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    
    # 1. Load JSON (contains session definitions)
    with open("thumos.json", 'r') as f:
        json_data = json.load(f)

    # 2. Datasets (Using the Fixed ThumosStreamingDataset)
    train_dataset = ThumosStreamingDataset(
        data_root=json_data.get("data_root", ""),
        json_data=json_data,
        batch_size_per_worker=1,
        
        chunk_size=240,
        fps=4.0, target_fps=4.0,
        
        # Matching Medical Logic
        use_memory_bank=True,
        context_seconds=600,
        context_fps=4,
        num_future=12,
        future_step=1, # future_fps=target_fps
        
        phase='train', shuffle=True, temporal_jitter=True
    )

    val_dataset = ThumosStreamingDataset(
        data_root=json_data.get("data_root", ""),
        json_data=json_data,
        batch_size_per_worker=1, # Val usually safer with 1
        
        chunk_size=240,
        fps=4.0, target_fps=4.0,
        
        use_memory_bank=True,
        context_seconds=600,
        context_fps=4,
        num_future=12,
        future_step=1, # future_fps=target_fps
        
        phase='test', shuffle=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Model Setup
    # Thumos features (concatenated RGB+Flow) are usually 2048 dims (1024+1024)
    # Check your feature files! If purely RGB, change to 1024.
    VISION_DIM = train_dataset._detect_feature_dim() 
    d_model = VISION_DIM
    
    config = MambaTemporalConfig(d_model=d_model, n_layer=10)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=json_data.get('ignore_index', -100))
    
    model = MambaTemporalSegmentation(
        config=config, 
        vision_dim=d_model, 
        num_classes=THUMOS_CLASSES, 
        device=device, 
        loss_fn=loss_fn
    )
    
    full_model = ContextMambav2(
        base_model=model.backbone, 
        vision_dim=VISION_DIM,
        d_model=d_model, 
        num_classes=THUMOS_CLASSES,
        use_multihead=True,
        target_fps=4.0,
        context_fps=4.0,
        query_fps=4.0,
        num_future=12,
        future_fps=4.0,
    ).to(device)

    print(full_model)
    base_d_state = full_model.base_model.layers[0].mixer.d_state
    print(f"Base Model d_state: {base_d_state}")

    # checkpoint_path = "/scratch/lt200353-pcllm/long_future_joint_thumos_mamba_0.6988.pth"
    # #print(f"Loading weights from {checkpoint_path}...")

    # # Load directly into full_model since this checkpoint includes the wrapper and backbone
    # state_dict = torch.load(checkpoint_path, map_location=device)
    # full_model.load_state_dict(state_dict)

    # if freeze==True:
    #     for param in full_model.base_model.parameters():
    #         param.requires_grad = False

    # 4. Training
    optimizer = torch.optim.AdamW(full_model.parameters(), lr=1.0e-4, weight_decay=1e-4)
    WARMUP_EPOCHS = 10 
    MAX_EPOCHS = 15 # Must match your training loop epochs
    
    # Phase 1: Linear Warmup (start at 1% of lr, go to 100% over 5 epochs)
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    
    # Phase 2: Cosine Decay (decay from 100% to eta_min over remaining epochs)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(MAX_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6
    )
    
    # Combine them sequentially
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[WARMUP_EPOCHS]
    )
    # ---------------------------

    epochs = MAX_EPOCHS

    best_map = 0.0
    
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)
    val_loss, val_map = validate_map(full_model, val_loader, device, THUMOS_CLASSES, with_future=True, ignore_index=json_data.get('ignore_index', -100))
    print(val_map)
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Train
        train_dataset.set_epoch(epoch)
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=16, worker_init_fn=seed_worker, generator=g)
        train_loss = train_one_epoch(full_model, train_loader, optimizer, device, with_future=True, ignore_index=json_data.get('ignore_index', -100), lambda_smooth=0.0, accumulation_steps=32)
        
        # Validate (mAP)
        val_loss, val_map = validate_map(full_model, val_loader, device, THUMOS_CLASSES, with_future=True, ignore_index=json_data.get('ignore_index', -100))
        
        print(f"Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val mAP:    {val_map:.4f}")
        
        # Scheduler Step (Monitor mAP, so 'max')
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"  Current LR: {current_lr:.6f}")
        
        if val_map > best_map:
            best_map = val_map
            print(f"New Best mAP! Saving...")
            torch.save(full_model.state_dict(), f"/scratch/lt200353-pcllm/small_epoch_big_long_mem_jilter_jont_mamba_{val_map:.4f}.pth")

if __name__ == "__main__":
    main()
