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
from model.CMamba import MambaTemporalSegmentation, detach_states, apply_reset_mask
from model.ContextMamba import ContextMambaForThumos, ContextMambaCmeRT, ContextMambav2
from metrics import perframe_average_precision

# -> CRITICAL UPDATE: Import the new Dataset and Sampler
from dataset.thumos import ThumosStatefulDataset, StatefulStreamingSampler 

# --- CONFIG ---
THUMOS_CLASSES = 22 
SEED = 42
VIRTUAL_BATCH_SIZE_TRAIN = 8  # How many videos to process in parallel during training
VIRTUAL_BATCH_SIZE_VAL = 1    # Safer to process validation videos 1-by-1 sequentially

@dataclass
class MambaTemporalConfig:
    d_model: int = 2048           
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
      "Background", "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk",
      "CliffDiving", "CricketBowling", "CricketShot", "Diving", "FrisbeeCatch",
      "GolfSwing", "HammerThrow", "HighJump", "JavelinThrow", "LongJump",
      "PoleVault", "Shotput", "SoccerPenalty", "TennisSwing", "ThrowDiscus",
      "VolleyballSpiking", "Ambiguous"
]

# --- LOSS FUNCTIONS ---
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
def train_one_epoch(model, dataloader, optimizer, device, virtual_batch_size, accumulation_steps=16, 
                    lambda_smooth=0.5, lambda_jump=0.0, with_future=True, ignore_index=-100):
    model.train()
    total_loss = 0.0
    steps = 0
    
    # --- SLOT STATE TRACKING ---
    slot_states = [None] * virtual_batch_size 
    
    transition_penalty_loss = TransitionPenaltyLoss(num_classes=model.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer.zero_grad() 

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        vision_embeddings, contexts, labels, future_labels, reset_masks, context_masks, session_names = batch
        
        batch_loss = 0.0
        num_of_valid_grad_in_vbatch = 0 

        # --- LOOP THROUGH VIRTUAL BATCH ONE-BY-ONE ---
        for i in range(virtual_batch_size):
            v_emb = vision_embeddings[i].unsqueeze(0).to(device)      
            lbls = labels[i].unsqueeze(0).to(device)                  
            f_lbls = future_labels[i].unsqueeze(0).to(device)         
            
            if reset_masks[i].item():
                slot_states[i] = None 
                
            if (lbls == ignore_index).all() and (f_lbls == ignore_index).all():
                continue

            valid_ctx = None
            if contexts is not None:
                ctx = contexts[i].unsqueeze(0).to(device)
                c_mask = context_masks[i].unsqueeze(0).to(device)
                
                actual_K = c_mask.sum().int().item()
                if actual_K > 0:
                    valid_ctx = ctx[:, :actual_K, :]
                else:
                    valid_ctx = torch.zeros((1, 1, ctx.shape[2]), device=device)

            logits_wo, f_logits, logits_w, next_state = model(
                vision_embeddings=v_emb, 
                contexts=valid_ctx,
                pass_states=slot_states[i],  
                labels=lbls 
            )
            
            slot_states[i] = detach_states(next_state)
            
            if with_future:
                loss_wo = safe_ce_loss(logits_wo.view(-1, model.num_classes), lbls.view(-1), criterion, ignore_index=ignore_index)
                loss_w  = safe_ce_loss(logits_w.view(-1, model.num_classes), lbls.view(-1), criterion, ignore_index=ignore_index)
                
                if isinstance(model, ContextMambaCmeRT):
                    loss_future = safe_ce_loss(f_logits.view(-1, model.num_classes), f_lbls[:, -1, :].view(-1), criterion, ignore_index=ignore_index)
                else:
                    loss_future = safe_ce_loss(f_logits.view(-1, model.num_classes), f_lbls.view(-1), criterion, ignore_index=ignore_index)

                ce_loss = (0.15 * loss_wo) + loss_w + (0.4 * loss_future)
                smooth_loss = compute_temporal_smoothing_loss(logits_w, lbls)
                jump_loss = transition_penalty_loss(logits_w, lbls)
            else:
                ce_loss = safe_ce_loss(logits_wo.view(-1, model.num_classes), lbls.view(-1), criterion, ignore_index=ignore_index)
                smooth_loss = compute_temporal_smoothing_loss(logits_w, lbls)
                jump_loss = transition_penalty_loss(logits_w, lbls)

            item_loss = ce_loss + (lambda_smooth * smooth_loss) + (lambda_jump * jump_loss)
            
            batch_loss += item_loss
            num_of_valid_grad_in_vbatch += 1

        if num_of_valid_grad_in_vbatch > 0:
            batch_loss = (batch_loss / num_of_valid_grad_in_vbatch) / accumulation_steps
            batch_loss.backward()
            
            total_loss += (batch_loss.item() * accumulation_steps)
            steps += 1
        
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        if step % 10 == 0 and num_of_valid_grad_in_vbatch > 0:
            print(f"  [Train] Step {step} | Total Loss: {batch_loss.item() * accumulation_steps:.4f}")
            
    return total_loss / (steps if steps > 0 else 1)

def thumos_postprocessing(ground_truth, prediction, smooth=False, switch=False):
    if smooth:
        prob = np.copy(prediction)
        prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
        prob2 = np.append(prob[0, :].reshape(1, -1), prob[0: -1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob4 = np.append(prob[0: 2, :], prob[0: -2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
        prediction = np.copy(probsmooth)

    if switch:
        switch_index = np.where(prediction[:, 5] > prediction[:, 8])[0]
        prediction[switch_index, 8] = prediction[switch_index, 5]

    valid_index = np.where(ground_truth[:, 21] != 1)[0]
    return ground_truth[valid_index], prediction[valid_index]

# --- VALIDATION LOOP (UPDATED FOR BATCH SAMPLER) ---
@torch.no_grad()
def validate_map(model, dataloader, device, num_classes, virtual_batch_size, with_future=True, ignore_index=-100):
    model.eval()
    
    # Slot tracker mirrors the training loop
    slot_states = [None] * virtual_batch_size 
    
    all_scores = []
    all_targets = []
    total_loss = 0.0
    steps = 0
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    for step, batch in enumerate(tqdm(dataloader, desc="Validating (mAP)")):
        vision_embeddings, contexts, labels, future_labels, reset_masks, context_masks, session_names = batch
        
        # Loop through virtual batch items
        for i in range(virtual_batch_size):
            v_emb = vision_embeddings[i].unsqueeze(0).to(device)
            lbls = labels[i].unsqueeze(0).to(device)
            
            if reset_masks[i].item():
                slot_states[i] = None
                
            if (lbls == ignore_index).all():
                continue

            valid_ctx = None
            if contexts is not None:
                ctx = contexts[i].unsqueeze(0).to(device)
                c_mask = context_masks[i].unsqueeze(0).to(device)
                actual_K = c_mask.sum().int().item()
                if actual_K > 0:
                    valid_ctx = ctx[:, :actual_K, :]
                else:
                    valid_ctx = torch.zeros((1, 1, ctx.shape[2]), device=device)

            logits_wo_future, _, logits_w_future, next_state = model(
                vision_embeddings=v_emb, 
                contexts=valid_ctx,
                pass_states=slot_states[i],
                labels=lbls 
            )
            
            slot_states[i] = next_state
            
            if with_future:
                loss = safe_ce_loss(logits_w_future.view(-1, num_classes), lbls.view(-1), criterion, ignore_index=ignore_index)
                probs = F.softmax(logits_w_future, dim=-1)
            else:
                loss = safe_ce_loss(logits_wo_future.view(-1, num_classes), lbls.view(-1), criterion, ignore_index=ignore_index)
                probs = F.softmax(logits_wo_future, dim=-1)
                
            total_loss += loss.item()
            steps += 1

            # Flatten and filter for mAP
            probs_flat = probs.view(-1, num_classes).cpu().numpy()
            labels_flat = lbls.view(-1).cpu().numpy()
            
            valid_mask = (labels_flat != ignore_index) & (labels_flat < num_classes) & (labels_flat >= 0)
            
            if valid_mask.sum() > 0:
                valid_probs = probs_flat[valid_mask]
                valid_labels = labels_flat[valid_mask]
                
                targets_one_hot = np.zeros((valid_labels.size, num_classes))
                targets_one_hot[np.arange(valid_labels.size), valid_labels] = 1
                
                all_scores.append(valid_probs)
                all_targets.append(targets_one_hot)

    if len(all_scores) == 0: return 0.0, 0.0
    
    ground_truth = np.concatenate(all_targets, axis=0)
    prediction = np.concatenate(all_scores, axis=0)
    global class_names
    
    results = perframe_average_precision(
        ground_truth=ground_truth,
        prediction=prediction,
        class_names=class_names,
        ignore_index=ignore_index,  
        metrics='AP',
        postprocessing=thumos_postprocessing 
    )
    
    return total_loss / steps, results['mean_AP']


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
    
    with open("thumos.json", 'r') as f:
        json_data = json.load(f)

    # 1. Initialize Map-Style Datasets (Removed Streaming parameters)
    train_dataset = ThumosStatefulDataset(
        data_root=json_data.get("data_root", ""),
        json_data=json_data,
        chunk_size=240,
        fps=4.0, target_fps=4.0,
        use_memory_bank=True,
        context_seconds=240,
        context_fps=4,
        num_future=4,
        future_step=1, 
        phase='train'
    )

    val_dataset = ThumosStatefulDataset(
        data_root=json_data.get("data_root", ""),
        json_data=json_data,
        chunk_size=240,
        fps=4.0, target_fps=4.0,
        use_memory_bank=True,
        context_seconds=240,
        context_fps=4,
        num_future=4,
        future_step=1, 
        phase='test'
    )
    
    # 2. Initialize the Global Orchestrator Samplers
    train_sampler = StatefulStreamingSampler(
        sessions=train_dataset.sessions,
        video_lengths=train_dataset.video_lengths,
        virtual_batch_size=VIRTUAL_BATCH_SIZE_TRAIN,
        chunk_step=train_dataset.chunk_span,
        shuffle=True,
        seed=SEED
    )
    
    val_sampler = StatefulStreamingSampler(
        sessions=val_dataset.sessions,
        video_lengths=val_dataset.video_lengths,
        virtual_batch_size=VIRTUAL_BATCH_SIZE_VAL,
        chunk_step=val_dataset.chunk_span,
        shuffle=False, # Do not shuffle validation
        seed=SEED
    )

    # 3. Use BatchSampler in DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler, 
        num_workers=8, # Adjusted to avoid overhead, tune based on CPU
        worker_init_fn=seed_worker, 
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_sampler=val_sampler, 
        num_workers=4 
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    VISION_DIM = train_dataset.feature_dim 
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
        num_future=4,
        future_fps=4.0,
    ).to(device)

    optimizer = torch.optim.AdamW(full_model.parameters(), lr=0.75e-4, weight_decay=1e-4)
    WARMUP_EPOCHS = 10 
    MAX_EPOCHS = 30 
    
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(MAX_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[WARMUP_EPOCHS])

    epochs = MAX_EPOCHS
    best_map = 0.0
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Sync Sampler Epochs for Shuffling
        train_sampler.set_epoch(epoch)
        
        train_loss = train_one_epoch(
            full_model, train_loader, optimizer, device, 
            virtual_batch_size=VIRTUAL_BATCH_SIZE_TRAIN, 
            with_future=True, 
            ignore_index=json_data.get('ignore_index', -100), 
            lambda_smooth=0.0, 
            accumulation_steps=16 # Adjusted due to virtual batch
        )
        
        val_loss, val_map = validate_map(
            full_model, val_loader, device, THUMOS_CLASSES, 
            virtual_batch_size=VIRTUAL_BATCH_SIZE_VAL,
            with_future=True, 
            ignore_index=json_data.get('ignore_index', -100)
        )
        
        print(f"Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val mAP:    {val_map:.4f}")
        
        scheduler.step()
        print(f"  Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_map > best_map:
            best_map = val_map
            print(f"New Best mAP! Saving...")
            torch.save(full_model.state_dict(), f"/scratch/lt200353-pcllm/XL_epoch_long_mem_jilter_jont_mamba_{val_map:.4f}.pth")

if __name__ == "__main__":
    main()