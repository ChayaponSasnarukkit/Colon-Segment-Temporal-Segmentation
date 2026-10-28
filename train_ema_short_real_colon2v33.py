"""REAL-Colon online temporal segmentation: original training + weight EMA.

Runs the raw model and EMA model through the same validation set every epoch.
Prints per-class precision, recall, F1, support, and predicted frame count,
with explicit cecum precision/recall monitoring. Saves independent best
checkpoints and (optionally) cached predictions for both models.

Requirements: the project's model.CMamba, model.ContextMamba, and
             dataset.real_locationv2 modules, plus PyTorch, NumPy, tqdm.

IMPORTANT: EVAL_PHASE is kept as 'test' to match the source script. If this
is a held-out TEST fold, do not select checkpoints on it! Provide a real
validation partition and set EVAL_PHASE to that supported dataset phase.
"""

import csv
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.CMamba import MambaTemporalSegmentation
from model.ContextMamba import ContextMambav2TASver
from dataset.real_locationv2 import RealColonStreamingDataset


# -----------------------------------------------------------------------------
# Experiment settings: original values retained unless stated otherwise
# -----------------------------------------------------------------------------
SEED = 421
FOLD = 2
VIDEO_ROOT = "/project/lt200353-pcllm/3d_report_gen/real-colon/"
SPLIT_DIR = "/home/csasnaru/temporal_segmentation/data/dataset/RC_lists/5_fold/"
PRETRAIN_DIR = "/project/lt200353-pcllm/3d_report_gen/real-colon/"
PRETRAIN_FILENAME = "neww3best_mamba_model.pth"
SAVE_DIR = f"/project/pv927002-dcpcce/rerun4_4mid_smoothing_dampw_realcolon_fold{FOLD}"
RUN_NAME = "re4_32drop5expansion4stages2batch12lr03_ema"

EVAL_PHASE = "test"  # Original code; use a genuine validation split for tuning.
EPOCHS = 100
PATIENCE = EPOCHS // 2
LR = 0.3e-4
WEIGHT_DECAY = 1e-2
SCHEDULER_CHOICE = "cosine_with_warmup"  # or "reduceonplateau"
WARMUP_EPOCHS = 5
VIRTUAL_BATCH_SIZE = 12          # original number of loader workers / accumulation
FREEZE_BASE = False
WEIGHTED_LOSS = True

EMA_DECAY = 0.995
EMA_USE_BUFFERS = False           # Copy model buffers instead of averaging them.
EARLY_STOP_METRIC = "ema_macro_f1"  # EMA F1 is primary; raw checkpoints independent.
CACHE_BEST_PREDICTIONS = True
SAVE_ALL_EPOCHS = False          # Original code saved every epoch; optional here.
GRAD_CLIP_NORM: Optional[float] = None  # None keeps original optimizer behavior.

TRAIN_SMOOTH_WEIGHT = 0.0        # Original main() passed lambda_smooth=0
VAL_SMOOTH_WEIGHT = 0.0

# Preserve this EXACT spelling: source dataset may encode labels as "ceacum".
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
DISPLAY_NAMES = {idx: ("cecum" if name == "ceacum" else name)
                 for name, idx in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)
CECUM_ID = CLASS_MAP["ceacum"]
IGNORE_INDEX = -100


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


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # These settings do not make Mamba's custom CUDA scan deterministic.


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------------------------------------------------------
# Recurrent state helpers: preserve original base_states + tcn_caches structure
# -----------------------------------------------------------------------------
def detach_states(states):
    if states is None:
        return None
    if isinstance(states, dict):
        result = {}
        base = states.get("base_states")
        result["base_states"] = (
            [(s.detach().clone() if s is not None else None,
              c.detach().clone() if c is not None else None) for s, c in base]
            if base is not None else None
        )
        tcn = states.get("tcn_caches")
        if isinstance(tcn, dict):
            result["tcn_caches"] = {
                stage: [c.detach() if c is not None else None for c in caches]
                for stage, caches in tcn.items()
            }
        elif isinstance(tcn, list):
            result["tcn_caches"] = [c.detach() if c is not None else None for c in tcn]
        else:
            result["tcn_caches"] = None
        return result
    return [
        (s.detach().clone() if s is not None else None,
         c.detach().clone() if c is not None else None)
        for s, c in states
    ]


def apply_reset_mask(states, reset_mask):
    if states is None:
        return None

    def masked(tensor):
        if tensor is None:
            return None
        broadcast = reset_mask.bool().reshape(-1, *([1] * (tensor.ndim - 1)))
        return tensor.masked_fill(broadcast, 0.0)

    if isinstance(states, dict):
        result = {}
        base = states.get("base_states")
        result["base_states"] = (
            [(masked(s), masked(c)) for s, c in base]
            if base is not None else None
        )
        tcn = states.get("tcn_caches")
        if isinstance(tcn, dict):
            result["tcn_caches"] = {
                stage: [masked(c) for c in caches] for stage, caches in tcn.items()
            }
        elif isinstance(tcn, list):
            result["tcn_caches"] = [masked(c) for c in tcn]
        else:
            result["tcn_caches"] = None
        return result
    return [(masked(s), masked(c)) for s, c in states]


def prepare_batch(batch, device):
    vision, contexts, labels, future_labels, reset_mask, context_masks, worker_id = batch
    vision = vision.to(device)
    contexts = contexts.to(device)
    labels = labels.to(device)
    future_labels = future_labels.to(device)
    reset_mask = reset_mask.to(device)
    context_masks = context_masks.to(device)

    # Original streaming loader assumes each sample belongs to a single worker.
    # It also assumes valid memory items occupy the first K positions.
    if labels.shape[0] != 1:
        raise ValueError("This script expects batch_size_per_worker=1 (B=1).")
    actual_k = int(context_masks[0].sum().item())
    valid_contexts = contexts[:, :actual_k, :]
    w_id = int(worker_id.reshape(-1)[0].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
    return vision, valid_contexts, labels, future_labels, reset_mask, w_id


def safe_ce_loss(logits, targets, criterion):
    if not (targets != IGNORE_INDEX).any():
        return logits.sum() * 0.0
    return criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def temporal_smoothing_loss(logits, labels):
    probs = F.softmax(logits, dim=-1)
    diffs = (probs[:, 1:, :] - probs[:, :-1, :]).square().sum(dim=-1)
    valid = (labels[:, 1:] != IGNORE_INDEX) & (labels[:, :-1] != IGNORE_INDEX)
    if not valid.any():
        return logits.sum() * 0.0
    return diffs[valid].mean()


# -----------------------------------------------------------------------------
# Training (same five CE losses and weights as the original script)
# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, device, class_weights,
                    ema_model=None, accumulation_steps=VIRTUAL_BATCH_SIZE,
                    lambda_smooth=TRAIN_SMOOTH_WEIGHT):
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    worker_states = {}
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    num_chunks = 0
    pending = 0
    optimizer_updates = 0

    def perform_update(pending_chunks):
        # The last group may have fewer than accumulation_steps chunks.
        # Renormalize once to give the remaining chunks their correct mean.
        if pending_chunks != accumulation_steps:
            scale = accumulation_steps / pending_chunks
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(model)  # AFTER optimizer.step(), not per chunk
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        vision, contexts, labels, future_labels, reset_mask, worker_id = prepare_batch(batch, device)
        previous = worker_states.get(worker_id)
        if previous is not None:
            previous = apply_reset_mask(previous, reset_mask)

        (logits_wo, future_logits, logits_w,
         tcn_tmp_logits, tcn_logits, next_states) = model(
            vision_embeddings=vision, contexts=contexts,
            pass_states=previous, labels=labels,
        )

        loss_wo = safe_ce_loss(logits_wo, labels, criterion)
        loss_w = safe_ce_loss(logits_w, labels, criterion)
        loss_future = safe_ce_loss(future_logits, future_labels, criterion)
        loss_tcn_tmp = safe_ce_loss(tcn_tmp_logits, labels, criterion)
        loss_final = safe_ce_loss(tcn_logits, labels, criterion)

        ce_loss = (0.1 * loss_wo + 0.1 * loss_w + 0.2 * loss_tcn_tmp
                   + 0.2 * loss_future + 1.0 * loss_final)
        smooth = temporal_smoothing_loss(tcn_logits, labels) if lambda_smooth else ce_loss.new_zeros(())
        loss = ce_loss + lambda_smooth * smooth
        (loss / accumulation_steps).backward()

        worker_states[worker_id] = detach_states(next_states)
        total_loss += float(loss.detach())
        num_chunks += 1
        pending += 1
        if pending == accumulation_steps:
            perform_update(pending)
            pending = 0
            optimizer_updates += 1

        if step % 50 == 0:
            print(f"  [Train] Step {step} | Total Loss: {float(loss.detach()):.4f} "
                  f"(CE: {float(ce_loss.detach()):.4f}, Smooth: {float(smooth.detach()):.4f})")

    if pending:
        perform_update(pending)
        optimizer_updates += 1

    return total_loss / max(num_chunks, 1), optimizer_updates


@dataclass
class ValResults:
    loss: float
    acc: float
    macro_f1: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    predicted: np.ndarray
    confusion: np.ndarray
    aux_losses: dict


# -----------------------------------------------------------------------------
# Validation: frame-wise metrics, with an explicit 9-class label order.
# Using a confusion matrix also avoids storing millions of frame predictions.
# -----------------------------------------------------------------------------
@torch.no_grad()
def validate(model, dataloader, device, class_weights, lambda_smooth=VAL_SMOOTH_WEIGHT,
             desc="Validating"):
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    cm = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)
    worker_states = {}  # CRITICAL: raw and EMA never share streaming states.
    losses = {name: 0.0 for name in ("final", "wo", "w", "future", "tcn_tmp", "smooth")}
    steps = 0

    for batch in tqdm(dataloader, desc=desc):
        vision, contexts, labels, future_labels, reset_mask, worker_id = prepare_batch(batch, device)
        previous = worker_states.get(worker_id)
        if previous is not None:
            previous = apply_reset_mask(previous, reset_mask)

        (logits_wo, future_logits, logits_w,
         tcn_tmp_logits, tcn_logits, next_states) = model(
            vision_embeddings=vision, contexts=contexts,
            pass_states=previous, labels=labels,
        )
        worker_states[worker_id] = detach_states(next_states)

        losses["wo"] += safe_ce_loss(logits_wo, labels, criterion).item()
        losses["w"] += safe_ce_loss(logits_w, labels, criterion).item()
        losses["future"] += safe_ce_loss(future_logits, future_labels, criterion).item()
        losses["tcn_tmp"] += safe_ce_loss(tcn_tmp_logits, labels, criterion).item()
        losses["final"] += safe_ce_loss(tcn_logits, labels, criterion).item()
        if lambda_smooth:
            losses["smooth"] += temporal_smoothing_loss(tcn_logits, labels).item()
        steps += 1

        preds = tcn_logits.argmax(dim=-1).reshape(-1).cpu().numpy().astype(np.int64)
        targets = labels.reshape(-1).cpu().numpy().astype(np.int64)
        valid = targets != IGNORE_INDEX
        targets, preds = targets[valid], preds[valid]
        if targets.size:
            if ((targets < 0) | (targets >= model.num_classes)).any():
                raise ValueError("Unexpected validation label outside class range")
            if ((preds < 0) | (preds >= model.num_classes)).any():
                raise ValueError("Unexpected validation prediction outside class range")
            counts = np.bincount(targets * model.num_classes + preds,
                                 minlength=model.num_classes ** 2)
            cm += counts.reshape(model.num_classes, model.num_classes)

    if steps == 0:
        raise RuntimeError("Validation yielded no chunks")

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(model.num_classes, dtype=float), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(model.num_classes, dtype=float), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(model.num_classes, dtype=float), where=(precision + recall) != 0)
    avg_losses = {name: value / steps for name, value in losses.items()}
    avg_loss = avg_losses["final"] + lambda_smooth * avg_losses["smooth"]
    acc = float(tp.sum() / cm.sum()) if cm.sum() else 0.0
    return ValResults(
        loss=avg_loss, acc=acc, macro_f1=float(f1.mean()),
        precision=precision, recall=recall, f1=f1, support=support,
        predicted=predicted, confusion=cm, aux_losses=avg_losses,
    )


def print_per_class_metrics(name, result):
    print(f"\n  [{name}] Val Loss: {result.loss:.4f} | Accuracy: {result.acc:.4f} "
          f"| Macro F1: {result.macro_f1:.4f}")
    print("  Per-Class Metrics (final TCN predictions):")
    print(f"    {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'Support':>10} {'Predicted':>10}")
    for i in range(NUM_CLASSES):
        print(f"    {DISPLAY_NAMES[i]:<15} {result.precision[i]:>10.4f} "
              f"{result.recall[i]:>10.4f} {result.f1[i]:>10.4f} "
              f"{result.support[i]:>10d} {result.predicted[i]:>10d}")


def print_cecum_comparison(raw, ema):
    c = CECUM_ID
    print("\n" + "=" * 78)
    print("  CECUM — most important clinical monitoring metric (frame-wise)")
    print(f"  RAW | Precision: {raw.precision[c]:.4f} | Recall: {raw.recall[c]:.4f} "
          f"| F1: {raw.f1[c]:.4f} | GT frames: {raw.support[c]} "
          f"| Predicted frames: {raw.predicted[c]}")
    print(f"  EMA | Precision: {ema.precision[c]:.4f} | Recall: {ema.recall[c]:.4f} "
          f"| F1: {ema.f1[c]:.4f} | GT frames: {ema.support[c]} "
          f"| Predicted frames: {ema.predicted[c]}")
    print(f"  EMA - RAW | Delta Precision: {ema.precision[c] - raw.precision[c]:+.4f} "
          f"| Delta Recall: {ema.recall[c] - raw.recall[c]:+.4f}")
    if raw.support[c] == 0:
        print("  WARNING: No ground-truth cecum frames; recall is undefined (shown as 0).")
    print("=" * 78)


def append_metrics_csv(path, epoch, train_loss, raw, ema, lr):
    row = {"epoch": epoch, "lr": lr, "train_loss": train_loss}
    for name, result in (("raw", raw), ("ema", ema)):
        row[f"{name}_val_loss"] = result.loss
        row[f"{name}_acc"] = result.acc
        row[f"{name}_macro_f1"] = result.macro_f1
        for idx in range(NUM_CLASSES):
            cls = DISPLAY_NAMES[idx]
            for metric in ("precision", "recall", "f1", "support", "predicted"):
                row[f"{name}_{cls}_{metric}"] = getattr(result, metric)[idx].item()
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------------------------------------------------------
# Prediction cache: corrected for ContextMambav2TASver's 6-value return.
# -----------------------------------------------------------------------------
@torch.no_grad()
def cache_predictions(model, dataloader, device, save_path):
    model.eval()
    probs_chunks, labels_chunks, reset_chunks = [], [], []
    worker_states = {}
    for batch in tqdm(dataloader, desc=f"Caching {os.path.basename(save_path)}"):
        vision, contexts, labels, _future_labels, reset_mask, worker_id = prepare_batch(batch, device)
        previous = worker_states.get(worker_id)
        if previous is not None:
            previous = apply_reset_mask(previous, reset_mask)

        (_logits_wo, _future_logits, _logits_w, _tmp_logits,
         final_logits, next_states) = model(
            vision_embeddings=vision, contexts=contexts,
            pass_states=previous, labels=labels,
        )
        worker_states[worker_id] = detach_states(next_states)
        probs = F.softmax(final_logits, dim=-1)
        frame_reset = torch.zeros_like(labels, dtype=torch.bool)
        frame_reset[:, 0] = reset_mask.bool().reshape(-1)

        probs_flat = probs.reshape(-1, model.num_classes).cpu().numpy()
        labels_flat = labels.reshape(-1).cpu().numpy()
        reset_flat = frame_reset.reshape(-1).cpu().numpy()
        valid = labels_flat != IGNORE_INDEX
        probs_chunks.append(probs_flat[valid])
        labels_chunks.append(labels_flat[valid])
        reset_chunks.append(reset_flat[valid])

    if not labels_chunks:
        raise RuntimeError("No data to cache")
    np.savez_compressed(
        save_path,
        probs=np.concatenate(probs_chunks),
        labels=np.concatenate(labels_chunks),
        reset_masks=np.concatenate(reset_chunks),
    )
    print(f"Saved prediction cache: {save_path}")


# -----------------------------------------------------------------------------
# Data and model assembly
# -----------------------------------------------------------------------------
def make_dataset(phase, shuffle):
    return RealColonStreamingDataset(
        video_root=VIDEO_ROOT,
        batch_size_per_worker=1,
        split_dir=SPLIT_DIR,
        fold=FOLD,
        phase=phase,
        chunk_size=300,
        num_future_seconds=12,
        fps=5,
        target_fps=5,
        use_memory_bank=True,
        context_seconds=600,
        context_fps=5,
        shuffle=shuffle,
        use_emb=True,
        emb_dim=1024,
        transform=None,
    )


def build_model(device):
    config = MambaTemporalConfig(d_model=1024, n_layer=8)
    base_container = MambaTemporalSegmentation(
        config=config,
        vision_dim=1024,
        num_classes=NUM_CLASSES,
        device=device,
        loss_fn=nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
    )

    checkpoint_path = os.path.join(PRETRAIN_DIR, PRETRAIN_FILENAME)
    found_pretrain = os.path.exists(checkpoint_path)
    if found_pretrain:
        print(f"Loading base checkpoint: {checkpoint_path}")
        result = base_container.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True),
            strict=False,
        )
        print(f"  Missing keys: {len(result.missing_keys)}; "
              f"unexpected keys: {len(result.unexpected_keys)}")
        if result.missing_keys:
            print(f"  Missing examples: {result.missing_keys[:8]}")
        if result.unexpected_keys:
            print(f"  Unexpected examples: {result.unexpected_keys[:8]}")
    else:
        print(f"Pretrain checkpoint not found ({checkpoint_path}); using random initialization")

    if FREEZE_BASE:
        if not found_pretrain:
            raise RuntimeError("Cannot freeze base backbone without pretrained weights")
        for param in base_container.parameters():
            param.requires_grad_(False)
        print("Base Mamba frozen")

    full_model = ContextMambav2TASver(
        base_model=base_container.backbone,
        d_model=1024,
        num_classes=NUM_CLASSES,
        num_future=12,
        use_multihead=True,
        target_fps=5,
        context_fps=5,
        query_fps=5,
        num_layers_per_stage=4,
        compression_ratio=300.0,
        frames_per_query=[30, 10],
        hidden_expansion=4,
        num_stages=2,
        dropout=0.5,
    ).to(device)
    return full_model


def main():
    if not (0 <= EMA_DECAY < 1):
        raise ValueError("EMA_DECAY must be in [0, 1)")
    if EARLY_STOP_METRIC != "ema_macro_f1":
        raise ValueError("This script currently supports EARLY_STOP_METRIC='ema_macro_f1'")

    set_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    raw_best_path = os.path.join(SAVE_DIR, f"{RUN_NAME}_best_raw.pth")
    ema_best_path = os.path.join(SAVE_DIR, f"{RUN_NAME}_best_ema.pth")
    metrics_csv = os.path.join(SAVE_DIR, f"{RUN_NAME}_epoch_metrics.csv")
    if os.path.exists(metrics_csv):
        os.remove(metrics_csv)  # Starting a new run, not resuming an old one.

    if EVAL_PHASE == "test":
        print("\nWARNING: EVAL_PHASE='test' matches your existing code. If this "
              "is a held-out test set, DO NOT use its results for checkpoint "
              "selection. Supply a genuine validation split.\n")

    print(f"Device={device}; Fold={FOLD}; EMA decay={EMA_DECAY}; "
          f"early stopping={EARLY_STOP_METRIC}")
    train_dataset = make_dataset(phase="train", shuffle=True)
    val_dataset = make_dataset(phase=EVAL_PHASE, shuffle=False)
    # Re-seed the single validation worker before RAW and EMA passes so any
    # random preprocessing is replayed identically. The dataset itself must
    # still enumerate identical validation examples in shuffle=False mode.
    val_generator = torch.Generator()
    val_loader = DataLoader(
        val_dataset, batch_size=None, num_workers=1,
        worker_init_fn=seed_worker, generator=val_generator,
    )

    if WEIGHTED_LOSS:
        # The source script ultimately used THIS function (the earlier
        # compute_weights_from_dataset result was overwritten).
        class_weights = train_dataset.compute_weights().to(device)
        print("Dataset class weights:", class_weights)
    else:
        class_weights = None
        print("Using unweighted cross entropy")

    full_model = build_model(device)
    ema_model = AveragedModel(
        full_model,
        multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY),
        use_buffers=EMA_USE_BUFFERS,
    )
    ema_model.requires_grad_(False)
    ema_model.eval()

    optimizer = torch.optim.AdamW(
        (p for p in full_model.parameters() if p.requires_grad),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    if SCHEDULER_CHOICE == "reduceonplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
        )
    elif SCHEDULER_CHOICE == "cosine_with_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - WARMUP_EPOCHS,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS],
        )
    else:
        raise ValueError(f"Unsupported scheduler: {SCHEDULER_CHOICE}")

    best_raw_f1 = -float("inf")
    best_ema_f1 = -float("inf")
    best_raw_epoch = None
    best_ema_epoch = None
    patience_counter = 0
    total_ema_updates = 0

    for epoch in range(EPOCHS):
        print(f"\n{'#' * 23} Epoch {epoch + 1}/{EPOCHS} {'#' * 23}")
        train_dataset.set_epoch(epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=VIRTUAL_BATCH_SIZE,
            worker_init_fn=seed_worker,
            generator=g,
        )

        train_loss, num_updates = train_one_epoch(
            full_model, train_loader, optimizer, device,
            class_weights=class_weights,
            ema_model=ema_model,
            accumulation_steps=VIRTUAL_BATCH_SIZE,
            lambda_smooth=TRAIN_SMOOTH_WEIGHT,
        )
        total_ema_updates += num_updates

        val_generator.manual_seed(SEED + 10000)
        raw_val = validate(full_model, val_loader, device, class_weights,
                           lambda_smooth=VAL_SMOOTH_WEIGHT, desc="Validating RAW")
        val_generator.manual_seed(SEED + 10000)
        ema_val = validate(ema_model.module, val_loader, device, class_weights,
                           lambda_smooth=VAL_SMOOTH_WEIGHT, desc="Validating EMA")

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1} Summary | Train Loss: {train_loss:.4f} "
              f"| LR: {lr_now:.7g} | EMA updates: {total_ema_updates}")
        print_per_class_metrics("RAW", raw_val)
        print_per_class_metrics("EMA", ema_val)
        print_cecum_comparison(raw_val, ema_val)
        append_metrics_csv(metrics_csv, epoch + 1, train_loss, raw_val, ema_val, lr_now)

        if SAVE_ALL_EPOCHS:
            torch.save(full_model.state_dict(), os.path.join(
                SAVE_DIR, f"{RUN_NAME}_raw_epoch{epoch + 1:03d}.pth"))
            torch.save(ema_model.module.state_dict(), os.path.join(
                SAVE_DIR, f"{RUN_NAME}_ema_epoch{epoch + 1:03d}.pth"))

        # Checkpoint by MACRO F1, not validation CE, for both branches.
        if raw_val.macro_f1 > best_raw_f1:
            best_raw_f1 = raw_val.macro_f1
            best_raw_epoch = epoch + 1
            torch.save(full_model.state_dict(), raw_best_path)
            print(f"  New RAW best Macro F1={best_raw_f1:.4f} (epoch {best_raw_epoch})")

        if ema_val.macro_f1 > best_ema_f1:
            best_ema_f1 = ema_val.macro_f1
            best_ema_epoch = epoch + 1
            patience_counter = 0
            torch.save(ema_model.module.state_dict(), ema_best_path)
            print(f"  New EMA best Macro F1={best_ema_f1:.4f} (epoch {best_ema_epoch})")
        else:
            patience_counter += 1
            print(f"  EMA early-stopping patience: {patience_counter}/{PATIENCE}")

        # Match the original optimization schedule: plateau uses RAW CE loss.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(raw_val.loss)
        else:
            scheduler.step()

        if patience_counter >= PATIENCE:
            print("Early stopping triggered based on EMA Macro F1")
            break

    print(f"\nBest RAW: epoch={best_raw_epoch}, Macro F1={best_raw_f1:.4f}, path={raw_best_path}")
    print(f"Best EMA: epoch={best_ema_epoch}, Macro F1={best_ema_f1:.4f}, path={ema_best_path}")
    print(f"Epoch metrics CSV: {metrics_csv}")

    if CACHE_BEST_PREDICTIONS:
        # Loading state_dict into the same architecture avoids invoking an EMA
        # wrapper during streaming inference (and uses independent state caches).
        full_model.load_state_dict(torch.load(raw_best_path, map_location=device, weights_only=True))
        val_generator.manual_seed(SEED + 10000)
        cache_predictions(full_model, val_loader, device,
                          os.path.join(SAVE_DIR, f"{RUN_NAME}_best_raw_predictions_fold{FOLD}.npz"))
        full_model.load_state_dict(torch.load(ema_best_path, map_location=device, weights_only=True))
        val_generator.manual_seed(SEED + 10000)
        cache_predictions(full_model, val_loader, device,
                          os.path.join(SAVE_DIR, f"{RUN_NAME}_best_ema_predictions_fold{FOLD}.npz"))


if __name__ == "__main__":
    main()

