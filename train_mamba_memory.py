"""Train the '+M' ablation (Base Mamba + long-term memory + fusion) for Reviewer Q3.

Runs the SAME pipeline for both compressors on either dataset, so the only
difference between paired runs is how memory tokens are formed:

    python train_mamba_memory.py --dataset cas  --compressor mamba     --fold 1
    python train_mamba_memory.py --dataset cas  --compressor attention --fold 1
    python train_mamba_memory.py --dataset real --compressor mamba     --fold 1
    python train_mamba_memory.py --dataset real --compressor attention --fold 1

Protocol (EMA, early stopping on macro F1, cosine schedule with warmup,
gradient accumulation, weighted CE, pretrained base Mamba) is copied from the
REAL-Colon TASver script. Values marked TODO(check) are guesses: replace them
with whatever your Table 3 '+M' runs used.
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.AttentionMemory import count_params
from model.CMamba import MambaTemporalSegmentation
from model.MambaMemory import MambaMemory

IGNORE_INDEX = -100


# -----------------------------------------------------------------------------
# Dataset-specific configuration
# -----------------------------------------------------------------------------
REAL_CLASS_MAP = {
    "outside": 0, "insertion": 1, "ceacum": 2, "ileum": 3, "ascending": 4,
    "transverse": 5, "descending": 6, "sigmoid": 7, "rectum": 8,
}
CAS_CLASS_NAMES = [
    "Terminal_Ileum", "Cecum", "Ascending_Colon", "Hepatic_Flexure", "Transverse_Colon",
    "Splenic_Flexure", "Descending_Colon", "Sigmoid_Colon", "Rectum", "Anal_Canal",
]

CONFIGS = {
    # Copied from your REAL-Colon TASver script.
    "real": dict(
        class_names=["cecum" if n == "ceacum" else n for n in REAL_CLASS_MAP],
        video_root="/project/lt200353-pcllm/3d_report_gen/real-colon/",
        split_dir="/home/csasnaru/temporal_segmentation/data/dataset/RC_lists/5_fold/",
        pretrain_path="/project/lt200353-pcllm/3d_report_gen/real-colon/neww3best_mamba_model.pth",
        save_root="/project/pv927002-dcpcce/rebuttal_q3_plus_m/real",
        source_fps=5,
        target_fps=5,
        context_fps=5,
        context_seconds=600,
        chunk_size=300,               # T_S = 60 s at 5 FPS
        compression_ratio=300,        # context frames per token (60 s at 5 FPS)
        frames_per_query=(30, 10),
        num_layers_per_stage=4,
        lr=3e-5,
        virtual_batch_size=12,        # Table S1: batch size 12
        dropout=0.5,
        hidden_expansion=4,
        eval_phase="test",            # TODO(check): split used for checkpoint selection
    ),
    # Table S1: r_local 30 FPS, r_L 5 FPS, T_L 10 min, T_S 1 min, lr 3e-5, batch 4.
    "cas": dict(
        class_names=CAS_CLASS_NAMES,
        video_root="/project/lt200353-pcllm/3d_report_gen/cas_colon/features_dinov3/",
        # TODO(check): path pattern of your per-fold CSVs ({fold}, {phase} are filled in).
        csv_template="./cv_folds_generated/fold{cfg_fold}_{phase}.csv",
        pretrain_path="",             # TODO(check): CAS base-Mamba checkpoint ('' = random init)
        save_root="/project/pv927002-dcpcce/rebuttal_q3_plus_m/cas",
        source_fps=60,
        target_fps=30,
        context_fps=5,
        context_seconds=600,
        chunk_size=1800,              # T_S = 60 s at 30 FPS
        compression_ratio=300,        # 60 s at 5 FPS
        frames_per_query=(30, 10),    # TODO(check): must multiply to compression_ratio
        num_layers_per_stage=2,       # TODO(check)
        lr=3e-5,
        virtual_batch_size=4,         # Table S1: batch size 4
        dropout=0.2,                  # TODO(check)
        hidden_expansion=2,           # TODO(check)
        eval_phase="test",             # TODO(check): split used for checkpoint selection
    ),
}

EPOCHS = 100
PATIENCE = EPOCHS // 2
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 5
WEIGHTED_LOSS = True
EMA_DECAY = 0.995
GRAD_CLIP_NORM: Optional[float] = None
NUM_FUTURE_SECONDS = 12       # only needed by the datasets; unused by '+M'


@dataclass
class MambaTemporalConfig:
    d_model: int = 1024
    n_layer: int = 8
    d_intermediate: int = 0
    ssm_cfg: dict = field(default_factory=lambda: {
        "d_state": 32, "d_conv": 4, "expand": 4, "dt_rank": "auto",
        "layer": "Mamba1", "use_fast_path": False,
    })
    rms_norm: bool = True
    norm_epsilon: float = 1e-5
    fused_add_norm: bool = True
    residual_in_fp32: bool = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cas", "real"], required=True)
    p.add_argument("--compressor", choices=["mamba", "attention"], required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=441)
    # Attention baseline: copy these from your CMeRT config (MODEL.LSTR.*)
    p.add_argument("--attn-enc", default="16,1;10,2",
                   help="ENC_MODULE as 'queries,layers;...'; last stage queries must equal the token count")
    p.add_argument("--attn-heads", type=int, default=16)
    p.add_argument("--attn-ffn", type=int, default=1024)
    p.add_argument("--attn-dropout", type=float, default=0.2)
    p.add_argument("--attn-activation", default="relu")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
def make_dataset(name, cfg, fold, phase, shuffle):
    common = dict(
        batch_size_per_worker=1, chunk_size=cfg["chunk_size"],
        fps=cfg["source_fps"], target_fps=cfg["target_fps"],
        use_memory_bank=True, context_seconds=cfg["context_seconds"],
        context_fps=cfg["context_fps"], num_future_seconds=NUM_FUTURE_SECONDS,
        shuffle=shuffle, use_emb=True, emb_dim=1024, transform=None,
    )
    if name == "real":
        from dataset.real_locationv2 import RealColonStreamingDataset
        return RealColonStreamingDataset(
            video_root=cfg["video_root"], split_dir=cfg["split_dir"],
            fold=fold, phase=phase, **common,
        )
    # TODO(check): module path of your CAS-Colon dataset file
    from dataset.cas_location import MedicalStreamingDataset
    return MedicalStreamingDataset(
        csv_path=cfg["csv_template"].format(fold=fold, phase=phase),
        video_root=cfg["video_root"], **common,
    )


def cas_class_weights(dataset, num_classes):
    """MedicalStreamingDataset has no compute_weights(). This is inverse frequency
    over the sampled training labels, normalized to mean 1.
    TODO(check): replace with the weighting your CAS runs used."""
    from dataset.cas_location import build_dense_label
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, row in dataset.df.iterrows():
        labels = build_dense_label(row, int(row["TotalFrames"]), dataset.fps)[:: dataset.step]
        labels = labels[labels >= 0]
        counts += np.bincount(labels, minlength=num_classes)
    freq = counts / counts.sum()
    weights = 1.0 / np.maximum(freq, 1e-8)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def class_weights_for(name, dataset, num_classes):
    if not WEIGHTED_LOSS:
        return None
    if name == "real":
        return dataset.compute_weights()
    return cas_class_weights(dataset, num_classes)


# -----------------------------------------------------------------------------
# Helpers (same logic as the TASver script)
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def detach_states(states):
    if states is None:
        return None
    detach = lambda t: t.detach().clone() if t is not None else None
    base = states.get("base_states")
    return {
        "base_states": [(detach(s), detach(c)) for s, c in base] if base is not None else None,
        "tcn_caches": None,
    }


def apply_reset_mask(states, reset_mask):
    if states is None:
        return None

    def masked(t):
        if t is None:
            return None
        return t.masked_fill(reset_mask.bool().reshape(-1, *([1] * (t.ndim - 1))), 0.0)

    base = states.get("base_states")
    return {
        "base_states": [(masked(s), masked(c)) for s, c in base] if base is not None else None,
        "tcn_caches": None,
    }


def prepare_batch(batch, device):
    # Both datasets yield: vision, contexts, labels, future_labels, reset_mask, context_masks, worker_id
    vision, contexts, labels, _future_labels, reset_mask, context_masks, worker_id = batch
    vision, contexts, labels = vision.to(device), contexts.to(device), labels.to(device)
    reset_mask, context_masks = reset_mask.to(device), context_masks.to(device)
    if labels.shape[0] != 1:
        raise ValueError("Expects batch_size_per_worker=1 (B=1).")
    # Valid memory frames come first (right padding), as in both datasets.
    valid_contexts = contexts[:, : int(context_masks[0].sum().item()), :]
    w_id = int(worker_id.reshape(-1)[0].item()) if isinstance(worker_id, torch.Tensor) else int(worker_id)
    return vision, valid_contexts, labels, reset_mask, w_id


def safe_ce_loss(logits, targets, criterion):
    if not (targets != IGNORE_INDEX).any():
        return logits.sum() * 0.0
    return criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


# -----------------------------------------------------------------------------
# Train / validate
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, criterion, ema_model, accum_steps):
    model.train()
    worker_states = {}
    optimizer.zero_grad(set_to_none=True)
    total_loss, num_chunks, pending = 0.0, 0, 0

    def perform_update(pending_chunks):
        if pending_chunks != accum_steps:
            scale = accum_steps / pending_chunks
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        ema_model.update_parameters(model)
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="Training")):
        vision, contexts, labels, reset_mask, w_id = prepare_batch(batch, device)
        previous = apply_reset_mask(worker_states.get(w_id), reset_mask)

        out = model(vision_embeddings=vision, contexts=contexts, pass_states=previous, labels=labels)
        loss = safe_ce_loss(out["logits"], labels, criterion)
        (loss / accum_steps).backward()

        worker_states[w_id] = detach_states(out["next_states"])
        total_loss += float(loss.detach())
        num_chunks += 1
        pending += 1
        if pending == accum_steps:
            perform_update(pending)
            pending = 0
        if step % 50 == 0:
            print(f"  [Train] Step {step} | Loss: {float(loss.detach()):.4f}")

    if pending:
        perform_update(pending)
    return total_loss / max(num_chunks, 1)


@dataclass
class ValResults:
    acc: float
    macro_f1: float
    f1: np.ndarray


@torch.no_grad()
def validate(model, loader, device, desc):
    model.eval()
    C = model.num_classes
    cm = np.zeros((C, C), dtype=np.int64)
    worker_states = {}  # raw and EMA never share streaming states
    for batch in tqdm(loader, desc=desc):
        vision, contexts, labels, reset_mask, w_id = prepare_batch(batch, device)
        previous = apply_reset_mask(worker_states.get(w_id), reset_mask)
        out = model(vision_embeddings=vision, contexts=contexts, pass_states=previous, labels=labels)
        worker_states[w_id] = detach_states(out["next_states"])

        preds = out["logits"].argmax(-1).reshape(-1).cpu().numpy()
        targets = labels.reshape(-1).cpu().numpy()
        valid = targets != IGNORE_INDEX
        if valid.any():
            cm += np.bincount(targets[valid] * C + preds[valid], minlength=C * C).reshape(C, C)

    tp, support, predicted = cm.diagonal(), cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(C), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(C), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(C), where=(precision + recall) != 0)
    acc = float(tp.sum() / cm.sum()) if cm.sum() else 0.0
    return ValResults(acc=acc, macro_f1=float(f1.mean()), f1=f1)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def parse_enc_module(spec):
    return tuple(tuple(int(v) for v in stage.split(",")) for stage in spec.split(";"))


def build_model(args, cfg, num_classes, device):
    base_container = MambaTemporalSegmentation(
        config=MambaTemporalConfig(), vision_dim=1024, num_classes=num_classes,
        device=device, loss_fn=nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
    )
    ckpt = cfg["pretrain_path"]
    if ckpt and os.path.exists(ckpt):
        result = base_container.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True), strict=False)
        print(f"Loaded base checkpoint {ckpt} ({len(result.missing_keys)} missing, "
              f"{len(result.unexpected_keys)} unexpected keys)")
    else:
        print(f"No base checkpoint ({ckpt!r}); random init")

    max_tokens = cfg["context_seconds"] * cfg["context_fps"] // cfg["compression_ratio"]
    model = MambaMemory(
        base_model=base_container.backbone,
        d_model=1024,
        num_classes=num_classes,
        compressor=args.compressor,
        compression_ratio=cfg["compression_ratio"],
        max_tokens=max_tokens,
        frames_per_query=cfg["frames_per_query"],
        num_layers_per_stage=cfg["num_layers_per_stage"],
        attn_kwargs=dict(
            enc_module=parse_enc_module(args.attn_enc),
            num_heads=args.attn_heads,
            dim_feedforward=args.attn_ffn,
            dropout=args.attn_dropout,
            activation=args.attn_activation,
        ),
        dropout=cfg["dropout"],
        hidden_expansion=cfg["hidden_expansion"],
    )
    return model.to(device)


def main():
    args = parse_args()
    cfg = CONFIGS[args.dataset]
    class_names = cfg["class_names"]
    num_classes = len(class_names)
    fpq = cfg["frames_per_query"]
    if int(np.prod(fpq)) != cfg["compression_ratio"]:
        raise ValueError(f"frames_per_query {fpq} must multiply to compression_ratio")

    set_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"plusM_{args.dataset}_{args.compressor}_fold{args.fold}_seed{args.seed}"
    save_dir = os.path.join(cfg["save_root"], run_name)
    os.makedirs(save_dir, exist_ok=True)
    raw_best_path = os.path.join(save_dir, "best_raw.pth")
    ema_best_path = os.path.join(save_dir, "best_ema.pth")
    metrics_csv = os.path.join(save_dir, "epoch_metrics.csv")
    if os.path.exists(metrics_csv):
        os.remove(metrics_csv)

    train_dataset = make_dataset(args.dataset, cfg, args.fold, "train", shuffle=True)
    val_dataset = make_dataset(args.dataset, cfg, args.fold, cfg["eval_phase"], shuffle=False)
    val_generator = torch.Generator()
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1,
                            worker_init_fn=seed_worker, generator=val_generator)

    class_weights = class_weights_for(args.dataset, train_dataset, num_classes)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print("Class weights:", class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    model = build_model(args, cfg, num_classes, device)
    print(f"Run: {run_name} | eval phase: {cfg['eval_phase']}")
    print(f"Compressor params: {count_params(model.compressor) / 1e6:.2f}M | "
          f"total params: {count_params(model) / 1e6:.2f}M")

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=False)
    ema_model.requires_grad_(False)
    ema_model.eval()

    accum = cfg["virtual_batch_size"]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS),
        ],
        milestones=[WARMUP_EPOCHS],
    )

    best_raw = (-1.0, None)   # (macro F1, epoch)
    best_ema = (-1.0, None)
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n######## Epoch {epoch}/{EPOCHS} ########")
        train_dataset.set_epoch(epoch - 1)
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=accum,
                                  worker_init_fn=seed_worker, generator=g)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion,
                                     ema_model, accum)

        val_generator.manual_seed(args.seed + 10000)
        raw = validate(model, val_loader, device, "Validating RAW")
        val_generator.manual_seed(args.seed + 10000)
        ema = validate(ema_model.module, val_loader, device, "Validating EMA")

        print(f"Train loss {train_loss:.4f} | RAW F1 {raw.macro_f1:.4f} Acc {raw.acc:.4f} "
              f"| EMA F1 {ema.macro_f1:.4f} Acc {ema.acc:.4f}")
        print("  EMA per-class F1: " + ", ".join(
            f"{class_names[i]}={ema.f1[i]:.3f}" for i in range(num_classes)))

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss,
               "raw_macro_f1": raw.macro_f1, "raw_acc": raw.acc,
               "ema_macro_f1": ema.macro_f1, "ema_acc": ema.acc}
        row.update({f"ema_f1_{class_names[i]}": float(ema.f1[i]) for i in range(num_classes)})
        write_header = not os.path.exists(metrics_csv)
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        if raw.macro_f1 > best_raw[0]:
            best_raw = (raw.macro_f1, epoch)
            torch.save(model.state_dict(), raw_best_path)
        if ema.macro_f1 > best_ema[0]:
            best_ema = (ema.macro_f1, epoch)
            torch.save(ema_model.module.state_dict(), ema_best_path)
            patience = 0
            print(f"  New EMA best macro F1 {ema.macro_f1:.4f}")
        else:
            patience += 1
            print(f"  EMA patience {patience}/{PATIENCE}")

        scheduler.step()
        if patience >= PATIENCE:
            print("Early stopping (EMA macro F1)")
            break

    print(f"\nBest RAW: F1={best_raw[0]:.4f} (epoch {best_raw[1]})")
    print(f"Best EMA: F1={best_ema[0]:.4f} (epoch {best_ema[1]})")
    print(f"Metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
