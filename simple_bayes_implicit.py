import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import yaml
from datetime import timedelta
from torch.utils.data import DataLoader
import os

# Modern Metrics
from model.BayesianFilter import PLWrapper, GatedFusionBayesianNeuralFilter_Explicit
from model.EndoMamba import endomamba_small
from dataset.cas_locationv2 import CasColonDataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
# ==========================================
#   CONFIG & HYPERPARAMETERS
# ==========================================
CONFIG = {
    # Data Paths
    "csv_path": "/scratch/lt200353-pcllm/location/cas_colon/Video_Label.csv",
    "video_root": "/scratch/lt200353-pcllm/location/cas_colon/",
    "cache_root": "/scratch/lt200353-pcllm/location/cas_colon/cache_frames/", 
    
    # Model Hyperparams
    "num_classes": 10,       # Based on your dataset class list
    "embed_dim": 384,        # Output dim of backbone (e.g., ResNet18-3D=512, ResNet50=2048)
    "state_dim": 384,        # Hidden state for Bayesian Filter
    
    # Training Hyperparams
    "batch_size": 8,         # Adjust based on GPU VRAM
    "epochs": 50,
    "lr": 2e-4,              # Learning rate for Head/Filter
    "weight_decay": 1e-4,
    "grad_accum_steps": 4,
    "num_workers": 4,
    "grad_accum_steps": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Dataset Params
    "clip_len": 20,
    "sampling_rate": 6,
    "stride": 60,
    "crop_size": 224,
}

# ==========================================
#   MAIN EXECUTION
# ==========================================
def main():
    print(f"🚀 Starting Training on {CONFIG['device']}")
    
    # --- 1. Datasets & Loaders ---
    print("📂 Initializing Datasets...")
    
    # Train Set
    train_dataset = CasColonDataset(
        csv_path=CONFIG["csv_path"],
        video_root=CONFIG["video_root"],
        cache_root=CONFIG["cache_root"],
        clip_len=CONFIG["clip_len"],
        sampling_rate=CONFIG["sampling_rate"],
        stride=CONFIG["stride"],
        mode="train",
        crop_size=CONFIG["crop_size"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True
    )

    # Validation Set (Assuming same CSV, just using mode='val' logic if implemented, 
    # or you might split IDs. Here we reuse for demo but typically you split)
    val_dataset = CasColonDataset(
        csv_path=CONFIG["csv_path"],
        video_root=CONFIG["video_root"],
        cache_root=CONFIG["cache_root"],
        clip_len=CONFIG["clip_len"],
        sampling_rate=CONFIG["sampling_rate"],
        stride=CONFIG["stride"],
        mode="test",  # Use 'test' or 'val' to trigger deterministic transforms
        crop_size=CONFIG["crop_size"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    print(f"   Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # --- 2. Model Initialization ---
    print("🧠 Building Model...")
    
    # A. Define Backbone
    backbone = endomamba_small(
        pretrained=True, 
        num_classes=CONFIG["num_classes"],
        with_cls_token=False
    )
    
    # B. Define Bayesian Filter Model
    model = GatedFusionBayesianNeuralFilter_Explicit(
        backbone=backbone,
        num_classes=CONFIG["num_classes"],
        embed_dim=CONFIG["embed_dim"],  # Must match backbone output (512 for R3D-18)
        state_dim=CONFIG["state_dim"]
    )
    
    model.to(CONFIG["device"])

    # --- 3. Trainer Initialization ---
    print("⚙️  Initializing Trainer...")
    
    # We pass the dataset length to the trainer for scheduler calculations
    model.all_samples = len(train_dataset)
    model.config = CONFIG # Attach config to model so configure_optimizers can access it
    
    checkpoint_cb = ModelCheckpoint(
        dirpath="/scratch/lt200353-pcllm/location/checkpoints/8fix_endomamba_video_last",
        filename='{epoch:02d}-{val/F1_macro:.4f}',
        save_top_k=3,
        monitor='val/F1_macro',
        mode='max'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(
        save_dir='./tb_log', 
        version='8fix_video_run_last'
    )

    trainer = L.Trainer(
        max_epochs=CONFIG['epochs'],
        accelerator="gpu",
        devices=1,  # Set to -1 for all GPUs
        # precision=CONFIG['precision'],
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=CONFIG['gradient_clip_val'],
        accumulate_grad_batches=CONFIG['grad_accum_steps'],
        # Strategy usually needed for multi-gpu video training
        # strategy="ddp_find_unused_parameters_true" 
    )

    lightning_module = PLWrapper(
        model=model,
        config=CONFIG,
    )

    # --- 4. Start Training ---
    print("🔥 Start Training Loop")
    trainer.fit(lightning_module, train_loader, val_loader)

if __name__ == "__main__":
    # Ensure cache directory exists
    os.makedirs(CONFIG["cache_root"], exist_ok=True)
    main()