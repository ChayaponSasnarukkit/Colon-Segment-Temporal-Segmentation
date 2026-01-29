import os
import sys
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
import torch

# --- CRITICAL FIX: Force load the compiled kernel ---
try:
    import selective_scan_cuda
    print("✅ Manually loaded selective_scan_cuda")
except ImportError as e:
    print(f"❌ Failed to manual load selective_scan_cuda: {e}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import yaml
from datetime import timedelta
from torch.utils.data import DataLoader

# Modern Metrics
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)

# Optimizer & Scheduler
from transformers import get_cosine_schedule_with_warmup
import math

# Callbacks & Logging
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from model.EndoMamba import endomamba_small

# Transforms V2
import torchvision.transforms.v2 as T
from dataset.cas_location import RawVideoDataModule
# --- PLACEHOLDER IMPORTS (Replace these with your actual files) ---
# from model.mamba import endomamba_small 
# from datamodule.endo_video import EndoVideoDataset

class EndoMambaModule(L.LightningModule):
    def __init__(self, config: dict, class_weights: torch.Tensor = None, all_samples=100):
        """
        Modernized EndoMamba Module.
        Args:
            config (dict): Configuration dictionary.
            class_weights (Tensor): Pre-calculated weights for imbalance handling.
            all_samples (int): Total samples for scheduler calculation.
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.all_samples = all_samples

        # 1. Initialize Model
        self.model = endomamba_small(
            pretrained=config["pretrained"], 
            num_classes=config["num_classes"]
        )

        # 2. Loss Function
        self.register_buffer('class_weights', class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # 3. Metrics - Using MetricCollection for cleaner logging
        metrics = MetricCollection({
            'Accuracy': MulticlassAccuracy(num_classes=config["num_classes"]),
            'F1_macro': MulticlassF1Score(num_classes=config["num_classes"], average='macro'),
            'AUROC_macro': MulticlassAUROC(num_classes=config["num_classes"], average='macro', thresholds=None),
            'Precision_macro': MulticlassPrecision(num_classes=config["num_classes"], average='macro'),
            'Recall_macro': MulticlassRecall(num_classes=config["num_classes"], average='macro'),
        })
        
        # Clone metrics for different stages
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # x shape: (B, C, T, H, W) for video, or (B, C, H, W) for image
        
        # Ensure y is correct shape/type for CE Loss
        if y.ndim > 1:
            target_indices = torch.argmax(y, dim=1)
        else:
            target_indices = y

        logits = self(x)
        loss = self.criterion(logits, target_indices)

        # Log Loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Log Metrics (Optional: Logging metrics every step slows down training, usually epoch is enough)
        preds = torch.softmax(logits, dim=1)
        self.train_metrics.update(preds, target_indices)
        
        return loss

    def on_train_epoch_end(self):
        # Compute and log train metrics at end of epoch
        metric_dict = self.train_metrics.compute()
        self.log_dict(metric_dict, logger=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if y.ndim > 1:
            target_indices = torch.argmax(y, dim=1)
        else:
            target_indices = y

        logits = self(x)
        loss = self.criterion(logits, target_indices)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        preds_proba = torch.softmax(logits, dim=1)
        self.valid_metrics.update(preds_proba, target_indices)

    def on_validation_epoch_end(self):
        metric_dict = self.valid_metrics.compute()
        self.log_dict(metric_dict, logger=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        if y.ndim > 1:
            target_indices = torch.argmax(y, dim=1)
        else:
            target_indices = y
            
        logits = self(x)
        preds_proba = torch.softmax(logits, dim=1)
        self.test_metrics.update(preds_proba, target_indices)

    def on_test_epoch_end(self):
        metric_dict = self.test_metrics.compute()
        self.log_dict(metric_dict, logger=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """
        Sets up AdamW with parameter grouping (weight decay handling) 
        and Cosine Scheduler with Warmup.
        """
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight'] # Exclude norms/bias from weight decay

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.get("weight_decay", 0.05)
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=self.config["lr"]
        )

        # Scheduler Calculation
        TOTAL_SAMPLES = int(self.all_samples)
        BATCH_SIZE = self.config["batch_size"]
        GRAD_ACCUM_STEPS = self.config.get("grad_accum_steps", 1)
        EPOCHS = self.config["epochs"]
        
        total_batches = TOTAL_SAMPLES // BATCH_SIZE
        num_training_steps = math.ceil(total_batches / GRAD_ACCUM_STEPS) * EPOCHS
        num_warmup_steps = int(num_training_steps * 0.1) # 10% Warmup

        print(f"📉 Scheduler: {num_warmup_steps} warmup steps, {num_training_steps} total steps.")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

def main():
    # 1. Configuration
    # We match these to the arguments required by RawVideoDataModule
    args = {
        # Model Hyperparameters
        "num_classes": 10,  # Updated to match your 10 anatomy classes
        "lr": 1e-4,
        "weight_decay": 0.05,
        "pretrained": True,
        
        # Data Hyperparameters
        "batch_size": 8,       # Lower batch size for video (memory heavy)
        "num_workers": 4,
        "context_length": 8,  # Number of frames (T)
        "downsample_factor": 60, # 1 FPS if video is 60fps
        "height": 224,
        "width": 224,
        
        # Training Hyperparameters
        "epochs": 5,
        "grad_accum_steps": 4, # Effective batch size = 8 * 4 = 32
        "precision": "16-mixed", # crucial for video training memory
        "gradient_clip_val": 1.0,
        
        # Paths (UPDATE THESE)
        "master_csv_path": "/scratch/lt200353-pcllm/location/cas_colon/Video_Label.csv",
        "video_root": "/scratch/lt200353-pcllm/location/cas_colon/",
        "seed": 42
    }

    # 2. Initialize Data Module
    # We use the LightningDataModule wrapper which handles the splitting and loaders
    dm = RawVideoDataModule(
        master_csv_path=args["master_csv_path"],
        video_root=args["video_root"],
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        seed=args["seed"],
        context_length=args["context_length"],
        downsample_factor=args["downsample_factor"]
    )
    
    # We must call setup() manually here if we want to calculate class weights 
    # BEFORE passing them to the model (since DM typically sets up in trainer.fit)
    print("⏳ Setting up data module to calculate class weights...")
    dm.setup(stage='fit')
    
    # 3. Calculate Class Weights for Imbalance
    # We iterate over the created dataset to count labels
    print("⚖️  Calculating class weights (this might take a moment)...")
    # Access the underlying dataset created in setup()
    train_labels = [sample[2] for sample in dm.train_ds.samples] 
    class_counts = np.bincount(train_labels, minlength=args["num_classes"])
    
    # Standard weighting: Total / (NumClasses * Count)
    total_samples = len(train_labels)
    weights = total_samples / (args["num_classes"] * class_counts + 1e-6)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print(f"✅ Class Weights calculated: {class_weights_tensor}")

    # 4. Initialize Model
    # We pass total_samples to the model for the scheduler calculation
    model = EndoMambaModule(
        config=args, 
        class_weights=class_weights_tensor, 
        all_samples=total_samples
    )

    # 5. Callbacks
    # Save best model based on macro F1 or AUROC
    checkpoint_cb = ModelCheckpoint(
        dirpath="/scratch/lt200353-pcllm/location/checkpoints/endomamba_video",
        filename='{epoch:02d}-{val/F1_macro:.4f}',
        save_top_k=3,
        monitor='val/F1_macro',
        mode='max'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(
        save_dir='./tb_log', 
        version='video_run_v1'
    )

    # 6. Trainer
    trainer = L.Trainer(
        max_epochs=args['epochs'],
        accelerator="gpu",
        devices=1,  # Set to -1 for all GPUs
        precision=args['precision'],
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=args['gradient_clip_val'],
        accumulate_grad_batches=args['grad_accum_steps'],
        # Strategy usually needed for multi-gpu video training
        # strategy="ddp_find_unused_parameters_true" 
    )

    # 7. Start Training
    # Passing 'dm' automatically handles train/val dataloaders
    print("🚀 Starting Video Training...")
    trainer.fit(model, datamodule=dm)
    
    # Optional: Test after training
    # trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    # Ensure numpy is imported for weight calc
    import numpy as np
    main()
