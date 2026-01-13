"""
src/intseq_bert/train.py

Pre-training entry point for IntSeqBERT.
Trains the Dual Stream Encoder using Masked Sequence Modeling with Automatic Weighted Loss.
"""

import argparse
import logging
import random
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast # PyTorch 2.x style
from tqdm import tqdm

# Internal modules
from . import config
from . import models
from . import loader
from . import collator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==========================================
# Helper Classes & Functions
# ==========================================

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_labels(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Converts Collator output to Model label format.
    
    Args:
        batch: Output dictionary from OEISCollator
        device: Target device
    Returns:
        Dictionary containing inputs moved to device and formatted labels
    """
    # Move inputs to device
    mag_features = batch["mag_inputs"].to(device)
    mod_features = batch["mod_inputs"].to(device)
    
    # Create padding mask (True where padding)
    # attention_mask is 1 for valid, 0 for pad -> invert for src_key_padding_mask
    src_key_padding_mask = (batch["attention_mask"] == 0).to(device)
    
    # Process Labels
    mag_labels = batch["mag_labels"].to(device)   # (B, L, 4)
    mod_labels = batch["mod_labels"].to(device)   # (B, L, 100)
    mask_matrix = batch["mask_matrix"].to(device) # (B, L)
    
    # 1. Magnitude: Extract log_val (index 0)
    mag_targets = mag_labels[:, :, 0] # (B, L)
    
    # 2. Sign: Convert One-hot to Index
    # [log, s+, s-, s0] -> slice [1:4] -> argmax
    sign_one_hot = mag_labels[:, :, 1:4] # (B, L, 3)
    sign_targets = torch.argmax(sign_one_hot, dim=-1) # (B, L)
    
    # 3. Modulo: Pass through (model handles ignore_index implicitly via masking)
    mod_targets = mod_labels
    
    return {
        "mag_features": mag_features,
        "mod_features": mod_features,
        "src_key_padding_mask": src_key_padding_mask,
        "labels": {
            "mag_targets": mag_targets,
            "sign_targets": sign_targets,
            "mod_targets": mod_targets,
            "mask_map": mask_matrix
        }
    }


# ==========================================
# Evaluation Logic
# ==========================================

def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device
) -> Dict[str, float]:
    """
    Run validation loop and calculate metrics.
    All accuracy metrics are returned as percentages (0-100).
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Metric Accumulators
    total_mask_count = 0
    total_mod_mask_count = 0 # mask_count * num_moduli
    
    correct_sign = 0
    correct_mag_thresh = 0 # |diff| < threshold
    sum_mag_mse = 0.0
    correct_mod = 0 # Sum of correct predictions across all moduli
    sum_mod_loss = 0.0  # Accumulated normalized mod loss
    
    with torch.no_grad():
        for raw_batch in tqdm(dataloader, desc="Validating", leave=False):
            # Prepare data
            inputs = prepare_labels(raw_batch, device)
            labels = inputs["labels"]
            mask_map = labels["mask_map"]
            
            if not mask_map.any():
                continue

            # Forward
            with autocast(device_type=device.type):
                outputs = model(
                    mag_features=inputs["mag_features"],
                    mod_features=inputs["mod_features"],
                    src_key_padding_mask=inputs["src_key_padding_mask"],
                    labels=labels
                )
            
            loss = outputs["loss"]
            total_loss += loss.item()
            num_batches += 1
            
            preds = outputs["predictions"]
            
            # --- Metrics Calculation (Masked positions only) ---
            # Using boolean indexing flattens the tensors
            
            # 1. Magnitude Metrics
            pred_mu = preds["mag_mu"][mask_map]
            target_mag = labels["mag_targets"][mask_map]
            
            diff = torch.abs(pred_mu - target_mag)
            
            # MSE
            sum_mag_mse += (diff ** 2).sum().item()
            # Accuracy (< threshold)
            correct_mag_thresh += (diff < config.MAG_ACC_THRESHOLD).sum().item()
            
            # 2. Sign Accuracy
            pred_sign = torch.argmax(preds["sign_logits"], dim=-1)[mask_map]
            target_sign = labels["sign_targets"][mask_map]
            correct_sign += (pred_sign == target_sign).sum().item()
            
            # 3. Modulo Accuracy (Mean across 100 mods)
            # Access helper method (handle DataParallel if needed)
            raw_model = model.module if hasattr(model, "module") else model
            mod_logits_split = raw_model._split_mod_logits(preds["mod_logits"]) # List of (B, L, m)
            
            # Vectorized calculation per modulus
            for i, m_logits in enumerate(mod_logits_split):
                # Slice and mask
                m_pred = torch.argmax(m_logits, dim=-1)[mask_map]
                m_target = labels["mod_targets"][:, :, i][mask_map]
                
                correct_mod += (m_pred == m_target).sum().item()
            
            # Get normalized mod loss from model output
            if "loss_breakdown" in outputs:
                sum_mod_loss += outputs["loss_breakdown"]["raw_mod"].item()
            
            # Update counts
            n_masked = mask_map.sum().item()
            total_mask_count += n_masked
            total_mod_mask_count += n_masked * config.NUM_MODULI

    # Calculate Averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    if total_mask_count > 0:
        mag_mse = sum_mag_mse / total_mask_count
        mag_acc = (correct_mag_thresh / total_mask_count) * 100
        sign_acc = (correct_sign / total_mask_count) * 100
        mod_acc = (correct_mod / total_mod_mask_count) * 100
        mod_loss = sum_mod_loss / num_batches if num_batches > 0 else 0.0
    else:
        mag_mse, mag_acc, sign_acc, mod_acc, mod_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        
    return {
        "val_loss": avg_loss,
        "mag_mse": mag_mse,
        "mag_acc": mag_acc,
        "sign_acc": sign_acc,
        "mod_acc": mod_acc,
        "mod_loss": mod_loss  # Normalized mod loss (1.0 = random prediction)
    }


# ==========================================
# Main Training Loop
# ==========================================

def train(args):
    # 1. Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Data Loading
    logger.info(f"Loading datasets (Split: {args.split_type})...")
    
    data_root = Path(args.data_root)
    split_dir = data_root / "splits" / args.split_type
    features_dir = data_root / "features"
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # Load Datasets using loader module
    train_dataset = loader.load_dataset(
        split_type=args.split_type,
        split_name="train",
        data_root=args.data_root
    )
    val_dataset = loader.load_dataset(
        split_type=args.split_type,
        split_name="val",
        data_root=args.data_root
    )
    
    # Initialize Collator
    collator_fn = collator.OEISCollator(mask_prob=config.MASK_PROB)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collator_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collator_fn,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 3. Model Initialization
    logger.info("Initializing model...")
    model = models.IntSeqForPreTraining(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    )
    
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    
    model.to(device)
    
    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=config.ADAMW_BETAS
    )
    
    # Calculate total steps for OneCycleLR
    total_steps = len(train_loader) * args.epochs // args.accum_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        total_steps=total_steps,
        pct_start=args.warmup_ratio
    )
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=args.patience)
    
    # 5. Training Loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, raw_batch in enumerate(progress_bar):
            # Prepare inputs & labels
            inputs = prepare_labels(raw_batch, device)
            
            # Forward
            with autocast(device_type=device.type):
                outputs = model(
                    mag_features=inputs["mag_features"],
                    mod_features=inputs["mod_features"],
                    src_key_padding_mask=inputs["src_key_padding_mask"],
                    labels=inputs["labels"]
                )
                loss = outputs["loss"] / args.accum_steps
            
            # Backward
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            current_loss = loss.item() * args.accum_steps
            train_loss += current_loss
            
            s_mag, s_sign, s_mod = model.loss_log_vars.detach().cpu().tolist()
            progress_bar.set_postfix({
                "loss": f"{current_loss:.3f}",
                "s_mag": f"{s_mag:.2f}",
                "s_mod": f"{s_mod:.2f}"
            })
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        logger.info(f"Validating Epoch {epoch+1}...")
        val_metrics = evaluate(model, val_loader, device)
        
        # Log Metrics
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
        logger.info(f"  Mag Acc:    {val_metrics['mag_acc']:.2f}% (MSE: {val_metrics['mag_mse']:.4f})")
        logger.info(f"  Sign Acc:   {val_metrics['sign_acc']:.2f}%")
        logger.info(f"  Mod Acc:    {val_metrics['mod_acc']:.2f}% (Loss: {val_metrics['mod_loss']:.4f})")
        
        # --- Checkpointing ---
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["val_loss"],
            "config": vars(args)
        }
        
        # Save Last
        torch.save(state, output_dir / "last_checkpoint.pt")
        
        # Save Best & Early Stopping check
        should_stop = early_stopping(val_metrics["val_loss"])
        
        if val_metrics["val_loss"] == early_stopping.best_loss:
            torch.save(state, output_dir / "best_model.pt")
            logger.info(f"New best model saved! (Loss: {val_metrics['val_loss']:.4f})")
        
        if should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train IntSeqBERT Encoder")
    
    # Path / Data
    parser.add_argument("--split_type", required=True, help="Split type (e.g., std, easy, all)")
    parser.add_argument("--data_root", type=str, default=config.DATA_ROOT, help="Path to data root")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    
    # Model Config
    parser.add_argument("--d_model", type=int, default=config.D_MODEL)
    parser.add_argument("--nhead", type=int, default=config.NHEAD)
    parser.add_argument("--num_layers", type=int, default=config.NUM_LAYERS)
    
    # Training Params
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup_ratio", type=float, default=config.DEFAULT_WARMUP_RATIO)
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=config.DEFAULT_NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
