"""
Training script for IntSeqBERT model (Dual Stream Architecture).
Pretrains the encoder using Masked Modeling on both Magnitude and Mod Spectrum streams.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from . import loader
from . import collator
from . import bert_model


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to console and file."""
    log_file = output_dir / "train.log"
    
    logger = logging.getLogger("intseq_bert.train_bert")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config: Dict[str, Any]) -> None:
    """Main training function for Dual Stream BERT."""
    
    # 1. Setup Output
    output_dir = Path(config.get("output_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("=" * 50)
    logger.info("Starting IntSeqBERT Training (Dual Stream)")
    logger.info("=" * 50)
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # 2. Data Loading
    logger.info("Loading data from directory...")
    
    # Use the new load_and_split_data for directory-based .pt files
    train_ds, val_ds, test_ds = loader.load_and_split_data(
        features_dir=config["features_dir"],  # Changed from features_path
        metadata_path=config.get("metadata_path"),
        include_tags=config.get("include_tags"),
        exclude_tags=config.get("exclude_tags"),
        val_ratio=config.get("val_ratio", 0.05),
        test_ratio=config.get("test_ratio", 0.05),
        seed=config.get("seed", 42),
        max_samples=config.get("max_samples") # Optional debugging limit
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Create DualStreamCollator
    data_collator = collator.DualStreamCollator(
        mask_prob=config.get("mask_prob", 0.15)
    )
    
    # Create DataLoaders
    # num_workers > 0 is now safe and recommended because Dataset is lazy-loading
    num_workers = config.get("num_workers", 4)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 3. Model Initialization
    logger.info("Initializing Dual Stream Model...")
    model = bert_model.IntSeqBERT(
        mag_dim=config.get("mag_dim", 5),
        mod_dim=config.get("mod_dim", 200),
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 6),
        dim_feedforward=config.get("dim_feedforward", 512),
        max_len=config.get("max_len", 5000),
        dropout=config.get("dropout", 0.1)
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 0.01)
    )
    
    epochs = config.get("epochs", 10)
    num_training_steps = len(train_loader) * epochs
    
    num_warmup_steps = config.get("warmup_steps")
    if num_warmup_steps is None:
        num_warmup_steps = max(1, num_training_steps // 10)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
    
    # 5. Training Loop
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'=' * 50}")
        
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Training")
        for batch in pbar:
            # Move batch to device (handle dict keys dynamically)
            # Keys: mag_inputs, mod_inputs, attention_mask, mask_matrix, mag_labels, mod_labels, targets
            
            mag_inputs = batch["mag_inputs"].to(device)
            mod_inputs = batch["mod_inputs"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mask_matrix = batch["mask_matrix"].to(device)
            mag_labels = batch["mag_labels"].to(device)
            mod_labels = batch["mod_labels"].to(device)
            
            # Forward pass
            outputs = model(
                mag_inputs=mag_inputs,
                mod_inputs=mod_inputs,
                attention_mask=attention_mask,
                mag_labels=mag_labels,
                mod_labels=mod_labels,
                mask_matrix=mask_matrix
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            train_steps += 1
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log periodically
            if global_step % config.get("log_interval", 100) == 0:
                logger.info(f"Step {global_step}: loss={loss.item():.4f}")
        
        avg_train_loss = train_loss / train_steps
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                mag_inputs = batch["mag_inputs"].to(device)
                mod_inputs = batch["mod_inputs"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                mask_matrix = batch["mask_matrix"].to(device)
                mag_labels = batch["mag_labels"].to(device)
                mod_labels = batch["mod_labels"].to(device)
                
                outputs = model(
                    mag_inputs=mag_inputs,
                    mod_inputs=mod_inputs,
                    attention_mask=attention_mask,
                    mag_labels=mag_labels,
                    mod_labels=mod_labels,
                    mask_matrix=mask_matrix
                )
                
                loss = outputs["loss"]
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        # --- Checkpointing ---
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "config": config
        }
        
        # Save last model
        last_path = output_dir / "last_model.pt"
        torch.save(checkpoint, last_path)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ New best model! Validation loss: {best_val_loss:.4f}")
    
    # 6. Testing (Final Eval)
    logger.info(f"\n{'=' * 50}")
    logger.info("Final Evaluation on Test Set")
    logger.info(f"{'=' * 50}")
    
    # Load best model
    best_checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    
    test_loss = 0.0
    test_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            mag_inputs = batch["mag_inputs"].to(device)
            mod_inputs = batch["mod_inputs"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mask_matrix = batch["mask_matrix"].to(device)
            mag_labels = batch["mag_labels"].to(device)
            mod_labels = batch["mod_labels"].to(device)
            
            outputs = model(
                mag_inputs=mag_inputs,
                mod_inputs=mod_inputs,
                attention_mask=attention_mask,
                mag_labels=mag_labels,
                mod_labels=mod_labels,
                mask_matrix=mask_matrix
            )
            test_loss += outputs["loss"].item()
            test_steps += 1
    
    avg_test_loss = test_loss / test_steps if test_steps > 0 else float('inf')
    logger.info(f"Test loss: {avg_test_loss:.4f}")
    
    logger.info("Training Complete.")


def main():
    parser = argparse.ArgumentParser(description="Train IntSeqBERT model (Dual Stream)")
    
    # Data arguments
    parser.add_argument("--features_dir", type=str, required=True,
                        help="Directory containing .pt files (output of preprocess features)")
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="Path to metadata JSONL for filtering (optional)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for debugging")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    
    # Model arguments
    parser.add_argument("--mag_dim", type=int, default=5)
    parser.add_argument("--mod_dim", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Masking
    parser.add_argument("--mask_prob", type=float, default=0.15)
    
    args = parser.parse_args()
    config = vars(args)
    
    train(config)


if __name__ == "__main__":
    main()