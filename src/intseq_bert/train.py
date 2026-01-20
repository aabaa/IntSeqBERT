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


class TrainingLogger:
    """
    Structured logging for training runs.
    Creates config.json, history.csv, best_metrics.json, and train.log.
    """
    
    # Representative moduli for console output (uses config)
    REPRESENTATIVE_MODS = config.REPRESENTATIVE_MODS
    
    def __init__(
        self, 
        output_dir: Path, 
        args: argparse.Namespace,
        data_stats: Optional[Dict[str, int]] = None,
        resume: bool = False
    ):
        """
        Initialize training logger.
        
        Args:
            output_dir: Directory to save logs
            args: Command line arguments
            data_stats: Optional dict with train_samples, val_samples, test_samples
            resume: If True, append to existing logs instead of overwriting
        """
        import json
        import csv
        import subprocess
        from datetime import datetime
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        
        # File paths
        self.config_path = self.output_dir / "config.json"
        self.history_path = self.output_dir / "history.csv"
        self.best_metrics_path = self.output_dir / "best_metrics.json"
        self.log_path = self.output_dir / "train.log"
        
        # 1. Create config.json (skip if resume and exists)
        if not (resume and self.config_path.exists()):
            self._save_config(args, data_stats)
        
        # 2. Create CSV header if needed
        if not self.history_path.exists():
            self._create_csv_header()
        
        # 3. Setup file handler for train.log
        self._setup_file_handler()
    
    def _save_config(
        self, 
        args: argparse.Namespace, 
        data_stats: Optional[Dict[str, int]]
    ) -> None:
        """Save experiment configuration to config.json."""
        import json
        import subprocess
        import platform
        from datetime import datetime
        
        # Try to get git hash
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.output_dir,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_hash = None
        
        config_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "args": vars(args),
            "loss_weights": {
                "mag": config.LOSS_WEIGHT_MAG,
                "sign": config.LOSS_WEIGHT_SIGN,
                "mod": config.LOSS_WEIGHT_MOD
            },
            "environment": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                "git_hash": git_hash
            },
            "data_stats": data_stats or {},
            "resume_from": getattr(args, "resume", None)
        }
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def _create_csv_header(self) -> None:
        """Create history.csv with header row."""
        import csv
        
        headers = self.get_csv_headers()
        
        with open(self.history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _setup_file_handler(self) -> None:
        """Add file handler to logger."""
        mode = "a" if self.resume else "w"
        file_handler = logging.FileHandler(self.log_path, mode=mode, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        file_handler.setLevel(logging.INFO)
        
        # Avoid adding duplicate handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        
        logger.addHandler(file_handler)
    
    def log_epoch(self, epoch_data: Dict) -> None:
        """
        Append one row to history.csv.
        
        Args:
            epoch_data: Dictionary containing all epoch metrics.
                Required keys: epoch, lr, time_sec, is_best, early_stop_counter,
                              train_loss, val_loss, val_mag_acc, val_mag_mse,
                              val_sign_acc, val_mod_acc, val_mod_loss,
                              mod_accuracies (list of 100 floats),
                              w_mag, w_sign, w_mod
        """
        import csv
        
        row = [
            epoch_data["epoch"],
            epoch_data["lr"],
            epoch_data.get("time_sec", 0),
            epoch_data.get("is_best", False),
            epoch_data.get("early_stop_counter", 0),
            epoch_data["train_loss"],
            epoch_data["val_loss"],
            epoch_data["val_mag_acc"],
            epoch_data.get("val_mag_mse", 0),
            epoch_data["val_sign_acc"],
            epoch_data["val_mod_acc"],
            epoch_data.get("val_mod_loss", 0)
        ]
        
        # Add per-mod accuracies
        mod_accuracies = epoch_data.get("mod_accuracies", [0.0] * len(config.MOD_RANGE))
        row.extend(mod_accuracies)
        
        # Add weights
        row.extend([
            epoch_data.get("w_mag", config.LOSS_WEIGHT_MAG),
            epoch_data.get("w_sign", config.LOSS_WEIGHT_SIGN),
            epoch_data.get("w_mod", config.LOSS_WEIGHT_MOD)
        ])
        
        with open(self.history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def save_best_metrics(self, metrics: Dict) -> None:
        """
        Save best model metrics to best_metrics.json.
        
        Args:
            metrics: Dictionary with validation metrics and mod_accuracies
        """
        import json
        from datetime import datetime
        
        # Extract representative mod accuracies
        mod_accuracies = metrics.get("mod_accuracies", [])
        representative_mods = {}
        
        for mod in self.REPRESENTATIVE_MODS:
            idx = self._mod_to_index(mod)
            if idx is not None and idx < len(mod_accuracies):
                representative_mods[f"mod_{mod}"] = mod_accuracies[idx]
        
        best_data = {
            "best_epoch": metrics.get("epoch", 0),
            "val_loss": metrics.get("val_loss", 0),
            "val_mag_acc": metrics.get("val_mag_acc", 0),
            "val_mag_mse": metrics.get("val_mag_mse", 0),
            "val_sign_acc": metrics.get("val_sign_acc", 0),
            "val_mod_acc": metrics.get("val_mod_acc", 0),
            "val_mod_loss": metrics.get("val_mod_loss", 0),
            "representative_mods": representative_mods,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.best_metrics_path, "w", encoding="utf-8") as f:
            json.dump(best_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def get_representative_mod_indices() -> list:
        """
        Get indices into MOD_RANGE for representative moduli.
        
        Returns:
            List of indices for mods [2,3,5,7,10,100,101]
        """
        indices = []
        for mod in TrainingLogger.REPRESENTATIVE_MODS:
            try:
                idx = config.MOD_RANGE.index(mod)
                indices.append(idx)
            except ValueError:
                pass  # mod not in range
        return indices
    
    @staticmethod
    def _mod_to_index(mod: int) -> Optional[int]:
        """Convert modulus value to index in MOD_RANGE."""
        try:
            return config.MOD_RANGE.index(mod)
        except ValueError:
            return None
    
    @staticmethod
    def get_csv_headers() -> list:
        """
        Get CSV column headers for history.csv / test_results.csv.
        
        Returns:
            List of column header strings.
        """
        headers = [
            "epoch", "lr", "time_sec", "is_best", "early_stop_counter",
            "train_loss", "val_loss",
            "val_mag_acc", "val_mag_mse", "val_sign_acc",
            "val_mod_acc", "val_mod_loss"
        ]
        
        # Add per-mod accuracy columns (mod_acc_2 through mod_acc_101)
        for m in config.MOD_RANGE:
            headers.append(f"mod_acc_{m}")
        
        # Add weight columns
        headers.extend(["w_mag", "w_sign", "w_mod"])
        
        return headers

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
    # Move IntSeq inputs to device
    mag_features = batch["mag_inputs"].to(device)
    mod_features = batch["mod_inputs"].to(device)
    
    # Move Vanilla inputs to device
    token_ids = batch["token_ids"].to(device)
    token_labels = batch["token_labels"].to(device)
    
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
        # IntSeqBERT inputs
        "mag_features": mag_features,
        "mod_features": mod_features,
        # Vanilla inputs
        "token_ids": token_ids,
        "token_labels": token_labels,
        # Common
        "src_key_padding_mask": src_key_padding_mask,
        "labels": {
            "mag_targets": mag_targets,
            "sign_targets": sign_targets,
            "mod_targets": mod_targets,
            "token_targets": token_labels,  # For Vanilla LM loss
            "mask_map": mask_matrix
        }
    }


# ==========================================
# Evaluation Logic
# ==========================================

def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device,
    model_type: str = "intseq"
) -> Dict[str, float]:
    """
    Run validation loop and calculate metrics.
    All accuracy metrics are returned as percentages (0-100).
    
    Returns:
        Dict containing:
        - val_loss, mag_mse, mag_acc, sign_acc, mod_acc, mod_loss
        - mod_accuracies: List of 100 floats, one per modulus (2-101)
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Metric Accumulators
    total_mask_count = 0
    
    correct_sign = 0
    correct_mag_thresh = 0 # |diff| < threshold
    sum_mag_mse = 0.0
    sum_mod_loss = 0.0  # Accumulated normalized mod loss
    
    # Per-mod accuracy tracking (100 moduli)
    correct_mod_per_mod = [0] * config.NUM_MODULI
    
    with torch.no_grad():
        for raw_batch in tqdm(dataloader, desc="Validating", leave=False):
            # Prepare data
            inputs = prepare_labels(raw_batch, device)
            labels = inputs["labels"]
            mask_map = labels["mask_map"]
            
            if not mask_map.any():
                continue

            # Forward (FP32 - AMP disabled due to FP16 overflow issues)
            if model_type == "intseq" or model_type == "ablation":
                outputs = model(
                    mag_features=inputs["mag_features"],
                    mod_features=inputs["mod_features"],
                    src_key_padding_mask=inputs["src_key_padding_mask"],
                    labels=labels
                )
            elif model_type == "vanilla":
                outputs = model(
                    input_ids=inputs["token_ids"],
                    src_key_padding_mask=inputs["src_key_padding_mask"],
                    labels=labels
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
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
            
            # 3. Modulo Accuracy (Per-mod tracking)
            # Access helper method (handle DataParallel if needed)
            raw_model = model.module if hasattr(model, "module") else model
            mod_logits_split = raw_model._split_mod_logits(preds["mod_logits"]) # List of (B, L, m)
            
            # Vectorized calculation per modulus
            for i, m_logits in enumerate(mod_logits_split):
                # Slice and mask
                m_pred = torch.argmax(m_logits, dim=-1)[mask_map]
                m_target = labels["mod_targets"][:, :, i][mask_map]
                
                correct_mod_per_mod[i] += (m_pred == m_target).sum().item()
            
            # Get normalized mod loss from model output
            if "loss_breakdown" in outputs:
                sum_mod_loss += outputs["loss_breakdown"]["raw_mod"].item()
            
            # Update counts
            n_masked = mask_map.sum().item()
            total_mask_count += n_masked

    # Calculate Averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    if total_mask_count > 0:
        mag_mse = sum_mag_mse / total_mask_count
        mag_acc = (correct_mag_thresh / total_mask_count) * 100
        sign_acc = (correct_sign / total_mask_count) * 100
        mod_loss = sum_mod_loss / num_batches if num_batches > 0 else 0.0
        
        # Per-mod accuracies (as percentages)
        mod_accuracies = [(correct / total_mask_count) * 100 for correct in correct_mod_per_mod]
        
        # Overall mod accuracy is mean of per-mod accuracies
        mod_acc = sum(mod_accuracies) / len(mod_accuracies) if mod_accuracies else 0.0
    else:
        mag_mse, mag_acc, sign_acc, mod_acc, mod_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        mod_accuracies = [0.0] * config.NUM_MODULI
        
    return {
        "val_loss": avg_loss,
        "mag_mse": mag_mse,
        "mag_acc": mag_acc,
        "sign_acc": sign_acc,
        "mod_acc": mod_acc,
        "mod_loss": mod_loss,  # Normalized mod loss (1.0 = random prediction)
        "mod_accuracies": mod_accuracies  # List of 100 per-mod accuracies
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
    
    # Initialize Training Logger
    data_stats = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset)
    }
    training_logger = TrainingLogger(
        output_dir=output_dir,
        args=args,
        data_stats=data_stats,
        resume=args.resume is not None
    )
    
    # 3. Model Initialization
    logger.info(f"Initializing model (type={args.model_type})...")
    if args.model_type == "intseq":
        model = models.IntSeqForPreTraining(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers
        )
    elif args.model_type == "vanilla":
        model = models.VanillaTransformerForPreTraining(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers
        )
    elif args.model_type == "ablation":
        model = models.AblationForPreTraining(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    
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
    best_val_loss = float("inf")
    
    # 5. Training Loop
    logger.info("Starting training...")
    
    import time
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, raw_batch in enumerate(progress_bar):
            # Prepare inputs & labels
            inputs = prepare_labels(raw_batch, device)
            
            # Forward - dispatch based on model type
            # NOTE: AMP disabled due to FP16 overflow with extreme log values (up to 210)
            if args.model_type == "intseq" or args.model_type == "ablation":
                outputs = model(
                    mag_features=inputs["mag_features"],
                    mod_features=inputs["mod_features"],
                    src_key_padding_mask=inputs["src_key_padding_mask"],
                    labels=inputs["labels"]
                )
            elif args.model_type == "vanilla":
                outputs = model(
                    input_ids=inputs["token_ids"],
                    src_key_padding_mask=inputs["src_key_padding_mask"],
                    labels=inputs["labels"]
                )
            else:
                raise ValueError(f"Unknown model_type: {args.model_type}")
            loss = outputs["loss"] / args.accum_steps
            
            # Backward (FP32)
            loss.backward()
            
            if (step + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            current_loss = loss.item() * args.accum_steps
            train_loss += current_loss
            
            progress_bar.set_postfix({
                "loss": f"{current_loss:.3f}"
            })
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        logger.info(f"Validating Epoch {epoch+1}...")
        val_metrics = evaluate(model, val_loader, device, args.model_type)
        
        epoch_time = time.time() - epoch_start_time
        
        # Determine if this is the best epoch
        is_best = val_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_loss"]
        
        # Log Metrics to console
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
        logger.info(f"  Mag Acc:    {val_metrics['mag_acc']:.2f}% (MSE: {val_metrics['mag_mse']:.4f})")
        logger.info(f"  Sign Acc:   {val_metrics['sign_acc']:.2f}%")
        logger.info(f"  Mod Acc:    {val_metrics['mod_acc']:.2f}% (Mean of {config.NUM_MODULI} mods)")
        
        # Display representative mod accuracies
        mod_accs = val_metrics.get("mod_accuracies", [])
        if mod_accs:
            rep_indices = TrainingLogger.get_representative_mod_indices()
            rep_strs = []
            for idx in rep_indices:
                if idx < len(mod_accs):
                    mod_val = config.MOD_RANGE[idx]
                    rep_strs.append(f"mod{mod_val}: {mod_accs[idx]:.1f}%")
            logger.info(f"    {', '.join(rep_strs)}")
        
        # Log to CSV via TrainingLogger
        epoch_data = {
            "epoch": epoch + 1,
            "lr": scheduler.get_last_lr()[0],
            "time_sec": epoch_time,
            "is_best": is_best,
            "early_stop_counter": early_stopping.counter,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_mag_acc": val_metrics["mag_acc"],
            "val_mag_mse": val_metrics["mag_mse"],
            "val_sign_acc": val_metrics["sign_acc"],
            "val_mod_acc": val_metrics["mod_acc"],
            "val_mod_loss": val_metrics["mod_loss"],
            "mod_accuracies": val_metrics.get("mod_accuracies", [0.0] * config.NUM_MODULI),
            "w_mag": config.LOSS_WEIGHT_MAG,
            "w_sign": config.LOSS_WEIGHT_SIGN,
            "w_mod": config.LOSS_WEIGHT_MOD
        }
        training_logger.log_epoch(epoch_data)
        
        # --- Checkpointing ---
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_metrics["val_loss"],
            "val_metrics": val_metrics,
            "config": vars(args)
        }
        
        # Save Last
        torch.save(state, output_dir / "last_checkpoint.pt")
        
        # Save Best & Early Stopping check
        should_stop = early_stopping(val_metrics["val_loss"])
        
        if is_best:
            torch.save(state, output_dir / "best_model.pt")
            logger.info(f"New best model saved! (Loss: {val_metrics['val_loss']:.4f})")
            
            # Save best metrics JSON
            best_metrics_data = {
                "epoch": epoch + 1,
                "val_loss": val_metrics["val_loss"],
                "val_mag_acc": val_metrics["mag_acc"],
                "val_mag_mse": val_metrics["mag_mse"],
                "val_sign_acc": val_metrics["sign_acc"],
                "val_mod_acc": val_metrics["mod_acc"],
                "val_mod_loss": val_metrics["mod_loss"],
                "mod_accuracies": val_metrics.get("mod_accuracies", [])
            }
            training_logger.save_best_metrics(best_metrics_data)
        
        if should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    logger.info("Training complete.")


# ==========================================
# Test-Only Mode Functions
# ==========================================

def _load_model_config(model_path: Path, checkpoint: Dict, args) -> Dict:
    """
    Load model configuration with fallback priority.
    
    Priority:
    1. checkpoint['config'] (strongest)
    2. config.json in same directory
    3. args (fallback)
    
    Args:
        model_path: Path to the model checkpoint file
        checkpoint: Loaded checkpoint dictionary
        args: Command line arguments
    
    Returns:
        Dictionary with d_model, nhead, num_layers
    """
    import json
    
    # 1. Try checkpoint config first
    ckpt_config = checkpoint.get("config", {})
    if ckpt_config:
        logger.info("Using config from checkpoint")
        return {
            "d_model": ckpt_config.get("d_model", args.d_model),
            "nhead": ckpt_config.get("nhead", args.nhead),
            "num_layers": ckpt_config.get("num_layers", args.num_layers)
        }
    
    # 2. Try config.json in same directory
    config_path = model_path.parent / "config.json"
    if config_path.exists():
        logger.info(f"Using config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)
            saved_args = saved_config.get("args", {})
            return {
                "d_model": saved_args.get("d_model", args.d_model),
                "nhead": saved_args.get("nhead", args.nhead),
                "num_layers": saved_args.get("num_layers", args.num_layers)
            }
    
    # 3. Fallback to args (with warning)
    logger.warning("No saved config found. Using command-line args (may cause size mismatch).")
    return {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers
    }


def _save_test_csv(path: Path, metrics: Dict, time_sec: float) -> None:
    """Save test results in history.csv compatible format."""
    import csv
    
    # Use shared header definition for consistency
    headers = TrainingLogger.get_csv_headers()
    
    row = [
        0,          # epoch (test mode marker)
        0.0,        # lr
        time_sec,
        True,       # is_best
        0,          # early_stop_counter
        0.0,        # train_loss (N/A)
        metrics["val_loss"],
        metrics["mag_acc"],
        metrics["mag_mse"],
        metrics["sign_acc"],
        metrics["mod_acc"],
        metrics["mod_loss"]
    ]
    row.extend(metrics.get("mod_accuracies", [0.0] * config.NUM_MODULI))
    row.extend([config.LOSS_WEIGHT_MAG, config.LOSS_WEIGHT_SIGN, config.LOSS_WEIGHT_MOD])
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)


def _save_test_json(
    path: Path, 
    args, 
    metrics: Dict, 
    time_sec: float,
    num_samples: int
) -> None:
    """Save detailed test metrics as JSON."""
    import json
    from datetime import datetime
    
    mod_accs = metrics.get("mod_accuracies", [])
    rep_mods = {}
    for mod in config.REPRESENTATIVE_MODS:
        idx = TrainingLogger._mod_to_index(mod)
        if idx is not None and idx < len(mod_accs):
            rep_mods[f"mod_{mod}"] = mod_accs[idx]
    
    data = {
        "model_path": args.model_path,
        "split_type": args.split_type,
        "test_samples": num_samples,
        "evaluation_time_sec": time_sec,
        "test_loss": metrics["val_loss"],
        "test_mag_acc": metrics["mag_acc"],
        "test_mag_mse": metrics["mag_mse"],
        "test_sign_acc": metrics["sign_acc"],
        "test_mod_acc": metrics["mod_acc"],
        "representative_mods": rep_mods,
        "all_mod_accuracies": mod_accs,
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def test_only(args):
    """Test-only mode: evaluate pre-trained model on test/val/train data."""
    import time
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Header
    logger.info("=" * 40)
    logger.info("Test-Only Mode Evaluation")
    logger.info("=" * 40)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Split: {args.split_type} ({args.test_split})")
    
    # 1. Load Checkpoint
    logger.info("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 2. Restore model config with fallback priority
    model_config = _load_model_config(Path(args.model_path), checkpoint, args)
    model_type = getattr(args, "model_type", "intseq")
    
    if model_type == "intseq":
        model = models.IntSeqForPreTraining(
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_layers=model_config["num_layers"]
        )
    elif model_type == "vanilla":
        model = models.VanillaTransformerForPreTraining(
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_layers=model_config["num_layers"]
        )
    elif model_type == "ablation":
        model = models.AblationForPreTraining(
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_layers=model_config["num_layers"]
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # 3. Load Dataset (using --test_split argument)
    logger.info(f"Loading {args.test_split} dataset...")
    test_dataset = loader.load_dataset(
        split_type=args.split_type,
        split_name=args.test_split,  # train / val / test
        data_root=args.data_root
    )
    logger.info(f"Samples: {len(test_dataset)}")
    
    collator_fn = collator.OEISCollator(mask_prob=config.MASK_PROB)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator_fn,
        pin_memory=True
    )
    
    # 4. Evaluate
    logger.info("-" * 40)
    start_time = time.time()
    test_metrics = evaluate(model, test_loader, device, args.model_type)
    eval_time = time.time() - start_time
    
    # 5. Log Results
    logger.info(f"Loss:        {test_metrics['val_loss']:.4f}")
    logger.info(f"Mag Acc:     {test_metrics['mag_acc']:.2f}% (MSE: {test_metrics['mag_mse']:.4f})")
    logger.info(f"Sign Acc:    {test_metrics['sign_acc']:.2f}%")
    logger.info(f"Mod Acc:     {test_metrics['mod_acc']:.2f}% (Mean of {config.NUM_MODULI} mods)")
    
    # Representative mods
    mod_accs = test_metrics.get("mod_accuracies", [])
    if mod_accs:
        rep_indices = TrainingLogger.get_representative_mod_indices()
        rep_strs = []
        for idx in rep_indices:
            if idx < len(mod_accs):
                mod_val = config.MOD_RANGE[idx]
                rep_strs.append(f"mod{mod_val}: {mod_accs[idx]:.1f}%")
        logger.info(f"  {', '.join(rep_strs)}")
    
    # 6. Save Results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV output (using TrainingLogger.get_csv_headers())
    csv_path = Path(args.test_output) if args.test_output else output_dir / "test_results.csv"
    _save_test_csv(csv_path, test_metrics, eval_time)
    
    # JSON output
    json_path = output_dir / "test_metrics.json"
    _save_test_json(json_path, args, test_metrics, eval_time, len(test_dataset))
    
    logger.info("-" * 40)
    logger.info(f"Results saved to: {csv_path}")
    logger.info("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Train IntSeqBERT Encoder")
    
    # Path / Data
    parser.add_argument("--split_type", required=True, help="Split type (e.g., std, easy, all)")
    parser.add_argument("--data_root", type=str, default=config.DATA_ROOT, help="Path to data root")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    
    # Model Config
    parser.add_argument("--model_type", type=str, default="intseq",
                       choices=["intseq", "vanilla"],
                       help="Model type: intseq (IntSeqBERT) or vanilla (Vanilla Transformer)")
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
    
    # Test-Only Mode
    parser.add_argument("--test_only", action="store_true",
                       help="Run evaluation on test data only (skip training)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model checkpoint for test-only mode")
    parser.add_argument("--test_split", type=str, default="test",
                       help="Split name to evaluate (train/val/test)")
    parser.add_argument("--test_output", type=str, default=None,
                       help="Custom output path for test results CSV")
    
    args = parser.parse_args()
    
    # Validation for test-only mode
    if args.test_only and not args.model_path:
        parser.error("--test_only requires --model_path")
    if args.model_path and not args.test_only:
        parser.error("--model_path requires --test_only flag")
    
    # Dispatch
    if args.test_only:
        test_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
