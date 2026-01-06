"""
Training script for IntSeqDecoder (Solver).
Uses a frozen pre-trained IntSeqBERT encoder to generate latent vectors,
then trains the decoder to solve for the next integer using Beam Search CRT targets.
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
from . import decoder_model
from .decoder_model import MOD_RANGE


def setup_logging(output_dir: Path) -> logging.Logger:
    log_file = output_dir / "train_decoder.log"
    logger = logging.getLogger("intseq_bert.train_decoder")
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def evaluate(
    decoder: nn.Module,
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    num_reconstruction_samples: int = 50
) -> float:
    """Run detailed evaluation: Loss, Accuracy per head, and CRT Reconstruction check."""
    decoder.eval()
    total_loss = 0.0
    steps = 0
    
    # Metrics counters
    correct_counts = {f"mod{m}": 0 for m in MOD_RANGE}
    correct_counts["sign"] = 0
    total_tokens = 0
    
    # Reconstruction counters (CRT Check)
    rec_perfect = 0
    rec_rescued = 0
    rec_failed = 0
    rec_evaluated = 0
    
    pbar = tqdm(dataloader, desc="Evaluation", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            mag_inputs = batch["mag_inputs"].to(device)
            mod_inputs = batch["mod_inputs"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            mask_matrix = batch["mask_matrix"].to(device)
            
            if mask_matrix.sum() == 0: continue

            # 1. Encoder Forward
            enc_out = encoder(mag_inputs, mod_inputs, attn_mask)
            full_latent = enc_out["encoded_state"]

            # 2. Filter Masked Positions
            flat_mask = mask_matrix.view(-1)
            masked_latent = full_latent.view(-1, full_latent.size(-1))[flat_mask]
            
            masked_targets = {}
            for k, v in batch["targets"].items():
                masked_targets[k] = v.to(device).view(-1)[flat_mask]
            
            # 3. Decoder Forward
            predictions = decoder(masked_latent)
            
            # 4. Compute Loss
            loss = decoder.compute_loss(predictions, masked_targets)
            total_loss += loss.item()
            steps += 1
            
            # 5. Compute Accuracies
            batch_tokens = masked_latent.size(0)
            total_tokens += batch_tokens
            
            # Mod Accuracies
            for m in MOD_RANGE:
                key = f"mod{m}"
                if key in predictions and key in masked_targets:
                    pred_idx = predictions[key].argmax(dim=1)
                    tgt_idx = masked_targets[key]
                    valid_mask = tgt_idx != -100
                    if valid_mask.sum() > 0:
                        correct_counts[key] += (pred_idx[valid_mask] == tgt_idx[valid_mask]).sum().item()

            # 6. CRT Reconstruction Check (Sample subset)
            if rec_evaluated < num_reconstruction_samples:
                for i in range(min(batch_tokens, num_reconstruction_samples - rec_evaluated)):
                    single_pred = {k: v[i:i+1] for k, v in predictions.items()}
                    results = decoder.beam_search_solve(single_pred, beam_width=10)
                    
                    if not results:
                        rec_failed += 1
                    else:
                        best_int, _ = results[0]
                        # Verify consistency with high-confidence mods (100, 101)
                        is_correct = True
                        for check_m in [100, 101]:
                            key = f"mod{check_m}"
                            if key in masked_targets:
                                true_rem = masked_targets[key][i].item()
                                if true_rem != -100 and best_int % check_m != true_rem:
                                    is_correct = False; break
                        
                        if is_correct: rec_perfect += 1
                        else: rec_failed += 1
                    rec_evaluated += 1

    avg_loss = total_loss / max(1, steps)
    
    # Logging
    logger.info(f" Evaluation Loss: {avg_loss:.4f}")
    if total_tokens > 0:
        mod_accs = []
        for m in [3, 7, 100, 101]:
            acc = correct_counts[f"mod{m}"] / total_tokens * 100
            mod_accs.append(f"Mod{m}:{acc:.1f}%")
        logger.info(f" Head Accuracies: {' | '.join(mod_accs)}")
        
    if rec_evaluated > 0:
        success_rate = (rec_perfect + rec_rescued) / rec_evaluated * 100
        logger.info(f" CRT Reconstruction (n={rec_evaluated}): Success ~{success_rate:.1f}%")

    return avg_loss


def train(config: Dict[str, Any]) -> None:
    output_dir = Path(config.get("output_dir", "decoder_checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("=" * 50)
    logger.info("Starting IntSeqDecoder Training (Dual Stream)")
    logger.info("=" * 50)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load Pre-trained Encoder (Frozen)
    logger.info(f"Loading pre-trained encoder from {config['encoder_checkpoint']}...")
    encoder, _ = bert_model.IntSeqBERT.load_from_checkpoint(
        config['encoder_checkpoint'], device=str(device)
    )
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    logger.info("Encoder loaded and frozen.")
    
    # 2. Initialize Decoder
    logger.info("Initializing Decoder...")
    decoder = decoder_model.IntSeqDecoder(
        d_model=encoder.d_model, # Match encoder dimension
        hidden_dim=config.get("hidden_dim", 512),
        dropout=config.get("dropout", 0.1)
    )
    decoder = decoder.to(device)
    
    # 3. Data Loading
    logger.info("Loading data...")
    train_ds, val_ds, test_ds = loader.load_and_split_data(
        features_dir=config["features_dir"],
        metadata_path=config.get("metadata_path"),
        val_ratio=config.get("val_ratio", 0.05),
        test_ratio=config.get("test_ratio", 0.05),
        seed=config.get("seed", 42),
        max_samples=config.get("max_samples")
    )
    
    # Use DualStreamCollator
    data_collator = collator.DualStreamCollator(
        mask_prob=config.get("mask_prob", 0.15)
    )
    
    num_workers = config.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, 
                              collate_fn=data_collator, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, 
                            collate_fn=data_collator, num_workers=num_workers)
    
    # 4. Optimization
    optimizer = AdamW(decoder.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    epochs = config["epochs"]
    
    # 5. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")
        
        # --- Train ---
        decoder.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            mag_inputs = batch["mag_inputs"].to(device)
            mod_inputs = batch["mod_inputs"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            mask_matrix = batch["mask_matrix"].to(device)
            
            if mask_matrix.sum() == 0: continue

            with torch.no_grad():
                enc_out = encoder(mag_inputs, mod_inputs, attn_mask)
                full_latent = enc_out["encoded_state"]

            # Extract Latent Vectors & Targets
            flat_mask = mask_matrix.view(-1)
            masked_latent = full_latent.view(-1, full_latent.size(-1))[flat_mask]
            
            masked_targets = {}
            for k, v in batch["targets"].items():
                masked_targets[k] = v.to(device).view(-1)[flat_mask]
            
            # Forward & Loss
            predictions = decoder(masked_latent)
            loss = decoder.compute_loss(predictions, masked_targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            # Show live accuracy for mod100
            with torch.no_grad():
                if "mod100" in predictions and "mod100" in masked_targets:
                    pred_100 = predictions["mod100"].argmax(dim=1)
                    tgt_100 = masked_targets["mod100"]
                    mask = tgt_100 != -100
                    acc = 0.0
                    if mask.sum() > 0:
                        acc = (pred_100[mask] == tgt_100[mask]).float().mean().item()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc100": f"{acc:.2%}"})
            
        avg_train_loss = train_loss / max(1, train_steps)
        logger.info(f"Avg Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation & Evaluation ---
        avg_val_loss = evaluate(
            decoder, encoder, val_loader, device, logger, 
            num_reconstruction_samples=50
        )
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(decoder.state_dict(), output_dir / "best_decoder.pt")
            logger.info("Saved best decoder.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, required=True, help="Path to .pt files")
    parser.add_argument("--encoder_checkpoint", type=str, required=True, help="Path to best_model.pt from train_bert")
    parser.add_argument("--output_dir", type=str, default="decoder_checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--metadata_path", type=str, default=None)
    
    # Model Config
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    
    args = parser.parse_args()
    train(vars(args))

if __name__ == "__main__":
    main()
