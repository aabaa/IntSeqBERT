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
    
    # Use DualStreamCollator (same as BERT training)
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
            
            # Skip if no masks (unlikely with p=0.15)
            if mask_matrix.sum() == 0:
                continue

            # 1. Encoder Forward (Get Latent Vectors)
            with torch.no_grad():
                enc_out = encoder(mag_inputs, mod_inputs, attn_mask)
                full_latent = enc_out["encoded_state"] # (B, L, D)

            # 2. Extract Latent Vectors for MASKED positions only
            # We flatten everything to train on individual tokens
            flat_mask = mask_matrix.view(-1)      # (B*L)
            flat_latent = full_latent.view(-1, full_latent.size(-1)) # (B*L, D)
            
            # Select only the vectors corresponding to [MASK] tokens
            masked_latent = flat_latent[flat_mask] # (N_masked, D)

            # 3. Prepare Targets for MASKED positions only
            # batch['targets'] contains full sequence targets (padded with 0 or -100)
            masked_targets = {}
            for k, v in batch["targets"].items():
                v = v.to(device).view(-1) # Flatten (B*L)
                masked_targets[k] = v[flat_mask] # Select masked
            
            # 4. Decoder Forward
            predictions = decoder(masked_latent)
            
            # 5. Loss
            loss = decoder.compute_loss(predictions, masked_targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / max(1, train_steps)
        logger.info(f"Avg Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation ---
        decoder.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                mag_inputs = batch["mag_inputs"].to(device)
                mod_inputs = batch["mod_inputs"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                mask_matrix = batch["mask_matrix"].to(device)
                
                if mask_matrix.sum() == 0: continue
                
                # Encoder
                enc_out = encoder(mag_inputs, mod_inputs, attn_mask)
                full_latent = enc_out["encoded_state"]
                
                # Filter Masked
                flat_mask = mask_matrix.view(-1)
                masked_latent = full_latent.view(-1, full_latent.size(-1))[flat_mask]
                
                masked_targets = {}
                for k, v in batch["targets"].items():
                    masked_targets[k] = v.to(device).view(-1)[flat_mask]
                
                # Decoder
                predictions = decoder(masked_latent)
                loss = decoder.compute_loss(predictions, masked_targets)
                
                val_loss += loss.item()
                val_steps += 1
                
        avg_val_loss = val_loss / max(1, val_steps)
        logger.info(f"Avg Val Loss: {avg_val_loss:.4f}")
        
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
    
    # Metadata filtering (optional)
    parser.add_argument("--metadata_path", type=str, default=None)
    
    args = parser.parse_args()
    train(vars(args))

if __name__ == "__main__":
    main()
