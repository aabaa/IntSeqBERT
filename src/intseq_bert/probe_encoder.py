"""
Diagnostic Script: Linear Probing for Encoder
Tests if the frozen encoder embeddings contain linearly separable Mod information.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from . import bert_model, loader, collator
from .decoder_model import MOD_RANGE

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class LinearProbe(nn.Module):
    """Simple Linear Classifiers for each Modulo task"""
    def __init__(self, input_dim):
        super().__init__()
        # Create a separate linear head for each mod task
        self.heads = nn.ModuleDict({
            f"mod{m}": nn.Linear(input_dim, m) for m in MOD_RANGE
        })

    def forward(self, x):
        return {k: head(x) for k, head in self.heads.items()}

def run_probe(features_dir, encoder_path, max_samples=5000, batch_size=64, epochs=10, unfreeze=False): # epochs増やし、unfreeze追加
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Encoder
    logger.info("Loading Encoder...")
    encoder, _ = bert_model.IntSeqBERT.load_from_checkpoint(encoder_path, device=str(device))
    
    if unfreeze:
        logger.info("MODE: Encoder Unfrozen (Testing limits)")
        encoder.train()
        for p in encoder.parameters(): p.requires_grad = True
    else:
        logger.info("MODE: Encoder Frozen (Linear Probe)")
        encoder.eval()
        for p in encoder.parameters(): p.requires_grad = False
    
    # 2. Init Linear Probes
    probe = LinearProbe(encoder.d_model).to(device)
    
    # Optimizer (Encoderも含めるか分岐)
    if unfreeze:
        params = list(probe.parameters()) + list(encoder.parameters())
        lr = 5e-5 # Encoderを壊さないよう低学習率
    else:
        params = probe.parameters()
        lr = 1e-3
    
    optimizer = AdamW(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Load Data (Subset for quick diagnosis)
    logger.info(f"Loading {max_samples} samples for diagnosis...")
    ds, _, _ = loader.load_and_split_data(
        features_dir, seed=42, max_samples=max_samples, val_ratio=0.1, test_ratio=0.0
    )
    collate = collator.DualStreamCollator(mask_prob=0.15)
    loader_iter = DataLoader(ds, batch_size=batch_size, collate_fn=collate)
    
    # 4. Training Loop
    logger.info("Starting Linear Probing...")
    for epoch in range(1, epochs + 1):
        probe.train()
        total_acc = {m: 0 for m in [3, 7, 100]}
        total_counts = 0
        
        for batch in tqdm(loader_iter, desc=f"Probe Epoch {epoch}"):
            mag = batch["mag_inputs"].to(device)
            mod = batch["mod_inputs"].to(device)
            mask = batch["attention_mask"].to(device)
            mask_matrix = batch["mask_matrix"].to(device)
            
            if mask_matrix.sum() == 0: continue
            
            # Extract Embeddings (Frozen)
            with torch.no_grad():
                enc_out = encoder(mag, mod, mask)
                full_latent = enc_out["encoded_state"]
            
            # Filter Masked Vectors
            flat_mask = mask_matrix.view(-1)
            latent = full_latent.view(-1, full_latent.size(-1))[flat_mask]
            
            # Targets
            targets = {}
            for m in [3, 7, 100]: # Diagnosis on representative mods
                targets[f"mod{m}"] = batch["targets"][f"mod{m}"].to(device).view(-1)[flat_mask]
            
            # Probe Forward
            logits = probe(latent)
            
            loss = 0
            for m in [3, 7, 100]:
                valid = targets[f"mod{m}"] != -100
                if valid.sum() > 0:
                    l = criterion(logits[f"mod{m}"][valid], targets[f"mod{m}"][valid])
                    loss += l
                    
                    # Calc Accuracy
                    pred = logits[f"mod{m}"].argmax(dim=1)
                    acc = (pred[valid] == targets[f"mod{m}"][valid]).float().mean().item()
                    total_acc[m] += acc
            
            if isinstance(loss, torch.Tensor):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_counts += 1
            
        # Report
        log_str = f"Epoch {epoch}: "
        for m in [3, 7, 100]:
            log_str += f"Mod{m} Acc: {total_acc[m]/total_counts:.1%} | "
        logger.info(log_str)

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir")
    parser.add_argument("encoder_path")
    parser.add_argument("--unfreeze", action="store_true")
    args = parser.parse_args()
    
    # Usage: python -m intseq_bert.probe_encoder FEATURES_DIR ENCODER_PATH --unfreeze
    run_probe(args.features_dir, args.encoder_path, unfreeze=args.unfreeze)