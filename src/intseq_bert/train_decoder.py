"""
Training script for NumberTheoreticDecoder using frozen IntSeqBERT representations.
"""

import argparse
import json
import logging
import math  # Added for log10 in magnitude binning
import random
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from . import schemas
from .bert_model import IntSeqBERT
from .decoder_model import NumberTheoreticDecoder, inverse_magnitude
from .features import log_magnitude


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to console and file."""
    log_file = output_dir / "train_decoder.log"
    
    logger = logging.getLogger("intseq_bert.train_decoder")
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


def get_targets(integers: List[int]) -> Dict[str, torch.Tensor]:
    """
    Generate multi-task targets from list of integers.
    
    Args:
        integers: List of ground truth integers
    
    Returns:
        Dictionary with target tensors for each head:
            - sign: (batch,) LongTensor [0=neg, 1=zero, 2=pos]
            - mag: (batch,) LongTensor [bin indices 0-4095]
            - mod3/5/7/8/10/11/13/100: (batch,) LongTensor [residues]
    """
    from .decoder_model import NUM_MAGNITUDE_BINS, MAX_LOG_VALUE
    
    batch_size = len(integers)
    
    # Sign: map to [0, 1, 2]
    signs = torch.tensor([
        0 if x < 0 else (1 if x == 0 else 2)
        for x in integers
    ], dtype=torch.long)
    
    # Magnitude: Convert to bin indices (classification)
    mag_bins = []
    for x in integers:
        if x == 0:
            log_val = 0.0
        else:
            log_val = math.log10(abs(x))  # Use log10
        # Map to bin index
        bin_idx = int((log_val / MAX_LOG_VALUE) * NUM_MAGNITUDE_BINS)
        bin_idx = max(0, min(bin_idx, NUM_MAGNITUDE_BINS - 1))
        mag_bins.append(bin_idx)
    
    mag_bins = torch.tensor(mag_bins, dtype=torch.long)
    
    # Modulo residues (Python % already handles negatives correctly)
    mod3 = torch.tensor([x % 3 for x in integers], dtype=torch.long)
    mod5 = torch.tensor([x % 5 for x in integers], dtype=torch.long)
    mod7 = torch.tensor([x % 7 for x in integers], dtype=torch.long)    # NEW
    mod8 = torch.tensor([x % 8 for x in integers], dtype=torch.long)
    mod10 = torch.tensor([x % 10 for x in integers], dtype=torch.long)
    mod11 = torch.tensor([x % 11 for x in integers], dtype=torch.long)  # NEW
    mod13 = torch.tensor([x % 13 for x in integers], dtype=torch.long)  # NEW
    mod100 = torch.tensor([x % 100 for x in integers], dtype=torch.long)  # NEW
    
    return {
        "sign": signs,
        "mag": mag_bins,  # Now bin indices, not floats
        "mod3": mod3,
        "mod5": mod5,
        "mod7": mod7,     # NEW
        "mod8": mod8,
        "mod10": mod10,
        "mod11": mod11,   # NEW
        "mod13": mod13,   # NEW
        "mod100": mod100  # NEW
    }


class DecoderDataset(Dataset):
    """
    Dataset that returns both features and original integers.
    """
    def __init__(self, data_items: List[Dict]):
        """
        Args:
            data_items: List of dicts with 'features' and 'integers'
        """
        self.data = data_items
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def decoder_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for decoder training.
    
    Args:
        batch: List of dicts with 'features' (tensor) and 'integers' (list)
    
    Returns:
        Dict with:
            - masked_inputs: (batch, max_len, 35) padded tensor
            - attention_mask: (batch, max_len) padding mask
            - mask_indices: (batch,) positions that were masked
            - target_integers: List[int] ground truth integers
            - target_features: (batch, 35) features at masked positions
    """
    batch_size = len(batch)
    
    # Get sequence lengths
    seq_lens = [item['features'].shape[0] for item in batch]
    max_len = max(seq_lens)
    
    # Initialize tensors
    masked_inputs = torch.zeros(batch_size, max_len, 35)
    attention_mask = torch.zeros(batch_size, max_len)
    mask_indices = torch.zeros(batch_size, dtype=torch.long)
    target_integers = []
    
    for i, item in enumerate(batch):
        features = item['features']  # (seq_len, 35)
        integers = item['integers']  # List[int]
        seq_len = len(features)
        
        # Randomly select position to mask
        mask_idx = random.randint(0, seq_len - 1)
        mask_indices[i] = mask_idx
        
        # Get target integer at masked position
        target_integers.append(integers[mask_idx])
        
        # Create masked features (zero out selected position)
        masked_feat = features.clone()
        masked_feat[mask_idx] = 0.0
        
        # Fill in batch tensors
        masked_inputs[i, :seq_len] = masked_feat
        attention_mask[i, :seq_len] = 1
    
    # Collect target features for bypass mode (features at masked positions)
    target_features = torch.stack([
        batch[i]['features'][mask_indices[i].item()]
        for i in range(batch_size)
    ])  # (batch_size, 35)
    
    return {
        'masked_inputs': masked_inputs,
        'attention_mask': attention_mask,
        'mask_indices': mask_indices,
        'target_integers': target_integers,
        'target_features': target_features  # For bypass mode
    }


def load_decoder_data(
    features_path: str,
    jsonl_path: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    Load both features and original integers, align them, and split.
    
    Args:
        features_path: Path to features.pt file
        jsonl_path: Path to original JSONL data
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Load features
    features_dict = torch.load(features_path)
    
    # Load original integers from JSONL
    integers_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = schemas.OEISRecord.from_json_line(line)
                integers_dict[record.oeis_id] = record.sequence
            except Exception as e:
                continue
    
    # Create aligned dataset
    aligned_data = []
    for oeis_id, features in features_dict.items():
        if oeis_id in integers_dict:
            integers = integers_dict[oeis_id]
            # Ensure feature and integer lengths match
            min_len = min(len(features), len(integers))
            aligned_data.append({
                'oeis_id': oeis_id,
                'features': features[:min_len],
                'integers': integers[:min_len]
            })
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(aligned_data)
    
    total = len(aligned_data)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    
    test_data = aligned_data[:test_size]
    val_data = aligned_data[test_size:test_size + val_size]
    train_data = aligned_data[test_size + val_size:]
    
    return (
        DecoderDataset(train_data),
        DecoderDataset(val_data),
        DecoderDataset(test_data)
    )


def evaluate_decoder(
    bert_model: IntSeqBERT,
    decoder: NumberTheoreticDecoder,
    dataloader: DataLoader,
    device: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Evaluate decoder with quantitative and qualitative metrics.
    
    Returns dict with:
        - mag_mae: Magnitude mean absolute error
        - sign_acc, mod3_acc, etc.: Classification accuracies
        - perfect_count: Reconstructions that matched exactly
        - rescued_count: Wrong magnitude but CRT saved it
        - failed_count: Reconstruction failed
    """
    decoder.eval()
    
    total_mag_correct = 0
    correct_counts = {
        'sign': 0, 'mod3': 0, 'mod5': 0, 'mod7': 0, 'mod8': 0,
        'mod10': 0, 'mod11': 0, 'mod13': 0, 'mod100': 0
    }
    total_samples = 0
    
    # CRT reconstruction tracking
    perfect_count = 0
    rescued_count = 0
    failed_count = 0

    pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            masked_inputs = batch['masked_inputs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mask_indices = batch['mask_indices'].to(device)
            target_integers = batch['target_integers']
            target_features = batch['target_features'].to(device)  # For bypass mode
            
            batch_size = masked_inputs.size(0)
            total_samples += batch_size
            
            # Get representations
            if bert_model is None:
                # Bypass mode: Use raw features directly
                bert_vectors = target_features
            else:
                # Standard mode: Pass through BERT
                bert_output = bert_model(masked_inputs, attention_mask)
                predictions = bert_output["prediction"]
                # Extract at masked positions
                bert_vectors = predictions[torch.arange(batch_size), mask_indices]
            
            # Decoder forward
            decoder_output = decoder(bert_vectors)
            targets = get_targets(target_integers)
            
            # Magnitude bin accuracy (classification)
            mag_pred_bins = decoder_output['mag'].argmax(dim=1).cpu()
            mag_target_bins = targets['mag']
            total_mag_correct += (mag_pred_bins == mag_target_bins).sum().item()
            
            # Classification accuracies for all heads
            for head in ['sign', 'mod3', 'mod5', 'mod7', 'mod8', 'mod10', 'mod11', 'mod13', 'mod100']:
                pred_classes = decoder_output[head].argmax(dim=1).cpu()
                correct_counts[head] += (pred_classes == targets[head]).sum().item()
            
            # CRT reconstruction evaluation (VECTORIZED!)
            # Use batch_reconstruct instead of per-sample loop
            reconstructed_ints, _ = decoder.batch_reconstruct(bert_vectors)
            reconstructed_ints = reconstructed_ints.cpu().numpy()
            
            # Categorize results
            for i in range(batch_size):
                true_int = target_integers[i]
                recon_int = int(reconstructed_ints[i])
                
                # Compute magnitude error for categorization
                true_mag = log_magnitude([true_int])[0]
                pred_mag = mag_pred[i].item()
                mag_error = abs(pred_mag - true_mag)
                
                if recon_int == true_int:
                    if mag_error > 0.5:
                        # Rescued: magnitude was wrong, but CRT fixed it!
                        rescued_count += 1
                    else:
                        # Perfect: correct from start
                        perfect_count += 1
                else:
                    # Failed: CRT couldn't save it
                    failed_count += 1
    
    # Compute final metrics
    if total_samples == 0:
        logger.info("No samples to evaluate (empty dataloader)")
        return {
            'mag_acc': 0.0,
            'sign_acc': 0.0,
            'mod3_acc': 0.0,
            'mod5_acc': 0.0,
            'mod7_acc': 0.0,
            'mod8_acc': 0.0,
            'mod10_acc': 0.0,
            'mod11_acc': 0.0,
            'mod13_acc': 0.0,
            'mod100_acc': 0.0,
            'perfect_count': 0,
            'rescued_count': 0,
            'failed_count': 0
        }
    
    mag_acc = (total_mag_correct / total_samples) * 100
    accuracies = {f"{k}_acc": (v / total_samples) * 100 for k, v in correct_counts.items()}
    
    results = {
        'mag_acc': mag_acc,
        **accuracies,
        'perfect_count': perfect_count,
        'rescued_count': rescued_count,
        'failed_count': failed_count
    }
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"  Mag Bin Acc: {mag_acc:.1f}%")
    logger.info(f"  Sign Acc: {accuracies['sign_acc']:.1f}% | " +
                f"Mod3: {accuracies['mod3_acc']:.1f}% | " +
                f"Mod7: {accuracies['mod7_acc']:.1f}% | " +
                f"Mod10: {accuracies['mod10_acc']:.1f}% | " +
                f"Mod100: {accuracies['mod100_acc']:.1f}%")
    logger.info(f"  Reconstruction (n={total_samples}):")
    logger.info(f"    ✓ Perfect: {perfect_count} ({perfect_count/total_samples*100:.1f}%)")
    logger.info(f"    ✓ Rescued: {rescued_count} ({rescued_count/total_samples*100:.1f}%)  ← CRT Success!")
    logger.info(f"    ✗ Failed: {failed_count} ({failed_count/total_samples*100:.1f}%)")
    
    decoder.train()
    return results


def train_decoder(config: Dict[str, Any]) -> None:
    """
    Main training function for decoder.
    
    Args:
        config: Configuration dictionary
    """
    # Setup
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("=" * 50)
    logger.info("Starting Decoder Training")
    logger.info("=" * 50)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Check if bypass mode is enabled
    bypass_bert = config.get('bypass_bert', False)
    
    # Load BERT (or skip in bypass mode)
    bert_model = None
    
    if bypass_bert:
        logger.info("=" * 50)
        logger.info("BYPASS MODE ENABLED")
        logger.info("Decoder will learn from raw 35-dim features")
        logger.info("(Skipping BERT - Sanity Check Mode)")
        logger.info("=" * 50)
    else:
        logger.info(f"Loading frozen BERT from {config['bert_checkpoint']}")
        bert_model, bert_checkpoint = IntSeqBERT.load_from_checkpoint(
            config['bert_checkpoint'],
            device=device
        )
        bert_model.eval()
        bert_model.requires_grad_(False)
        logger.info("BERT frozen successfully")
    
    # Load data
    logger.info("Loading data...")
    train_ds, val_ds, test_ds = load_decoder_data(
        config['features_path'],
        config['jsonl_path'],
        val_ratio=config.get('val_ratio', 0.1),
        test_ratio=config.get('test_ratio', 0.1),
        seed=config.get('seed', 42)
    )
    logger.info(f"Loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        collate_fn=decoder_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        collate_fn=decoder_collate_fn
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        collate_fn=decoder_collate_fn
    )
    
    # Initialize decoder
    # Note: Decoder now expects 35-dim input
    decoder = NumberTheoreticDecoder(input_dim=35).to(device)
    logger.info(f"Decoder input_dim: 35 ({'raw features' if bypass_bert else 'BERT predictions'})")
    logger.info(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(
        decoder.parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Training loop
    best_val_loss = float('inf')
    epochs = config.get('epochs', 10)
    
    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'=' * 50}")
        
        # Training
        decoder.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            masked_inputs = batch['masked_inputs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mask_indices = batch['mask_indices'].to(device)
            target_integers = batch['target_integers']
            target_features = batch['target_features'].to(device)
            
            batch_size = masked_inputs.size(0)
            
            # Get feature vectors
            if bypass_bert:
                # Bypass mode: Use raw features directly
                bert_vectors = target_features
            else:
                # Standard mode: Pass through frozen BERT
                with torch.no_grad():
                    bert_output = bert_model(masked_inputs, attention_mask)
                    predictions = bert_output["prediction"]
                    # Extract masked positions (vectorized gather)
                    bert_vectors = predictions[torch.arange(batch_size), mask_indices]
            
            # Decoder forward
            decoder_output = decoder(bert_vectors)
            
            # Generate targets
            targets = get_targets(target_integers)
            
            # Multi-task loss
            # Magnitude is now classification (CrossEntropy) instead of regression (MSE)
            loss = (
                F.cross_entropy(decoder_output['sign'], targets['sign'].to(device)) +
                5.0 * F.cross_entropy(decoder_output['mag'], targets['mag'].to(device)) +  # Weighted for importance
                F.cross_entropy(decoder_output['mod3'], targets['mod3'].to(device)) +
                F.cross_entropy(decoder_output['mod5'], targets['mod5'].to(device)) +
                F.cross_entropy(decoder_output['mod7'], targets['mod7'].to(device)) +
                F.cross_entropy(decoder_output['mod8'], targets['mod8'].to(device)) +
                F.cross_entropy(decoder_output['mod10'], targets['mod10'].to(device)) +
                F.cross_entropy(decoder_output['mod11'], targets['mod11'].to(device)) +
                F.cross_entropy(decoder_output['mod13'], targets['mod13'].to(device)) +
                F.cross_entropy(decoder_output['mod100'], targets['mod100'].to(device))
            )
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / train_steps
        logger.info(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation
        logger.info("Running validation...")
        val_results = evaluate_decoder(bert_model, decoder, val_loader, device, logger)
        
        # Save best model
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_results': val_results,
                'config': config
            }, output_dir / "best_decoder.pt")
            logger.info(f"✓ Saved best decoder (loss: {best_val_loss:.4f})")
    
    # Final evaluation on test set
    logger.info(f"\n{'=' * 50}")
    logger.info("Final Test Set Evaluation")
    logger.info(f"{'=' * 50}")
    test_results = evaluate_decoder(bert_model, decoder, test_loader, device, logger)
    
    logger.info("\nTraining Complete!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train decoder for IntSeqBERT")
    
    parser.add_argument("--bert_checkpoint", type=str, default=None,
                        help="Path to trained BERT checkpoint (not needed if --bypass_bert)")
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to features.pt file")
    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="Path to original JSONL data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/decoder",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--bypass_bert", action="store_true",
                        help="Bypass BERT and use raw features (sanity check mode)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.bypass_bert and args.bert_checkpoint is None:
        parser.error("--bert_checkpoint is required unless --bypass_bert is specified")
    
    config = vars(args)
    
    train_decoder(config)


if __name__ == "__main__":
    main()
