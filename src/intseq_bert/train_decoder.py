"""
Training script for NumberTheoreticDecoder using frozen IntSeqBERT representations.
"""

import argparse
import json
import logging
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
            - mag: (batch,) FloatTensor [log magnitude]
            - mod3/5/8/10: (batch,) LongTensor [residues]
    """
    batch_size = len(integers)
    
    # Sign: map to [0, 1, 2]
    signs = torch.tensor([
        0 if x < 0 else (1 if x == 0 else 2)
        for x in integers
    ], dtype=torch.long)
    
    # Magnitude: use same log_magnitude as encoder
    mags = torch.tensor([
        log_magnitude([x])[0] for x in integers
    ], dtype=torch.float32)
    
    # Modulo residues (Python % already handles negatives correctly)
    mod3 = torch.tensor([x % 3 for x in integers], dtype=torch.long)
    mod5 = torch.tensor([x % 5 for x in integers], dtype=torch.long)
    mod8 = torch.tensor([x % 8 for x in integers], dtype=torch.long)
    mod10 = torch.tensor([x % 10 for x in integers], dtype=torch.long)
    
    return {
        "sign": signs,
        "mag": mags,
        "mod3": mod3,
        "mod5": mod5,
        "mod8": mod8,
        "mod10": mod10
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
            - masked_inputs: (batch, max_len, 27) padded tensor
            - attention_mask: (batch, max_len) padding mask
            - mask_indices: (batch,) positions that were masked
            - target_integers: List[int] ground truth integers
    """
    batch_size = len(batch)
    
    # Get sequence lengths
    seq_lens = [item['features'].shape[0] for item in batch]
    max_len = max(seq_lens)
    
    # Initialize tensors
    masked_inputs = torch.zeros(batch_size, max_len, 27)
    attention_mask = torch.zeros(batch_size, max_len)
    mask_indices = torch.zeros(batch_size, dtype=torch.long)
    target_integers = []
    
    for i, item in enumerate(batch):
        features = item['features']  # (seq_len, 27)
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
    
    return {
        'masked_inputs': masked_inputs,
        'attention_mask': attention_mask,
        'mask_indices': mask_indices,
        'target_integers': target_integers
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
    
    total_mag_error = 0.0
    correct_counts = {
        'sign': 0, 'mod3': 0, 'mod5': 0, 'mod8': 0, 'mod10': 0
    }
    total_samples = 0
    
    # CRT reconstruction tracking
    perfect_count = 0
    rescued_count = 0
    failed_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            masked_inputs = batch['masked_inputs'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mask_indices = batch['mask_indices'].to(device)
            target_integers = batch['target_integers']
            
            batch_size = masked_inputs.size(0)
            total_samples += batch_size
            
            # Get BERT representations
            bert_output = bert_model(masked_inputs, attention_mask)
            predictions = bert_output["prediction"]
            
            # Extract at masked positions
            bert_vectors = predictions[torch.arange(batch_size), mask_indices]
            
            # Decoder forward
            decoder_output = decoder(bert_vectors)
            targets = get_targets(target_integers)
            
            # Magnitude MAE
            mag_pred = decoder_output['mag'].squeeze(-1).cpu()
            mag_target = targets['mag']
            total_mag_error += (mag_pred - mag_target).abs().sum().item()
            
            # Classification accuracies
            for head in ['sign', 'mod3', 'mod5', 'mod8', 'mod10']:
                pred_classes = decoder_output[head].argmax(dim=1).cpu()
                correct_counts[head] += (pred_classes == targets[head]).sum().item()
            
            # CRT reconstruction evaluation
            for i in range(batch_size):
                true_int = target_integers[i]
                recon_int, _ = decoder.reconstruct_value(bert_vectors[i].cpu())
                
                # Compute magnitude error
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
            'mag_mae': 0.0,
            'sign_acc': 0.0,
            'mod3_acc': 0.0,
            'mod5_acc': 0.0,
            'mod8_acc': 0.0,
            'mod10_acc': 0.0,
            'perfect_count': 0,
            'rescued_count': 0,
            'failed_count': 0
        }
    
    mag_mae = total_mag_error / total_samples
    accuracies = {f"{k}_acc": (v / total_samples) * 100 for k, v in correct_counts.items()}
    
    results = {
        'mag_mae': mag_mae,
        **accuracies,
        'perfect_count': perfect_count,
        'rescued_count': rescued_count,
        'failed_count': failed_count
    }
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"  Mag MAE: {mag_mae:.4f}")
    logger.info(f"  Sign Acc: {accuracies['sign_acc']:.1f}% | " +
                f"Mod3: {accuracies['mod3_acc']:.1f}% | " +
                f"Mod10: {accuracies['mod10_acc']:.1f}%")
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
    
    # Load frozen BERT
    logger.info(f"Loading frozen BERT from {config['bert_checkpoint']}")
    bert_model, _ = IntSeqBERT.load_from_checkpoint(
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
    decoder = NumberTheoreticDecoder().to(device)
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
            
            batch_size = masked_inputs.size(0)
            
            # Pass through frozen BERT
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
            loss = (
                F.cross_entropy(decoder_output['sign'], targets['sign'].to(device)) +
                5.0 * F.mse_loss(decoder_output['mag'], targets['mag'].unsqueeze(1).to(device)) +
                F.cross_entropy(decoder_output['mod3'], targets['mod3'].to(device)) +
                F.cross_entropy(decoder_output['mod5'], targets['mod5'].to(device)) +
                F.cross_entropy(decoder_output['mod8'], targets['mod8'].to(device)) +
                F.cross_entropy(decoder_output['mod10'], targets['mod10'].to(device))
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
    
    parser.add_argument("--bert_checkpoint", type=str, required=True,
                        help="Path to trained BERT checkpoint")
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
    
    args = parser.parse_args()
    config = vars(args)
    
    train_decoder(config)


if __name__ == "__main__":
    main()
