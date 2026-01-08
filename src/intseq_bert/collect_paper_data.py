"""
collect_paper_data.py:
Collects detailed analysis data for the research paper.
Focuses on the Encoder's raw performance:
1. Accuracy per Modulo head (2-101) -> For Bar Charts/Heatmaps
2. Magnitude Prediction Correlation -> For Scatter Plots
"""

import json
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional

# Project modules
from intseq_bert import bert_model, loader, collator


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Collect Analysis Data for Paper")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--features_dir", type=str, required=True, help="Path to features directory")
    parser.add_argument("--output_file", type=str, default="paper_analysis_data.json")
    
    # Data Splitting (Must match training!)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for faster inference")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for debugging")
    
    return parser.parse_args()


def create_empty_mod_stats() -> Dict[int, Dict[str, int]]:
    """Create empty mod statistics dictionary."""
    return {m: {"correct": 0, "total": 0} for m in range(2, 102)}


def calculate_mod_accuracy_for_single_mod(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    mod_size: int
) -> Tuple[int, int]:
    """
    Calculate accuracy for a single mod head.
    
    Args:
        logits: (B, L, mod_size) logits tensor
        targets: (B, L) target tensor
        mask: (B, L) mask tensor (1 = masked position to evaluate)
        mod_size: Size of the modulo
        
    Returns:
        Tuple of (correct_count, total_count)
    """
    # Flatten
    flat_mask = mask.view(-1)
    flat_logits = logits.view(-1, mod_size)
    flat_targets = targets.view(-1)
    
    if flat_mask.sum() == 0:
        return 0, 0
    
    # Select masked positions
    logits_masked = flat_logits[flat_mask == 1]
    targets_masked = flat_targets[flat_mask == 1]
    
    # Ignore -100 (padding)
    valid_indices = targets_masked != -100
    if valid_indices.sum() == 0:
        return 0, 0
    
    logits_final = logits_masked[valid_indices]
    targets_final = targets_masked[valid_indices]
    
    # Compute accuracy
    preds = torch.argmax(logits_final, dim=-1)
    correct = (preds == targets_final).sum().item()
    total = targets_final.size(0)
    
    return correct, total


def calculate_mod_accuracies(
    outputs: Dict[str, torch.Tensor], 
    targets: Dict[str, torch.Tensor],
    mask_matrix: torch.Tensor,
    mod_stats: Dict[int, Dict[str, int]]
) -> None:
    """
    Update accuracy statistics for each Mod head.
    Evaluates on ALL masked positions for robust statistics.
    """
    for m in range(2, 102):
        key = f"mod{m}"
        if key not in outputs or key not in targets:
            continue
            
        correct, total = calculate_mod_accuracy_for_single_mod(
            outputs[key], 
            targets[key], 
            mask_matrix, 
            m
        )
        
        mod_stats[m]["correct"] += correct
        mod_stats[m]["total"] += total


def extract_magnitude_pairs(
    pred_mag: torch.Tensor,
    target_mag: torch.Tensor,
    mask_matrix: torch.Tensor
) -> List[Tuple[float, float]]:
    """
    Extract (target, prediction) pairs for magnitude from masked positions.
    
    Args:
        pred_mag: (B, L, 5) predicted magnitude (dim 0 is log_magnitude)
        target_mag: (B, L, 5) target magnitude
        mask_matrix: (B, L) mask tensor
        
    Returns:
        List of (target_log, pred_log) tuples
    """
    pred_log = pred_mag[..., 0]  # (B, L)
    target_log = target_mag[..., 0]  # (B, L)
    
    flat_mask = mask_matrix.view(-1)
    pred_flat = pred_log.view(-1)[flat_mask == 1]
    target_flat = target_log.view(-1)[flat_mask == 1]
    
    pairs = []
    for t, p in zip(target_flat.cpu().numpy(), pred_flat.cpu().numpy()):
        pairs.append((float(t), float(p)))
    
    return pairs


def calculate_magnitude_statistics(scatter_data: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Calculate magnitude prediction statistics.
    
    Args:
        scatter_data: List of (target, prediction) tuples
        
    Returns:
        Dictionary with correlation_r, r_squared, mae, rmse
    """
    if len(scatter_data) == 0:
        return {
            "correlation_r": 0.0,
            "r_squared": 0.0,
            "mae": 0.0,
            "rmse": 0.0
        }
    
    all_targets = np.array([x[0] for x in scatter_data])
    all_preds = np.array([x[1] for x in scatter_data])
    
    # Pearson correlation (R)
    if np.std(all_targets) > 0 and np.std(all_preds) > 0:
        correlation = float(np.corrcoef(all_targets, all_preds)[0, 1])
    else:
        correlation = 0.0
    
    # R² (Coefficient of Determination)
    r_squared = correlation ** 2
    
    # Mean Absolute Error (MAE)
    mae = float(np.mean(np.abs(all_targets - all_preds)))
    
    # Root Mean Squared Error (RMSE)
    rmse = float(np.sqrt(np.mean((all_targets - all_preds) ** 2)))
    
    return {
        "correlation_r": correlation,
        "r_squared": r_squared,
        "mae": mae,
        "rmse": rmse
    }


def aggregate_mod_results(mod_stats: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, Any]]:
    """
    Aggregate mod statistics into final results format.
    
    Args:
        mod_stats: Dictionary of {mod: {correct, total}}
        
    Returns:
        Dictionary of {mod: {accuracy, correct, total}}
    """
    results = {}
    for m in range(2, 102):
        s = mod_stats[m]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        results[m] = {
            "accuracy": acc,
            "correct": s["correct"],
            "total": s["total"]
        }
    return results


def collect_data(args):
    """Main data collection function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model
    print(f"Loading model from {args.model_path}...")
    model, _ = bert_model.IntSeqBERT.load_from_checkpoint(args.model_path, device=device)
    model.eval()
    
    # 2. Load Data (Test Split)
    print("Loading test dataset...")
    _, _, test_ds = loader.load_and_split_data(
        features_dir=args.features_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    if args.limit:
        test_ds.feature_files = test_ds.feature_files[:args.limit]
        print(f"Limiting to {args.limit} samples.")
    
    # Use Collator (same mask_prob as training)
    data_collator = collator.DualStreamCollator(mask_prob=0.15)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    print(f"Processing {len(test_ds)} samples...")
    
    # 3. Stats Containers
    mod_stats = create_empty_mod_stats()
    scatter_data = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting Data"):
            # Move to device
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
            
            # Move targets to device
            targets = batch["targets"]
            for k, v in targets.items():
                targets[k] = v.to(device)
            
            # Analysis 1: Mod Accuracies
            calculate_mod_accuracies(outputs, targets, mask_matrix, mod_stats)
            
            # Analysis 2: Magnitude Correlation
            pairs = extract_magnitude_pairs(outputs["pred_mag"], mag_labels, mask_matrix)
            scatter_data.extend(pairs)

    # 4. Aggregation & Statistics
    mag_stats = calculate_magnitude_statistics(scatter_data)
    mod_results = aggregate_mod_results(mod_stats)
    
    results = {
        "mod_accuracy": mod_results,
        "magnitude_scatter": [[x[0], x[1]] for x in scatter_data],
        "magnitude_correlation_r": mag_stats["correlation_r"],
        "magnitude_r_squared": mag_stats["r_squared"],
        "magnitude_mae": mag_stats["mae"],
        "magnitude_rmse": mag_stats["rmse"]
    }
    
    # Print statistics
    print(f"\n--- Magnitude Prediction Statistics ---")
    print(f"Correlation (R) : {mag_stats['correlation_r']:.4f}")
    print(f"R² (R-squared)  : {mag_stats['r_squared']:.4f}")
    print(f"MAE (log scale) : {mag_stats['mae']:.4f}")
    print(f"RMSE (log scale): {mag_stats['rmse']:.4f}")
    
    print("\n--- Modulo Head Accuracy (sample) ---")
    for m in [10, 20, 50, 100]:
        s = mod_results[m]
        print(f"Mod {m:3d}: {s['accuracy']:.2%} ({s['total']} samples)")
            
    # Save
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nData saved to {args.output_file}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} min")
    
    return results


if __name__ == "__main__":
    args = setup_args()
    collect_data(args)
