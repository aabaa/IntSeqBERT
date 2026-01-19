"""
analyze_mod_spectrum.py:
Modulo Spectrum Analysis for IntSeqBERT and comparison models.

Computes per-modulus metrics (NIG, accuracy, loss) and provides
tag-stratified analysis with Bootstrap confidence intervals.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from intseq_bert import config
from intseq_bert.analysis.common import (
    ModelWrapper,
    create_model_wrapper,
    split_mod_logits,
)


# ==========================================
# Constants
# ==========================================

INTERPRETATION_MAP = {
    2: "Parity (Odd/Even)",
    3: "Mod-3 (Ternary)",
    4: "Mod-4 (Binary Suffix)",
    5: "Mod-5 (Base-5 Last Digit)",
    10: "Base-10 (Last Digit)",
    12: "Dozenal Last Digit",
    16: "Hexadecimal Last Digit",
    20: "Base-10 Multiple",
    50: "Base-10 Multiple",
    60: "Babylonian (Base-60)",
    64: "Base-64",
    100: "Base-10 (Last 2 Digits)",
    101: "Large Prime (Near 100)",
}

# Base-10 related moduli (excluded from non_base10_acc calculation)
BASE10_RELATED_MODS = config.BASE10_RELATED_MODS

# Minimum number of samples per tag for stratified analysis
MIN_TAG_SAMPLES = config.MIN_TAG_SAMPLES


def get_interpretation(modulus: int) -> str:
    """Get human-readable interpretation for a modulus."""
    return INTERPRETATION_MAP.get(modulus, "Other")


def compute_nig(ce_loss: float, modulus: int) -> float:
    """
    Compute Normalized Information Gain.
    
    R(m) = 1.0 - (Loss(m) / log(m))
    """
    max_entropy = np.log(modulus)
    return 1.0 - (ce_loss / max_entropy)


# ==========================================
# Streaming Evaluation
# ==========================================

class StreamingEvaluator:
    """
    Evaluates model metrics batch-by-batch to avoid OOM.
    Stores per-sample statistics instead of full logits.
    """
    
    def __init__(self):
        self.results = {
            "loss_sum_per_sample": [],  # List of (B, 100) tensors
            "acc_sum_per_sample": [],   # List of (B, 100) tensors
            "counts_per_sample": [],    # List of (B,) tensors
            "oeis_ids": []
        }
    
    def process_batch(self, preds: Dict, batch: Dict):
        """
        Process a single batch and accumulate stats.
        """
        # (B, L, sum(mods))
        mod_logits = preds["mod_logits"].cpu()
        # (B, L, 100)
        mod_targets = batch["mod_labels"].cpu()
        # (B, L)
        mask_matrix = batch["mask_matrix"].cpu()
        
        # Valid mask: (B, L)
        # Note: mod_targets has IGNORE_INDEX where invalid, 
        # but we use mask_matrix explicitly to be safe and consistent.
        valid_mask = mask_matrix.bool()
        
        # Count valid tokens per sample: (B,)
        sample_counts = valid_mask.sum(dim=1).float()
        
        # Split logits: List of (B, L, m)
        split_logits = split_mod_logits(mod_logits)
        
        batch_loss_sums = []
        batch_acc_sums = []
        
        for i, m in enumerate(config.MOD_RANGE):
            # Logits for mod m: (B, L, m)
            logits_m = split_logits[i]
            # Targets for mod m: (B, L)
            targets_m = mod_targets[:, :, i]
            
            # --- Accuracy ---
            pred_classes = logits_m.argmax(dim=-1) # (B, L)
            correct = (pred_classes == targets_m) & valid_mask
            acc_sum = correct.float().sum(dim=1) # (B,)
            batch_acc_sums.append(acc_sum)
            
            # --- Loss ---
            # We compute CE loss with 'none' reduction to get (B, L)
            # and then mask it.
            # Permute logits to (B, m, L) for cross_entropy
            loss_per_token = F.cross_entropy(
                logits_m.permute(0, 2, 1), 
                targets_m, 
                reduction='none', 
                ignore_index=config.IGNORE_INDEX
            ) # (B, L)
            
            # Zero out invalid positions (just in case ignore_index didn't catch everything, though it should)
            loss_per_token = loss_per_token * valid_mask.float()
            
            loss_sum = loss_per_token.sum(dim=1) # (B,)
            batch_loss_sums.append(loss_sum)
            
        # Stack metrics for this batch: (B, 100)
        self.results["loss_sum_per_sample"].append(torch.stack(batch_loss_sums, dim=1))
        self.results["acc_sum_per_sample"].append(torch.stack(batch_acc_sums, dim=1))
        self.results["counts_per_sample"].append(sample_counts)
        self.results["oeis_ids"].extend(batch["oeis_ids"])

    def finalize(self) -> Dict[str, torch.Tensor]:
        """
        Concatenate all collected stats.
        """
        return {
            "loss_sum": torch.cat(self.results["loss_sum_per_sample"], dim=0), # (N, 100)
            "acc_sum": torch.cat(self.results["acc_sum_per_sample"], dim=0),   # (N, 100)
            "counts": torch.cat(self.results["counts_per_sample"], dim=0),     # (N,)
            "oeis_ids": self.results["oeis_ids"]
        }


# ==========================================
# Metrics Calculation
# ==========================================

def compute_mod_metrics_from_stats(
    stats: Dict[str, torch.Tensor]
) -> pd.DataFrame:
    """
    Compute global metrics from accumulated statistics.
    """
    loss_sum = stats["loss_sum"] # (N, 100)
    acc_sum = stats["acc_sum"]   # (N, 100)
    counts = stats["counts"]     # (N,)
    
    # Global aggregation
    total_valid_tokens = counts.sum().item()
    
    if total_valid_tokens == 0:
        return pd.DataFrame(columns=["modulus", "accuracy", "ce_loss", "nig_score"])

    # Sum over samples
    global_loss_sum = loss_sum.sum(dim=0) # (100,)
    global_acc_sum = acc_sum.sum(dim=0)   # (100,)
    
    # Averages
    global_loss = global_loss_sum / total_valid_tokens
    global_acc = (global_acc_sum / total_valid_tokens) * 100
    
    results = []
    for i, m in enumerate(config.MOD_RANGE):
        ce_loss = global_loss[i].item()
        acc = global_acc[i].item()
        nig = compute_nig(ce_loss, m)
        
        results.append({
            "modulus": m,
            "accuracy": acc,
            "ce_loss": ce_loss,
            "nig_score": nig
        })
        
    return pd.DataFrame(results)


def bootstrap_ci_from_stats(
    stats: Dict[str, torch.Tensor],
    n_samples: int = config.BOOTSTRAP_SAMPLES_DEFAULT,
    ci_level: float = config.CI_LEVEL_DEFAULT,
    seed: int = None,
    quiet: bool = False
) -> pd.DataFrame:
    """
    Compute Bootstrap CI for NIG using accumulated stats.
    Resamples sequences (rows of stats).
    """
    if seed is not None:
        np.random.seed(seed)
        
    loss_sum = stats["loss_sum"].numpy() # (N, 100)
    counts = stats["counts"].numpy()     # (N,)
    
    N = len(counts)
    n_mods = len(config.MOD_RANGE)
    
    # Store NIG estimates: (n_samples, 100)
    nig_estimates = np.zeros((n_samples, n_mods))
    
    iterator = range(n_samples)
    if not quiet:
        iterator = tqdm(iterator, desc="Bootstrapping")
        
    for b in iterator:
        # Resample indices with replacement
        indices = np.random.choice(N, size=N, replace=True)
        
        # Gather resampled stats
        resampled_loss_sum = loss_sum[indices].sum(axis=0) # (100,)
        resampled_counts = counts[indices].sum()           # scalar
        
        if resampled_counts == 0:
            continue
            
        resampled_loss = resampled_loss_sum / resampled_counts
        
        # Calculate NIG for all mods
        for i, m in enumerate(config.MOD_RANGE):
            nig_estimates[b, i] = compute_nig(resampled_loss[i], m)
            
    # Calculate intervals
    alpha = (1.0 - ci_level) / 2.0
    lower = np.percentile(nig_estimates, alpha * 100, axis=0)
    upper = np.percentile(nig_estimates, (1.0 - alpha) * 100, axis=0)
    
    results = []
    for i, m in enumerate(config.MOD_RANGE):
        results.append({
            "modulus": m,
            "nig_lower": lower[i],
            "nig_upper": upper[i]
        })
        
    return pd.DataFrame(results)


def tag_stratified_analysis_from_stats(
    stats: Dict[str, torch.Tensor],
    id_to_tags: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Compute metrics stratified by tags using stats.
    """
    oeis_ids = stats["oeis_ids"]
    loss_sum = stats["loss_sum"] # (N, 100)
    acc_sum = stats["acc_sum"]   # (N, 100)
    counts = stats["counts"]     # (N,)
    
    # Map tag -> list of indices
    tag_to_indices = defaultdict(list)
    for idx, oid in enumerate(oeis_ids):
        tags = id_to_tags.get(oid, [])
        for tag in tags:
            tag_to_indices[tag].append(idx)
            
    results = []
    
    for tag, indices in tag_to_indices.items():
        if len(indices) < MIN_TAG_SAMPLES:
            continue
            
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        
        # Aggregate for this tag
        tag_counts = counts[indices_tensor].sum().item()
        
        if tag_counts == 0:
            continue
            
        tag_loss_sum = loss_sum[indices_tensor].sum(dim=0)
        tag_acc_sum = acc_sum[indices_tensor].sum(dim=0)
        
        tag_loss = tag_loss_sum / tag_counts
        tag_acc = (tag_acc_sum / tag_counts) * 100
        
        # Calculate per-modulus NIG
        tag_nigs = []
        for i, m in enumerate(config.MOD_RANGE):
            tag_nigs.append(compute_nig(tag_loss[i].item(), m))
        
        # Summary metrics
        overall_acc = tag_acc.mean().item()
        mean_nig = float(np.mean(tag_nigs))
        
        # Top-1 Modulus
        best_idx = np.argmax(tag_nigs)
        top_modulus = config.MOD_RANGE[best_idx]
        
        # Non-trivial accuracy (exclude Base-10)
        non_base10_accs = [
            tag_acc[i].item() 
            for i, m in enumerate(config.MOD_RANGE) 
            if m not in BASE10_RELATED_MODS
        ]
        non_trivial_acc = np.mean(non_base10_accs) if non_base10_accs else 0.0
        
        results.append({
            "tag": tag,
            "count": len(indices),
            "overall_acc": overall_acc,
            "non_trivial_acc": non_trivial_acc,
            "nig_score": mean_nig,
            "top_modulus": top_modulus
        })
        
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("nig_score", ascending=False)
    return df


def load_oeis_tags(jsonl_path: str) -> Dict[str, List[str]]:
    """Load IDs and tags from JSONL."""
    id_to_tags = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            oeis_id = record["oeis_id"]
            tags = record.get("keywords", [])
            id_to_tags[oeis_id] = tags
    return id_to_tags


# ==========================================
# CLI
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Modulo Spectrum Analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split_type", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="intseq")
    parser.add_argument("--jsonl_path", type=str, default="data/oeis/data.jsonl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    model = create_model_wrapper(args.model_type, args.checkpoint, device)
    
    # Load dataset and dataloader
    from intseq_bert.loader import load_dataset
    from intseq_bert.collator import OEISCollator
    from torch.utils.data import DataLoader
    
    dataset = load_dataset(
        split_type=args.split_type,
        split_name=args.split_name
    )
    
    collator = OEISCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    # Run streaming evaluation
    logging.info("Starting streaming evaluation (collecting stats)...")
    evaluator = StreamingEvaluator()
    for batch in tqdm(dataloader, desc="Evaluating"):
        preds = model.predict(batch)
        evaluator.process_batch(preds, batch)
        
    stats = evaluator.finalize()
    logging.info(f"Collected stats for {len(stats['oeis_ids'])} sequences.")
    
    # Compute metrics
    logging.info("Computing per-modulus metrics...")
    metrics_df = compute_mod_metrics_from_stats(stats)
    metrics_df["interpretation"] = metrics_df["modulus"].apply(get_interpretation)
    metrics_df = metrics_df.sort_values("nig_score", ascending=False)
    metrics_df.to_csv(output_dir / "mod_spectrum_ranking.csv", index=False)
    logging.info(f"Saved: {output_dir / 'mod_spectrum_ranking.csv'}")
    
    # Bootstrap CI
    if args.bootstrap_samples > 0:
        logging.info("Computing Bootstrap confidence intervals...")
        ci_df = bootstrap_ci_from_stats(
            stats,
            n_samples=args.bootstrap_samples,
            seed=args.seed,
            quiet=args.quiet
        )
        ci_df.to_csv(output_dir / "mod_spectrum_with_ci.csv", index=False)
        logging.info(f"Saved: {output_dir / 'mod_spectrum_with_ci.csv'}")
    
    # Tag-stratified analysis
    if Path(args.jsonl_path).exists():
        logging.info("Performing tag-stratified analysis...")
        id_to_tags = load_oeis_tags(args.jsonl_path)
        tag_df = tag_stratified_analysis_from_stats(
            stats,
            id_to_tags
        )
        tag_df.to_csv(output_dir / "tag_performance.csv", index=False)
        logging.info(f"Saved: {output_dir / 'tag_performance.csv'}")
    
    # Save config
    config_data = {
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "split_type": args.split_type,
        "split_name": args.split_name,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed
    }
    with open(output_dir / "analysis_config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    
    logging.info("Done!")


if __name__ == "__main__":
    main()
