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
from typing import Dict, List
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
    # Basic periodicity
    2: "Parity (Odd/Even)",
    3: "Mod-3 (Digit Sum Remainder)",
    4: "Last 2 Bits",
    5: "Last Digit (Base-5)",
    6: "LCM(2,3) - 2 & 3 Combined",
    7: "Prime",
    8: "Last 3 Bits",
    9: "Mod-9 (Digital Root)",
    
    # Base-10 related (representation dependency)
    10: "Base-10 (Last Digit)",
    20: "Base-10 Multiple",
    50: "Base-10 Multiple",
    100: "Base-10 (Last 2 Digits)",
    
    # Highly Composite Numbers
    12: "Highly Composite (LCM(3,4))",
    24: "Highly Composite",
    60: "Sexagesimal Base",
    
    # Large primes
    97: "Large Prime",
    101: "Large Prime (Near 100)",
}

# Base-10 related moduli (excluded from non_base10_acc calculation)
BASE10_RELATED_MODS = config.BASE10_RELATED_MODS

# Minimum number of samples per tag for stratified analysis
MIN_TAG_SAMPLES = config.MIN_TAG_SAMPLES


def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_interpretation(modulus: int) -> str:
    """Get human-readable interpretation for a modulus."""
    if modulus in INTERPRETATION_MAP:
        return INTERPRETATION_MAP[modulus]
    elif is_prime(modulus):
        return f"Prime ({modulus})"
    elif modulus % 10 == 0:
        return "Base-10 Multiple"
    elif modulus % 2 == 0:
        return f"Even ({modulus})"
    else:
        return ""


# ==========================================
# Core Functions
# ==========================================

def compute_nig(ce_loss: float, modulus: int) -> float:
    """
    Compute Normalized Information Gain.
    
    Formula: R(m) = 1.0 - (Loss / log(m))
    
    Returns:
        NIG score (1.0 = perfect, 0.0 = random, < 0 = worse than random)
    """
    if modulus <= 1:
        raise ValueError(f"modulus must be >= 2, got {modulus}")
    max_entropy = np.log(modulus)
    return 1.0 - (ce_loss / max_entropy)


# Re-export split_mod_logits for backward compatibility
_split_mod_logits = split_mod_logits


def compute_mod_metrics(
    mod_logits: torch.Tensor,
    mod_targets: torch.Tensor,
    mask_map: torch.Tensor
) -> pd.DataFrame:
    """
    Compute per-modulus metrics.
    
    Returns:
        DataFrame with columns: [modulus, accuracy, ce_loss, nig_score]
    """
    results = []
    split_logits_list = split_mod_logits(mod_logits)
    
    for i, m in enumerate(config.MOD_RANGE):
        logits_m = split_logits_list[i]  # (N, L, m)
        targets_m = mod_targets[:, :, i]  # (N, L)
        
        # Only valid (masked) positions
        valid = mask_map.bool()
        logits_flat = logits_m[valid]  # (num_valid, m)
        targets_flat = targets_m[valid].long()  # (num_valid,)
        
        if logits_flat.size(0) == 0:
            results.append({
                "modulus": m,
                "accuracy": 0.0,
                "ce_loss": float('inf'),
                "nig_score": float('-inf')
            })
            continue
        
        # Accuracy
        preds = logits_flat.argmax(dim=-1)
        accuracy = (preds == targets_flat).float().mean().item() * 100
        
        # CE Loss
        ce_loss = F.cross_entropy(logits_flat, targets_flat).item()
        
        # NIG
        nig = compute_nig(ce_loss, m)
        
        results.append({
            "modulus": m,
            "accuracy": accuracy,
            "ce_loss": ce_loss,
            "nig_score": nig
        })
    
    return pd.DataFrame(results)


def bootstrap_ci(
    mod_logits: torch.Tensor,
    mod_targets: torch.Tensor,
    mask_map: torch.Tensor,
    n_samples: int = config.BOOTSTRAP_SAMPLES_DEFAULT,
    ci_level: float = config.CI_LEVEL_DEFAULT,
    seed: int = None,
    quiet: bool = False
) -> pd.DataFrame:
    """
    Estimate NIG confidence intervals via Bootstrap.
    
    Args:
        seed: Random seed for reproducibility
        quiet: If True, disable progress bar
    
    Returns:
        DataFrame with columns: [modulus, nig_mean, nig_lower, nig_upper]
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_sequences = mod_logits.size(0)
    results = {m: [] for m in config.MOD_RANGE}
    
    for _ in tqdm(range(n_samples), desc="Bootstrap", disable=quiet):
        # Resample with replacement
        indices = np.random.choice(n_sequences, n_sequences, replace=True)
        sample_logits = mod_logits[indices]
        sample_targets = mod_targets[indices]
        sample_mask = mask_map[indices]
        
        # Per-modulus NIG
        metrics = compute_mod_metrics(sample_logits, sample_targets, sample_mask)
        for _, row in metrics.iterrows():
            results[row["modulus"]].append(row["nig_score"])
    
    # Compute CI
    alpha = (1 - ci_level) / 2
    ci_data = []
    for m in config.MOD_RANGE:
        nig_values = np.array(results[m])
        ci_data.append({
            "modulus": m,
            "nig_mean": nig_values.mean(),
            "nig_lower": np.percentile(nig_values, alpha * 100),
            "nig_upper": np.percentile(nig_values, (1 - alpha) * 100)
        })
    
    return pd.DataFrame(ci_data)


# ==========================================
# Tag-Stratified Analysis
# ==========================================

def load_oeis_tags(jsonl_path: str) -> Dict[str, List[str]]:
    """
    Load OEIS tags from JSONL file.
    
    Returns:
        {oeis_id: [tag1, tag2, ...], ...}
    """
    id_to_tags = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            id_to_tags[record["oeis_id"]] = record.get("keywords", [])
    return id_to_tags


def _compute_non_base10_acc(metrics: pd.DataFrame) -> float:
    """
    Compute accuracy excluding Base-10 related moduli.
    
    Excludes: 10, 20, 50, 100 (defined in BASE10_RELATED_MODS)
    """
    non_base10 = metrics[~metrics["modulus"].isin(BASE10_RELATED_MODS)]
    return non_base10["accuracy"].mean()


def tag_stratified_analysis(
    mod_logits: torch.Tensor,
    mod_targets: torch.Tensor,
    mask_map: torch.Tensor,
    oeis_ids: List[str],
    id_to_tags: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Perform tag-stratified analysis.
    
    Returns:
        DataFrame with columns: [tag, count, overall_acc, non_base10_acc, nig_score, top_modulus]
    """
    # Build tag -> indices mapping
    tag_to_indices = defaultdict(list)
    for i, oeis_id in enumerate(oeis_ids):
        for tag in id_to_tags.get(oeis_id, []):
            tag_to_indices[tag].append(i)
    
    results = []
    for tag, indices in tag_to_indices.items():
        if len(indices) < MIN_TAG_SAMPLES:
            continue
        
        indices_t = torch.tensor(indices, dtype=torch.long)
        tag_logits = mod_logits[indices_t]
        tag_targets = mod_targets[indices_t]
        tag_mask = mask_map[indices_t]
        
        # Per-tag metrics
        metrics = compute_mod_metrics(tag_logits, tag_targets, tag_mask)
        
        # Top modulus by NIG (with safety check)
        max_nig = metrics["nig_score"].max()
        if max_nig == float('-inf') or pd.isna(max_nig):
            top_modulus = config.MOD_RANGE[0]  # Fallback to first modulus
        else:
            top_row = metrics.loc[metrics["nig_score"].idxmax()]
            top_modulus = int(top_row["modulus"])
        
        results.append({
            "tag": tag,
            "count": len(indices),
            "overall_acc": metrics["accuracy"].mean(),
            "non_base10_acc": _compute_non_base10_acc(metrics),
            "nig_score": metrics["nig_score"].mean(),
            "top_modulus": top_modulus
        })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("nig_score", ascending=False)
    return df


# ==========================================
# Prediction Collection
# ==========================================

def collect_predictions(
    model: ModelWrapper,
    dataloader
) -> Dict[str, torch.Tensor]:
    """
    Collect predictions over entire dataset.
    
    Returns:
        {
            "mod_logits": (N, L, ~5150),
            "mod_targets": (N, L, 100),
            "mask_map": (N, L),
            "oeis_ids": List[str]
        }
    """
    all_logits, all_targets, all_masks, all_ids = [], [], [], []
    max_len = config.MAX_SEQUENCE_LENGTH
    
    for batch in tqdm(dataloader, desc="Predicting"):
        preds = model.predict(batch)
        logits = preds["mod_logits"].cpu()
        targets = batch["mod_labels"].cpu()
        masks = batch["mask_matrix"].cpu()
        
        B, L = masks.shape
        
        # Pad to max_len if needed
        if L < max_len:
            pad_len = max_len - L
            # Pad logits: (B, L, D) -> (B, max_len, D)
            logits = F.pad(logits, (0, 0, 0, pad_len), value=0)
            # Pad targets: (B, L, num_mods) -> (B, max_len, num_mods)
            targets = F.pad(targets, (0, 0, 0, pad_len), value=config.IGNORE_INDEX)
            # Pad masks: (B, L) -> (B, max_len)
            masks = F.pad(masks, (0, pad_len), value=False)
        elif L > max_len:
            # Truncate if longer than max_len
            logits = logits[:, :max_len, :]
            targets = targets[:, :max_len, :]
            masks = masks[:, :max_len]
        
        all_logits.append(logits)
        all_targets.append(targets)
        all_masks.append(masks)
        all_ids.extend(batch["oeis_ids"])
    
    return {
        "mod_logits": torch.cat(all_logits, dim=0),
        "mod_targets": torch.cat(all_targets, dim=0),
        "mask_map": torch.cat(all_masks, dim=0),
        "oeis_ids": all_ids
    }


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
    parser.add_argument("--seed", type=int, default=None, help="Random seed for bootstrap")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars")
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
    
    # Collect predictions
    logging.info("Collecting predictions...")
    preds = collect_predictions(model, dataloader)
    
    # Debug: Check tensor shapes
    print(f"DEBUG: mod_logits shape = {preds['mod_logits'].shape}", flush=True)
    print(f"DEBUG: mod_targets shape = {preds['mod_targets'].shape}", flush=True)
    print(f"DEBUG: mask_map shape = {preds['mask_map'].shape}", flush=True)
    print(f"DEBUG: oeis_ids count = {len(preds['oeis_ids'])}", flush=True)
    
    # Compute metrics
    logging.info("Computing per-modulus metrics...")
    print("DEBUG: Starting compute_mod_metrics...", flush=True)
    metrics_df = compute_mod_metrics(
        preds["mod_logits"],
        preds["mod_targets"],
        preds["mask_map"]
    )
    print("DEBUG: compute_mod_metrics completed", flush=True)
    metrics_df["interpretation"] = metrics_df["modulus"].apply(get_interpretation)
    metrics_df = metrics_df.sort_values("nig_score", ascending=False)
    metrics_df.to_csv(output_dir / "nig_ranking.csv", index=False)
    logging.info(f"Saved: {output_dir / 'nig_ranking.csv'}")
    
    # Bootstrap CI
    if args.bootstrap_samples > 0:
        logging.info("Computing Bootstrap confidence intervals...")
        ci_df = bootstrap_ci(
            preds["mod_logits"],
            preds["mod_targets"],
            preds["mask_map"],
            n_samples=args.bootstrap_samples,
            seed=args.seed,
            quiet=args.quiet
        )
        ci_df.to_csv(output_dir / "nig_ci.csv", index=False)
        logging.info(f"Saved: {output_dir / 'nig_ci.csv'}")
    
    # Tag-stratified analysis
    if Path(args.jsonl_path).exists():
        logging.info("Performing tag-stratified analysis...")
        id_to_tags = load_oeis_tags(args.jsonl_path)
        tag_df = tag_stratified_analysis(
            preds["mod_logits"],
            preds["mod_targets"],
            preds["mask_map"],
            preds["oeis_ids"],
            id_to_tags
        )
        tag_df.to_csv(output_dir / "tag_analysis.csv", index=False)
        logging.info(f"Saved: {output_dir / 'tag_analysis.csv'}")
    
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
