"""
analyze_magnitude.py:
Magnitude (numerical scale) prediction analysis for IntSeqBERT and comparison models.

Computes scale-wise metrics, calibration analysis, error distribution, and worst-K analysis.
"""

import torch
import numpy as np
import pandas as pd
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from scipy import stats

from intseq_bert import config
from intseq_bert.analysis.common import (
    ModelWrapper,
    create_model_wrapper,
)


logger = logging.getLogger(__name__)


# ==========================================
# Constants
# ==========================================

BUCKET_BOUNDS = [
    (0, 2, "Small"),
    (2, 5, "Medium"),
    (5, 20, "Large"),
    (20, 50, "Huge"),
    (50, float('inf'), "Astronomical"),
]

MIN_RELIABLE_SAMPLES = 30


# ==========================================
# Accuracy Metrics
# ==========================================

def compute_mse(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute Mean Squared Error."""
    return ((gt - pred) ** 2).mean().item()


def compute_rmse(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(compute_mse(gt, pred))


def compute_mae(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute Mean Absolute Error."""
    return (gt - pred).abs().mean().item()


def compute_medae(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute Median Absolute Error."""
    return (gt - pred).abs().median().item()


def compute_r2(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = ((gt - pred) ** 2).sum().item()
    ss_tot = ((gt - gt.mean()) ** 2).sum().item()
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def compute_tolerance_accuracy(
    gt: torch.Tensor, 
    pred: torch.Tensor, 
    tolerance: float
) -> float:
    """Compute percentage of predictions within tolerance."""
    within = (gt - pred).abs() < tolerance
    return within.float().mean().item() * 100


def compute_pearson(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient."""
    gt_np = gt.numpy() if isinstance(gt, torch.Tensor) else gt
    pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else pred
    rho, _ = stats.pearsonr(gt_np.flatten(), pred_np.flatten())
    return rho


def compute_spearman(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute Spearman rank correlation coefficient."""
    gt_np = gt.numpy() if isinstance(gt, torch.Tensor) else gt
    pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else pred
    rho, _ = stats.spearmanr(gt_np.flatten(), pred_np.flatten())
    return rho


# ==========================================
# Uncertainty Metrics
# ==========================================

def compute_nll(
    gt: torch.Tensor, 
    pred: torch.Tensor, 
    sigma: torch.Tensor
) -> float:
    """
    Compute Negative Log Likelihood assuming Gaussian distribution.
    
    NLL = 0.5 * (log(2*pi) + log(sigma^2) + (y - pred)^2 / sigma^2)
    """
    sigma = sigma.clamp(min=1e-6)  # Avoid log(0)
    nll = 0.5 * (np.log(2 * np.pi) + 2 * torch.log(sigma) + ((gt - pred) ** 2) / (sigma ** 2))
    return nll.mean().item()


def compute_expected_calibration_error(
    gt: torch.Tensor,
    pred: torch.Tensor,
    sigma: torch.Tensor,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Measures the difference between predicted uncertainty and actual errors.
    """
    errors = (gt - pred).abs()
    
    # Sort by predicted sigma
    sorted_indices = sigma.argsort()
    sorted_sigma = sigma[sorted_indices]
    sorted_errors = errors[sorted_indices]
    
    n = len(sigma)
    bin_size = n // n_bins
    
    ece = 0.0
    total_count = 0
    
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        
        bin_sigma = sorted_sigma[start:end]
        bin_errors = sorted_errors[start:end]
        
        if len(bin_sigma) == 0:
            continue
        
        mean_sigma = bin_sigma.mean().item()
        rmse = torch.sqrt((bin_errors ** 2).mean()).item()
        
        # ECE contribution: |mean_sigma - rmse| weighted by bin size
        ece += abs(mean_sigma - rmse) * len(bin_sigma)
        total_count += len(bin_sigma)
    
    return ece / total_count if total_count > 0 else 0.0


def compute_calibration_data(
    gt: torch.Tensor,
    pred: torch.Tensor,
    sigma: torch.Tensor,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Compute calibration data for plotting.
    
    Returns DataFrame with columns: [bin, mean_sigma, rmse, count]
    """
    errors = (gt - pred).abs()
    
    # Sort by predicted sigma
    sorted_indices = sigma.argsort()
    sorted_sigma = sigma[sorted_indices]
    sorted_errors = errors[sorted_indices]
    
    n = len(sigma)
    bin_size = n // n_bins
    
    results = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        
        bin_sigma = sorted_sigma[start:end]
        bin_errors = sorted_errors[start:end]
        
        if len(bin_sigma) == 0:
            continue
        
        results.append({
            "bin": i + 1,
            "mean_sigma": bin_sigma.mean().item(),
            "rmse": torch.sqrt((bin_errors ** 2).mean()).item(),
            "count": len(bin_sigma)
        })
    
    return pd.DataFrame(results)


# ==========================================
# Scale-wise Analysis
# ==========================================

def get_bucket_name(log_value: float) -> str:
    """Get bucket name for a log10 value."""
    for low, high, name in BUCKET_BOUNDS:
        if low <= log_value < high:
            return name
    return "Astronomical"


def compute_scale_wise_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    n_bootstrap: int = 100
) -> pd.DataFrame:
    """
    Compute scale-wise (bucket) metrics.
    
    Returns DataFrame with columns: [bucket, count, mse, mae, mse_ci_lower, mse_ci_upper, is_reliable]
    """
    # Flatten and apply mask
    if gt.dim() > 1:
        gt_flat = gt[mask.bool()].flatten()
        pred_flat = pred[mask.bool()].flatten()
    else:
        gt_flat = gt.flatten()
        pred_flat = pred.flatten()
    
    # Assign buckets
    bucket_data = defaultdict(lambda: {"gt": [], "pred": []})
    
    for g, p in zip(gt_flat.tolist(), pred_flat.tolist()):
        bucket = get_bucket_name(g)
        bucket_data[bucket]["gt"].append(g)
        bucket_data[bucket]["pred"].append(p)
    
    results = []
    bucket_order = ["Small", "Medium", "Large", "Huge", "Astronomical"]
    
    for bucket_name in bucket_order:
        if bucket_name not in bucket_data:
            continue
        
        data = bucket_data[bucket_name]
        gt_bucket = torch.tensor(data["gt"])
        pred_bucket = torch.tensor(data["pred"])
        count = len(gt_bucket)
        
        if count == 0:
            continue
        
        mse = compute_mse(gt_bucket, pred_bucket)
        mae = compute_mae(gt_bucket, pred_bucket)
        
        # Bootstrap CI for MSE
        ci_lower, ci_upper = bootstrap_ci(
            gt_bucket.numpy(), 
            pred_bucket.numpy(),
            lambda g, p: ((g - p) ** 2).mean(),
            n_samples=n_bootstrap
        )
        
        is_reliable = count >= MIN_RELIABLE_SAMPLES
        if not is_reliable:
            logger.warning(f"Warning: Bucket '{bucket_name}' has only N={count} samples (unreliable)")
        
        results.append({
            "bucket": bucket_name,
            "count": count,
            "mse": mse,
            "mae": mae,
            "mse_ci_lower": ci_lower,
            "mse_ci_upper": ci_upper,
            "is_reliable": is_reliable
        })
    
    return pd.DataFrame(results)


# ==========================================
# Bootstrap CI
# ==========================================

def bootstrap_ci(
    gt: np.ndarray,
    pred: np.ndarray,
    metric_fn,
    n_samples: int = 1000,
    ci: float = 0.95
) -> Tuple[float, float]:
    """
    Estimate confidence interval via Bootstrap.
    
    Args:
        gt: Ground truth values
        pred: Predicted values  
        metric_fn: Function(gt, pred) -> float
        n_samples: Number of bootstrap samples
        ci: Confidence level (default 0.95)
    
    Returns:
        (lower, upper) bounds
    """
    estimates = []
    n = len(gt)
    
    for _ in range(n_samples):
        indices = np.random.choice(n, size=n, replace=True)
        sample_gt = gt[indices]
        sample_pred = pred[indices]
        estimates.append(metric_fn(sample_gt, sample_pred))
    
    lower = np.percentile(estimates, (1 - ci) / 2 * 100)
    upper = np.percentile(estimates, (1 + ci) / 2 * 100)
    return lower, upper


# ==========================================
# Error Distribution
# ==========================================

def compute_error_distribution_stats(errors: torch.Tensor) -> Dict[str, float]:
    """
    Compute error distribution statistics.
    
    Returns dict with: mean, median, std, skewness, kurtosis
    """
    errors_np = errors.numpy() if isinstance(errors, torch.Tensor) else errors
    errors_np = errors_np.flatten()
    
    return {
        "mean": float(np.mean(errors_np)),
        "median": float(np.median(errors_np)),
        "std": float(np.std(errors_np)),
        "skewness": float(stats.skew(errors_np)),
        "kurtosis": float(stats.kurtosis(errors_np))
    }


# ==========================================
# Sign-Magnitude Consistency
# ==========================================

def compute_sign_magnitude_consistency(
    pred_mag: torch.Tensor,
    pred_sign: torch.Tensor
) -> float:
    """
    Compute consistency between magnitude and sign predictions.
    
    Args:
        pred_mag: Predicted magnitude values (log scale, can be negative)
        pred_sign: Predicted sign class (0=positive, 1=negative, 2=zero)
    
    Returns:
        Consistency rate in percentage (0-100)
    """
    # Define consistency rules:
    # - mag > 0 and sign == 0 (positive) -> consistent
    # - mag < 0 and sign == 1 (negative) -> consistent
    # - mag == 0 and sign == 2 (zero) -> consistent (approximately)
    
    SIGN_POSITIVE = 0
    SIGN_NEGATIVE = 1
    SIGN_ZERO = 2
    
    consistent = (
        ((pred_mag > 0) & (pred_sign == SIGN_POSITIVE)) |
        ((pred_mag < 0) & (pred_sign == SIGN_NEGATIVE)) |
        ((pred_mag.abs() < 1e-6) & (pred_sign == SIGN_ZERO))
    )
    
    return consistent.float().mean().item() * 100


# ==========================================
# Worst-K Analysis
# ==========================================

def format_context(sequence: List[float], position: int, window: int = 2) -> str:
    """
    Generate context string showing surrounding values.
    
    Args:
        sequence: The full sequence (log scale values)
        position: Target position (0-indexed)
        window: Number of items before/after to include
    
    Returns:
        Formatted string like "..., 1.2, 3.4, [5.6], 7.8, ..."
    """
    start_idx = max(0, position - window)
    end_idx = min(len(sequence), position + window + 1)
    
    parts = []
    
    # Leading ellipsis if truncated
    if start_idx > 0:
        parts.append("...")
    
    # Values before target
    for i in range(start_idx, position):
        parts.append(f"{sequence[i]:.2f}")
    
    # Target value (highlighted with brackets)
    parts.append(f"[{sequence[position]:.2f}]")
    
    # Values after target
    for i in range(position + 1, end_idx):
        parts.append(f"{sequence[i]:.2f}")
    
    # Trailing ellipsis if truncated
    if end_idx < len(sequence):
        parts.append("...")
    
    return ", ".join(parts)


def extract_worst_k_samples(
    gt: torch.Tensor,
    pred: torch.Tensor,
    oeis_ids: List[str],
    k: int = 100,
    id_to_tags: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Extract worst-K samples by prediction error.
    
    Returns DataFrame with columns: [rank, oeis_id, position, gt_value, pred_value, error, tag, context]
    """
    # Handle 2D (N, L) tensors
    if gt.dim() == 2:
        N, L = gt.shape
        errors = (gt - pred).abs()
        
        # Find top-K errors
        flat_errors = errors.flatten()
        topk_values, topk_indices = torch.topk(flat_errors, min(k, len(flat_errors)))
        
        results = []
        for rank, (error_val, flat_idx) in enumerate(zip(topk_values, topk_indices), 1):
            seq_idx = flat_idx.item() // L
            pos_idx = flat_idx.item() % L
            
            # Get sequence for context
            seq_gt = gt[seq_idx].tolist()
            context = format_context(seq_gt, pos_idx)
            
            # Get tag
            tag = ""
            if id_to_tags and seq_idx < len(oeis_ids):
                tags = id_to_tags.get(oeis_ids[seq_idx], [])
                tag = tags[0] if tags else ""
            
            results.append({
                "rank": rank,
                "oeis_id": oeis_ids[seq_idx] if seq_idx < len(oeis_ids) else "",
                "position": pos_idx,
                "gt_value": gt[seq_idx, pos_idx].item(),
                "pred_value": pred[seq_idx, pos_idx].item(),
                "error": error_val.item(),
                "tag": tag,
                "context": context
            })
    else:
        # 1D case
        errors = (gt - pred).abs()
        topk_values, topk_indices = torch.topk(errors, min(k, len(errors)))
        
        results = []
        for rank, (error_val, idx) in enumerate(zip(topk_values, topk_indices), 1):
            idx = idx.item()
            
            # Simple context for 1D
            seq_gt = gt.tolist()
            context = format_context(seq_gt, idx)
            
            tag = ""
            if id_to_tags and idx < len(oeis_ids):
                tags = id_to_tags.get(oeis_ids[idx], [])
                tag = tags[0] if tags else ""
            
            results.append({
                "rank": rank,
                "oeis_id": oeis_ids[idx] if idx < len(oeis_ids) else "",
                "position": idx,
                "gt_value": gt[idx].item(),
                "pred_value": pred[idx].item(),
                "error": error_val.item(),
                "tag": tag,
                "context": context
            })
    
    return pd.DataFrame(results)


# ==========================================
# Tag-Stratified Analysis
# ==========================================

def compute_tag_stratified_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    oeis_ids: List[str],
    id_to_tags: Dict[str, List[str]],
    min_samples: int = 10
) -> pd.DataFrame:
    """
    Compute metrics stratified by OEIS tags.
    
    Returns DataFrame with columns: [tag, count, mse, mae]
    """
    # Build tag -> indices mapping
    tag_to_indices = defaultdict(list)
    for i, oeis_id in enumerate(oeis_ids):
        for tag in id_to_tags.get(oeis_id, []):
            tag_to_indices[tag].append(i)
    
    results = []
    for tag, indices in tag_to_indices.items():
        if len(indices) < min_samples:
            continue
        
        indices_t = torch.tensor(indices, dtype=torch.long)
        tag_gt = gt[indices_t]
        tag_pred = pred[indices_t]
        tag_mask = mask[indices_t] if mask.dim() > 1 else mask
        
        # Flatten and apply mask
        if tag_gt.dim() > 1:
            tag_gt_flat = tag_gt[tag_mask.bool()]
            tag_pred_flat = tag_pred[tag_mask.bool()]
        else:
            tag_gt_flat = tag_gt
            tag_pred_flat = tag_pred
        
        if len(tag_gt_flat) == 0:
            continue
        
        mse = compute_mse(tag_gt_flat, tag_pred_flat)
        mae = compute_mae(tag_gt_flat, tag_pred_flat)
        
        results.append({
            "tag": tag,
            "count": len(indices),
            "mse": mse,
            "mae": mae
        })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("mse", ascending=True)
    return df


# ==========================================
# Overall Metrics
# ==========================================

def compute_overall_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute all overall metrics.
    
    Returns dict with all accuracy and uncertainty metrics.
    """
    # Apply mask if provided
    if mask is not None and gt.dim() > 1:
        gt_flat = gt[mask.bool()]
        pred_flat = pred[mask.bool()]
        if sigma is not None:
            sigma_flat = sigma[mask.bool()]
        else:
            sigma_flat = None
    else:
        gt_flat = gt.flatten()
        pred_flat = pred.flatten()
        sigma_flat = sigma.flatten() if sigma is not None else None
    
    metrics = {
        "mse": compute_mse(gt_flat, pred_flat),
        "rmse": compute_rmse(gt_flat, pred_flat),
        "mae": compute_mae(gt_flat, pred_flat),
        "medae": compute_medae(gt_flat, pred_flat),
        "r2": compute_r2(gt_flat, pred_flat),
        "acc_0.5": compute_tolerance_accuracy(gt_flat, pred_flat, 0.5),
        "acc_0.1": compute_tolerance_accuracy(gt_flat, pred_flat, 0.1),
        "acc_0.05": compute_tolerance_accuracy(gt_flat, pred_flat, 0.05),
        "pearson": compute_pearson(gt_flat, pred_flat),
        "spearman": compute_spearman(gt_flat, pred_flat),
    }
    
    # Uncertainty metrics (IntSeq only)
    if sigma_flat is not None:
        metrics["nll"] = compute_nll(gt_flat, pred_flat, sigma_flat)
        metrics["ece"] = compute_expected_calibration_error(gt_flat, pred_flat, sigma_flat)
    
    return metrics


# ==========================================
# Prediction Collection
# ==========================================

def collect_predictions(
    model: ModelWrapper,
    dataloader
) -> Dict[str, torch.Tensor]:
    """
    Collect magnitude predictions over entire dataset.
    
    Returns:
        {
            "gt_mag": (N, L),
            "pred_mag": (N, L),
            "pred_sigma": (N, L) or None,
            "mask": (N, L),
            "oeis_ids": List[str]
        }
    """
    all_gt, all_pred, all_sigma, all_masks, all_ids = [], [], [], [], []
    
    for batch in tqdm(dataloader, desc="Predicting"):
        preds = model.predict(batch)
        
        # Ground truth magnitude (log scale)
        mag_labels = batch["mag_labels"]  # (B, L, 4) -> [log, s+, s-, s0]
        gt_mag = mag_labels[:, :, 0]  # Extract log value
        
        all_gt.append(gt_mag.cpu())
        all_pred.append(preds["mag_pred"].cpu())
        
        if "mag_sigma" in preds:
            all_sigma.append(preds["mag_sigma"].cpu())
        
        all_masks.append(batch["mask_matrix"].cpu())
        all_ids.extend(batch["oeis_id"])
    
    result = {
        "gt_mag": torch.cat(all_gt, dim=0),
        "pred_mag": torch.cat(all_pred, dim=0),
        "mask": torch.cat(all_masks, dim=0),
        "oeis_ids": all_ids
    }
    
    if all_sigma:
        result["pred_sigma"] = torch.cat(all_sigma, dim=0)
    else:
        result["pred_sigma"] = None
    
    return result


# ==========================================
# CLI
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Magnitude Prediction Analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split_type", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="intseq")
    parser.add_argument("--jsonl_path", type=str, default="data/oeis/data.jsonl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--worst_k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
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
    logger.info("Collecting predictions...")
    preds = collect_predictions(model, dataloader)
    
    gt_mag = preds["gt_mag"]
    pred_mag = preds["pred_mag"]
    pred_sigma = preds["pred_sigma"]
    mask = preds["mask"]
    oeis_ids = preds["oeis_ids"]
    
    # 1. Overall metrics
    logger.info("Computing overall metrics...")
    metrics = compute_overall_metrics(gt_mag, pred_mag, pred_sigma, mask)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "overall_metrics.csv", index=False)
    logger.info(f"Saved: {output_dir / 'overall_metrics.csv'}")
    
    # 2. Scale-wise metrics
    logger.info("Computing scale-wise metrics...")
    scale_df = compute_scale_wise_metrics(gt_mag, pred_mag, mask, n_bootstrap=args.bootstrap_samples)
    scale_df.to_csv(output_dir / "scale_wise_metrics.csv", index=False)
    logger.info(f"Saved: {output_dir / 'scale_wise_metrics.csv'}")
    
    # 3. Error distribution
    logger.info("Computing error distribution...")
    if mask is not None:
        errors = (gt_mag - pred_mag)[mask.bool()]
    else:
        errors = gt_mag - pred_mag
    error_stats = compute_error_distribution_stats(errors)
    error_df = pd.DataFrame([error_stats])
    error_df.to_csv(output_dir / "error_distribution.csv", index=False)
    logger.info(f"Saved: {output_dir / 'error_distribution.csv'}")
    
    # 4. Calibration (if sigma available)
    if pred_sigma is not None:
        logger.info("Computing calibration data...")
        if mask is not None:
            gt_flat = gt_mag[mask.bool()]
            pred_flat = pred_mag[mask.bool()]
            sigma_flat = pred_sigma[mask.bool()]
        else:
            gt_flat = gt_mag.flatten()
            pred_flat = pred_mag.flatten()
            sigma_flat = pred_sigma.flatten()
        
        cal_df = compute_calibration_data(gt_flat, pred_flat, sigma_flat)
        cal_df.to_csv(output_dir / "calibration_data.csv", index=False)
        logger.info(f"Saved: {output_dir / 'calibration_data.csv'}")
    
    # 5. Worst-K samples
    logger.info(f"Extracting worst-{args.worst_k} samples...")
    id_to_tags = {}
    if Path(args.jsonl_path).exists():
        with open(args.jsonl_path, "r") as f:
            for line in f:
                record = json.loads(line)
                id_to_tags[record["oeis_id"]] = record.get("keywords", [])
    
    worst_df = extract_worst_k_samples(gt_mag, pred_mag, oeis_ids, k=args.worst_k, id_to_tags=id_to_tags)
    worst_df.to_csv(output_dir / "worst_k_samples.csv", index=False)
    logger.info(f"Saved: {output_dir / 'worst_k_samples.csv'}")
    
    # 6. Tag-stratified analysis
    if id_to_tags:
        logger.info("Computing tag-stratified metrics...")
        tag_df = compute_tag_stratified_metrics(gt_mag, pred_mag, mask, oeis_ids, id_to_tags)
        tag_df.to_csv(output_dir / "tag_wise_metrics.csv", index=False)
        logger.info(f"Saved: {output_dir / 'tag_wise_metrics.csv'}")
    
    # 7. Consistency report (placeholder)
    consistency_data = {"sign_mag_consistency": "N/A - requires sign predictions"}
    consistency_df = pd.DataFrame([consistency_data])
    consistency_df.to_csv(output_dir / "consistency_report.csv", index=False)
    
    # Save config
    config_data = {
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "split_type": args.split_type,
        "split_name": args.split_name,
        "bootstrap_samples": args.bootstrap_samples,
        "worst_k": args.worst_k,
        "seed": args.seed
    }
    with open(output_dir / "analysis_config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
