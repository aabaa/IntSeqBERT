"""
analyze_magnitude.py:
Magnitude (numerical scale) prediction analysis for IntSeqBERT and comparison models.

Computes scale-wise metrics, calibration analysis, error distribution, and worst-K analysis.
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
from scipy import stats

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None

from intseq_bert import config
from intseq_bert.analysis.common import (
    ModelWrapper,
    create_model_wrapper,
)


logger = logging.getLogger(__name__)


# ==========================================
# Constants
# ==========================================

BUCKET_BOUNDS = config.MAGNITUDE_BUCKETS

MIN_RELIABLE_SAMPLES = config.MIN_RELIABLE_SAMPLES


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
    sigma = sigma.clamp(min=config.EPSILON)  # Avoid log(0)
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
    n_bins: int = config.CALIBRATION_BINS_DEFAULT
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
    n_samples: int = config.BOOTSTRAP_SAMPLES_DEFAULT,
    ci: float = config.CI_LEVEL_DEFAULT
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
        ((pred_mag.abs() < config.EPSILON) & (pred_sign == SIGN_ZERO))
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
    mask: Optional[torch.Tensor],
    oeis_ids: List[str],
    k: int = config.WORST_K_DEFAULT,
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
        
        # Mask invalid errors (set to 0 so they aren't picked as top-k)
        if mask is not None:
             errors = errors * mask.float()
        
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
        
        # Mask invalid errors
        if mask is not None:
             errors = errors * mask.float()
             
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
    min_samples: int = config.MIN_TAG_SAMPLES
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
# Growth Type Analysis
# ==========================================

def analyze_log_linearity(sequence: List[int]) -> bool:
    """
    Determine if a sequence exhibits exponential (log-linear) growth.
    
    A sequence is considered log-linear if the log10 of its absolute values
    shows a high linear correlation with the position indices.
    
    Args:
        sequence: Integer sequence values
    
    Returns:
        True if the sequence appears to grow exponentially (R² > threshold)
    """
    # Filter out zeros and take absolute values
    abs_values = []
    indices = []
    for i, val in enumerate(sequence):
        if val != 0:
            abs_values.append(abs(val))
            indices.append(i)
    
    # Need at least 3 points to determine trend
    if len(abs_values) < 3:
        return False
    
    # Take log10 of absolute values
    log_values = np.log10(abs_values)
    indices_arr = np.array(indices, dtype=np.float64)
    
    # Linear regression: log_value = a * index + b
    # Calculate R² (coefficient of determination)
    n = len(log_values)
    mean_x = np.mean(indices_arr)
    mean_y = np.mean(log_values)
    
    # Avoid division by zero for constant sequences
    ss_tot = np.sum((log_values - mean_y) ** 2)
    if ss_tot < config.EPSILON:
        return False
    
    # Calculate slope and R²
    numerator = np.sum((indices_arr - mean_x) * (log_values - mean_y))
    denominator_x = np.sum((indices_arr - mean_x) ** 2)
    
    if denominator_x < config.EPSILON:
        return False
    
    slope = numerator / denominator_x
    intercept = mean_y - slope * mean_x
    
    # Predicted values
    y_pred = slope * indices_arr + intercept
    ss_res = np.sum((log_values - y_pred) ** 2)
    
    r_squared = 1.0 - (ss_res / ss_tot)
    
    return r_squared > config.LOG_LINEARITY_R2_THRESHOLD


def compute_growth_type_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    sequences: List[List[int]],
    n_bootstrap: int = config.BOOTSTRAP_SAMPLES_DEFAULT
) -> pd.DataFrame:
    """
    Compute metrics stratified by growth type (log-linear vs non-log-linear).
    
    Args:
        gt: Ground truth magnitude values (N, L)
        pred: Predicted magnitude values (N, L)
        mask: Validity mask (N, L)
        sequences: Original integer sequences for growth type determination
        n_bootstrap: Number of bootstrap samples for CI
    
    Returns:
        DataFrame with columns: [growth_type, count, mse, mae, mse_ci_lower, mse_ci_upper, is_reliable]
    """
    # Classify each sequence by growth type
    log_linear_indices = []
    non_log_linear_indices = []
    
    for i, seq in enumerate(sequences):
        if analyze_log_linearity(seq):
            log_linear_indices.append(i)
        else:
            non_log_linear_indices.append(i)
    
    results = []
    
    for growth_type, indices in [("Log-Linear", log_linear_indices), 
                                   ("Non-Log-Linear", non_log_linear_indices)]:
        if len(indices) == 0:
            continue
        
        indices_t = torch.tensor(indices, dtype=torch.long)
        group_gt = gt[indices_t]
        group_pred = pred[indices_t]
        group_mask = mask[indices_t]
        
        # Flatten and apply mask
        gt_flat = group_gt[group_mask.bool()]
        pred_flat = group_pred[group_mask.bool()]
        
        if len(gt_flat) == 0:
            continue
        
        mse = compute_mse(gt_flat, pred_flat)
        mae = compute_mae(gt_flat, pred_flat)
        
        # Bootstrap CI for MSE
        ci_lower, ci_upper = bootstrap_ci(
            gt_flat.numpy(),
            pred_flat.numpy(),
            lambda g, p: ((g - p) ** 2).mean(),
            n_samples=n_bootstrap
        )
        
        is_reliable = len(indices) >= MIN_RELIABLE_SAMPLES
        if not is_reliable:
            logger.warning(f"Warning: Growth type '{growth_type}' has only N={len(indices)} sequences (unreliable)")
        
        results.append({
            "growth_type": growth_type,
            "count": len(indices),
            "mse": mse,
            "mae": mae,
            "mse_ci_lower": ci_lower,
            "mse_ci_upper": ci_upper,
            "is_reliable": is_reliable
        })
    
    return pd.DataFrame(results)


def plot_growth_type_comparison(
    growth_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Plot comparison of metrics between Log-Linear and Non-Log-Linear sequences.
    
    Args:
        growth_df: DataFrame from compute_growth_type_metrics
        output_path: Path to save the figure
    """
    if not HAS_PLOTTING:
        logger.warning("Plotting not available (matplotlib/seaborn not installed)")
        return
    
    if len(growth_df) == 0:
        logger.warning("No growth type data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    growth_types = growth_df["growth_type"].tolist()
    x = range(len(growth_types))
    
    # Plot MSE comparison
    ax1 = axes[0]
    mse_values = growth_df["mse"].tolist()
    ci_lower = growth_df["mse_ci_lower"].tolist()
    ci_upper = growth_df["mse_ci_upper"].tolist()
    is_reliable = growth_df["is_reliable"].tolist()
    
    colors = ['#2ecc71' if r else '#e74c3c' for r in is_reliable]
    bars = ax1.bar(x, mse_values, color=colors, edgecolor='black', linewidth=1)
    
    # Error bars for CI
    for i, (xi, yi, lower, upper) in enumerate(zip(x, mse_values, ci_lower, ci_upper)):
        ax1.errorbar(xi, yi, yerr=[[yi - lower], [upper - yi]], fmt='none', color='black', capsize=5)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(growth_types, rotation=0)
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE by Growth Type')
    
    # Add count annotations
    for i, (xi, yi, count) in enumerate(zip(x, mse_values, growth_df["count"].tolist())):
        ax1.annotate(f'N={count}', (xi, yi), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=10)
    
    # Plot MAE comparison
    ax2 = axes[1]
    mae_values = growth_df["mae"].tolist()
    bars = ax2.bar(x, mae_values, color=colors, edgecolor='black', linewidth=1)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(growth_types, rotation=0)
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE by Growth Type')
    
    # Add count annotations
    for i, (xi, yi, count) in enumerate(zip(x, mae_values, growth_df["count"].tolist())):
        ax2.annotate(f'N={count}', (xi, yi), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=10)
    
    # Add legend for reliability
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Reliable (N≥30)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Unreliable (N<30)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {output_path}")


# ==========================================
# Plotting Functions
# ==========================================

def plot_error_vs_scale(
    scale_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Plot error metrics vs scale (bucket).
    
    Args:
        scale_df: DataFrame from compute_scale_wise_metrics
        output_path: Path to save the figure
    """
    if not HAS_PLOTTING:
        logger.warning("Plotting not available (matplotlib/seaborn not installed)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(scale_df))
    buckets = scale_df["bucket"].tolist()
    mse = scale_df["mse"].tolist()
    ci_lower = scale_df["mse_ci_lower"].tolist()
    ci_upper = scale_df["mse_ci_upper"].tolist()
    is_reliable = scale_df["is_reliable"].tolist()
    
    # Plot with different styles for reliable/unreliable buckets
    colors = ['#2ecc71' if r else '#e74c3c' for r in is_reliable]
    alphas = [1.0 if r else 0.3 for r in is_reliable]
    
    for i, (xi, yi, lower, upper, color, alpha) in enumerate(zip(x, mse, ci_lower, ci_upper, colors, alphas)):
        ax.bar(xi, yi, color=color, alpha=alpha, edgecolor='black', linewidth=1)
        ax.errorbar(xi, yi, yerr=[[yi - lower], [upper - yi]], fmt='none', color='black', capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=45, ha='right')
    ax.set_xlabel('Scale (Log10)')
    ax.set_ylabel('MSE')
    ax.set_title('Error vs Scale')
    
    # Add count annotations
    for i, (xi, yi, count) in enumerate(zip(x, mse, scale_df["count"].tolist())):
        ax.annotate(f'N={count}', (xi, yi), textcoords="offset points", 
                   xytext=(0, 5), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_prediction_scatter(
    gt: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    output_path: Path,
    sample_size: int = config.SCATTER_SAMPLE_SIZE
) -> None:
    """
    Plot GT vs Prediction scatter plot with diagonal line.
    
    Args:
        gt: Ground truth values
        pred: Predicted values
        mask: Mask tensor
        output_path: Path to save the figure
        sample_size: Max number of points to plot (for performance)
    """
    if not HAS_PLOTTING:
        logger.warning("Plotting not available (matplotlib/seaborn not installed)")
        return
    
    # Flatten and apply mask
    if gt.dim() > 1:
        gt_flat = gt[mask.bool()].numpy()
        pred_flat = pred[mask.bool()].numpy()
    else:
        gt_flat = gt.numpy()
        pred_flat = pred.numpy()
    
    # Sample if too many points
    if len(gt_flat) > sample_size:
        indices = np.random.choice(len(gt_flat), sample_size, replace=False)
        gt_flat = gt_flat[indices]
        pred_flat = pred_flat[indices]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(gt_flat, pred_flat, alpha=0.3, s=5, c='#3498db')
    
    # Diagonal line
    min_val = min(gt_flat.min(), pred_flat.min())
    max_val = max(gt_flat.max(), pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    
    ax.set_xlabel('Ground Truth (Log Scale)')
    ax.set_ylabel('Prediction (Log Scale)')
    ax.set_title('Prediction vs Ground Truth')
    ax.legend()
    
    # Add R² annotation
    r2 = 1 - np.sum((gt_flat - pred_flat)**2) / np.sum((gt_flat - gt_flat.mean())**2)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_calibration(
    calibration_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Plot calibration curve (predicted sigma vs actual RMSE).
    
    Args:
        calibration_df: DataFrame from compute_calibration_data
        output_path: Path to save the figure
    """
    if not HAS_PLOTTING:
        logger.warning("Plotting not available (matplotlib/seaborn not installed)")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    mean_sigma = calibration_df["mean_sigma"].values
    rmse = calibration_df["rmse"].values
    
    ax.scatter(mean_sigma, rmse, c='#3498db', s=100, zorder=3)
    ax.plot(mean_sigma, rmse, 'b-', alpha=0.5, zorder=2)
    
    # y=x line (perfect calibration)
    min_val = min(mean_sigma.min(), rmse.min())
    max_val = max(mean_sigma.max(), rmse.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Calibration')
    
    ax.set_xlabel('Predicted σ (Mean)')
    ax.set_ylabel('Actual RMSE')
    ax.set_title('Uncertainty Calibration')
    ax.legend()
    
    # Add regions annotation
    ax.fill_between([min_val, max_val], [min_val, max_val], max_val,
                   alpha=0.1, color='red', label='Overconfident')
    ax.fill_between([min_val, max_val], min_val, [min_val, max_val],
                   alpha=0.1, color='blue', label='Underconfident')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_error_histogram(
    errors: torch.Tensor,
    output_path: Path
) -> None:
    """
    Plot error distribution histogram.
    
    Args:
        errors: Error values (gt - pred)
        output_path: Path to save the figure
    """
    if not HAS_PLOTTING:
        logger.warning("Plotting not available (matplotlib/seaborn not installed)")
        return
    
    errors_np = errors.numpy() if isinstance(errors, torch.Tensor) else errors
    errors_np = errors_np.flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram with KDE
    sns.histplot(errors_np, kde=True, ax=ax, color='#3498db', bins=config.HISTOGRAM_BINS)
    
    # Statistics
    mean = np.mean(errors_np)
    median = np.median(errors_np)
    std = np.std(errors_np)
    
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
    
    ax.set_xlabel('Error (GT - Pred)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution (σ={std:.4f})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_error_qq(
    errors: torch.Tensor,
    output_path: Path
) -> None:
    """
    Plot Q-Q plot to assess normality of errors.
    
    Args:
        errors: Error values (gt - pred)
        output_path: Path to save the figure
    """
    if not HAS_PLOTTING:
        logger.warning("Plotting not available (matplotlib/seaborn not installed)")
        return
    
    errors_np = errors.numpy() if isinstance(errors, torch.Tensor) else errors
    errors_np = errors_np.flatten()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    stats.probplot(errors_np, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normal Distribution)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {output_path}")


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
        
        # Pad to MAX_SEQUENCE_LENGTH
        B, L = gt_mag.shape
        max_len = config.MAX_SEQUENCE_LENGTH
        
        if L < max_len:
            pad_len = max_len - L
            # Pad dims: (left, right, top, bottom) -> (0, pad_len, 0, 0)
            gt_mag = F.pad(gt_mag, (0, pad_len), value=config.PAD_VALUE_FEATURE)
            preds_mu = F.pad(preds["mag_mu"].cpu(), (0, pad_len), value=config.PAD_VALUE_FEATURE)
            mask_matrix = F.pad(batch["mask_matrix"].cpu(), (0, pad_len), value=False)
        else:
            gt_mag = gt_mag[:, :max_len]
            preds_mu = preds["mag_mu"].cpu()[:, :max_len]
            mask_matrix = batch["mask_matrix"].cpu()[:, :max_len]

        all_gt.append(gt_mag)
        all_pred.append(preds_mu)
        
        if "mag_log_var" in preds:
            log_var = preds["mag_log_var"].cpu()
            if L < max_len:
                 log_var = F.pad(log_var, (0, max_len - L), value=config.PAD_VALUE_FEATURE)
            else:
                 log_var = log_var[:, :max_len]
                 
            log_var_clipped = torch.clamp(
                log_var, 
                min=config.LOG_VAR_CLIP_MIN, 
                max=config.LOG_VAR_CLIP_MAX
            )
            sigma = torch.sqrt(torch.exp(log_var_clipped))
            all_sigma.append(sigma)
        
        all_masks.append(mask_matrix)
        all_ids.extend(batch["oeis_ids"])
    
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
    
    worst_df = extract_worst_k_samples(gt_mag, pred_mag, mask, oeis_ids, k=args.worst_k, id_to_tags=id_to_tags)
    worst_df.to_csv(output_dir / "worst_k_samples.csv", index=False)
    logger.info(f"Saved: {output_dir / 'worst_k_samples.csv'}")
    
    # 6. Tag-stratified analysis
    if id_to_tags:
        logger.info("Computing tag-stratified metrics...")
        tag_df = compute_tag_stratified_metrics(gt_mag, pred_mag, mask, oeis_ids, id_to_tags)
        tag_df.to_csv(output_dir / "tag_wise_metrics.csv", index=False)
        logger.info(f"Saved: {output_dir / 'tag_wise_metrics.csv'}")
    
    # 7. Growth Type Analysis
    logger.info("Computing growth type metrics...")
    # Load original sequences from JSONL
    id_to_sequence = {}
    if Path(args.jsonl_path).exists():
        with open(args.jsonl_path, "r") as f:
            for line in f:
                record = json.loads(line)
                id_to_sequence[record["oeis_id"]] = record.get("terms", [])
    
    # Build sequences list aligned with oeis_ids
    sequences = [id_to_sequence.get(oid, []) for oid in oeis_ids]
    
    # Filter to only include sequences with sufficient data
    valid_seq_mask = [len(seq) >= 3 for seq in sequences]
    if sum(valid_seq_mask) > 0:
        growth_df = compute_growth_type_metrics(
            gt_mag, pred_mag, mask, sequences, n_bootstrap=args.bootstrap_samples
        )
        growth_df.to_csv(output_dir / "growth_type_metrics.csv", index=False)
        logger.info(f"Saved: {output_dir / 'growth_type_metrics.csv'}")
    else:
        logger.warning("No valid sequences found for growth type analysis")
        growth_df = pd.DataFrame()
    
    # 8. Consistency report (placeholder)
    consistency_data = {"sign_mag_consistency": "N/A - requires sign predictions"}
    consistency_df = pd.DataFrame([consistency_data])
    consistency_df.to_csv(output_dir / "consistency_report.csv", index=False)
    
    # 8. Generate figures
    logger.info("Generating figures...")
    
    # 8.1 Error vs Scale plot
    if len(scale_df) > 0:
        plot_error_vs_scale(scale_df, figures_dir / "error_vs_scale.png")
    
    # 8.2 Prediction scatter plot
    plot_prediction_scatter(gt_mag, pred_mag, mask, figures_dir / "prediction_scatter.png")
    
    # 8.3 Calibration plot (if sigma available)
    if pred_sigma is not None and 'cal_df' in locals():
        plot_calibration(cal_df, figures_dir / "calibration_plot.png")
    
    # 9.4 Error histogram
    plot_error_histogram(errors, figures_dir / "error_histogram.png")
    
    # 9.5 Error QQ plot
    plot_error_qq(errors, figures_dir / "error_qq_plot.png")
    
    # 9.6 Growth type comparison plot
    if len(growth_df) > 0:
        plot_growth_type_comparison(growth_df, figures_dir / "growth_type_comparison.png")
    
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
