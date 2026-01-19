"""
analyze_solver.py:
Evaluate IntSeqBERT + Solver for exact match accuracy on integer reconstruction.

Measures:
- Top-1 / Top-K Exact Match Accuracy
- Per-magnitude bucket performance
- Per-solver-mode performance
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from intseq_bert import config
from intseq_bert.features import process_sequence
from intseq_bert.solver import IntegerSolver
from intseq_bert.collator import OEISCollator
from intseq_bert.analysis.common import IntSeqWrapper

logger = logging.getLogger(__name__)


# ============================================================
# Data Loading Functions
# ============================================================


def load_split_ids(split_path: Path) -> Set[str]:
    """
    Load OEIS IDs from a split file.
    
    Args:
        split_path: Path to split file (one ID per line)
    
    Returns:
        Set of OEIS IDs
    """
    with open(split_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_test_samples(
    jsonl_path: Path,
    split_ids: Set[str],
    max_samples: int,
    filter_magnitude: Optional[str] = None
) -> List[Dict]:
    """
    Load test samples from JSONL, filtering by split IDs.
    
    Each sample contains the input sequence (without target) and the target integer.
    The target is preserved as a Python int to handle arbitrarily large numbers.
    
    Args:
        jsonl_path: Path to OEIS JSONL file
        split_ids: Set of OEIS IDs to include
        max_samples: Maximum number of samples to load
        filter_magnitude: Optional magnitude filter ('small', 'medium', 'large', 'huge', 'astronomical')
    
    Returns:
        List of dicts with keys: oeis_id, input_seq, target, target_str
    """
    samples = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            
            record = json.loads(line)
            oeis_id = record.get("oeis_id", record.get("id", ""))
            
            if oeis_id not in split_ids:
                continue
            
            seq = record.get("sequence", [])
            if len(seq) < 2:
                continue
            
            target = seq[-1]
            input_seq = seq[:-1]
            
            # Filter by magnitude if specified
            if filter_magnitude:
                bucket = get_magnitude_bucket(target)
                if bucket.lower() != filter_magnitude.lower():
                    continue
            
            samples.append({
                "oeis_id": oeis_id,
                "input_seq": input_seq,
                "target": target,
                "target_str": str(target)
            })
    
    return samples


# ============================================================
# Model Loading
# ============================================================


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str
) -> Tuple[Any, IntSeqWrapper]:
    """
    Load model from checkpoint using IntSeqWrapper.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (raw_model, wrapper)
    """
    wrapper = IntSeqWrapper(str(checkpoint_path), device)
    return wrapper.model, wrapper


# ============================================================
# Magnitude Bucket Functions
# ============================================================


def get_log10_magnitude(value: int) -> float:
    """
    Get log10 of absolute value, handling edge cases.
    
    Args:
        value: Integer value
    
    Returns:
        log10(|value|), or 0 for value=0
    """
    if value == 0:
        return 0.0
    
    abs_val = abs(value)
    
    try:
        return math.log10(abs_val)
    except (ValueError, OverflowError):
        # For very large integers, use string length approximation
        return float(len(str(abs_val)))


def get_magnitude_bucket(value: int) -> str:
    """
    Classify integer into magnitude bucket.
    
    Uses config.MAGNITUDE_BUCKETS:
    - Small: 0 ~ 2 (1 ~ 100)
    - Medium: 2 ~ 5 (100 ~ 100K)
    - Large: 5 ~ 20 (100K ~ 10^20)
    - Huge: 20 ~ 50 (10^20 ~ 10^50)
    - Astronomical: 50+ (10^50+)
    
    Args:
        value: Integer value
    
    Returns:
        Bucket name
    """
    log_mag = get_log10_magnitude(value)
    
    for low, high, name in config.MAGNITUDE_BUCKETS:
        if low <= log_mag < high:
            return name
    
    return "Unknown"


# ============================================================
# Inference Functions
# ============================================================


def prepare_single_batch(
    input_seq: List[int],
    collator: OEISCollator,
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Prepare a single sequence as a batch for model input.
    
    Appends a dummy token (0) to the end to serve as the prediction target position,
    then explicitly masks that position to prevent data leakage.
    
    Masking Strategy (matches collator.py):
    1. Magnitude Stream: Set is_masked flag (channel -1) to 1.0, content channels to 0.0
    2. Modulo Stream: Zero out Sin/Cos values (origin shift)
    
    Args:
        input_seq: Input sequence (list of integers)
        collator: OEISCollator instance
        device: Device string
    
    Returns:
        Batch dict ready for model
    """
    # Append dummy (0) to end for next-token prediction
    # This makes sequence length N -> N+1, where position N is the prediction target
    input_seq_for_pred = input_seq + [0]
    
    # Process sequence to features
    features_dict = process_sequence(input_seq_for_pred)
    
    # Create batch using collator (expects list of dicts)
    # We need to add oeis_id for collator
    features_dict["oeis_id"] = "temp"
    batch = collator([features_dict])
    
    # =================================================================
    # CRITICAL: Explicitly mask the last position to prevent data leakage
    # Without this, the model sees the dummy "0" and can copy it.
    # =================================================================
    
    # Magnitude Stream: (B, L, MAG_EXTENDED_DIM)
    # The last channel is the is_masked flag.
    # Set content channels to 0 and is_masked flag to 1.0
    batch["mag_inputs"][:, -1, :config.MAG_RAW_DIM] = 0.0  # Zero content
    batch["mag_inputs"][:, -1, -1] = 1.0  # Set is_masked flag
    
    # Modulo Stream: (B, L, MOD_FEATURE_DIM)
    # Zero out Sin/Cos values at masked position (origin = masked)
    batch["mod_inputs"][:, -1, :] = 0.0
    
    return batch


def run_inference_single(
    model_wrapper: IntSeqWrapper,
    batch: Dict[str, torch.Tensor],
    solver: IntegerSolver,
    top_k: int
) -> Tuple[List[Dict], int]:
    """
    Run inference for a single sample and solve.
    
    Args:
        model_wrapper: Model wrapper
        batch: Prepared batch dict
        solver: IntegerSolver instance
        top_k: Number of candidates to return
    
    Returns:
        Tuple of (candidates_list, last_position_index)
    """
    # Get predictions
    predictions = model_wrapper.predict(batch)
    
    # Get last valid position (next item prediction)
    seq_len = batch["attention_mask"].sum(dim=1).item()
    last_pos = int(seq_len) - 1
    
    # Convert to solver format
    args = IntegerSolver.from_model_output(
        predictions, position=last_pos, model=model_wrapper.model
    )
    
    # Solve
    candidates = solver.solve(*args, top_k=top_k)
    
    return candidates, last_pos


def compute_match_rank(candidates: List[Dict], target: int) -> int:
    """
    Compute the rank at which target appears in candidates.
    
    Args:
        candidates: List of candidate dicts with 'value' key
        target: Target integer
    
    Returns:
        Rank (1-based) if found, -1 if not found
    """
    for rank, cand in enumerate(candidates, 1):
        if cand["value"] == target:
            return rank
    return -1


def get_sign_idx(value: int) -> int:
    """
    Get sign index from integer value.
    
    Args:
        value: Integer
    
    Returns:
        0 for positive, 1 for negative, 2 for zero
    """
    if value == 0:
        return config.SIGN_ZERO
    elif value > 0:
        return config.SIGN_POSITIVE
    else:
        return config.SIGN_NEGATIVE


# ============================================================
# Evaluation Loop
# ============================================================


def evaluate_samples(
    samples: List[Dict],
    model_wrapper: IntSeqWrapper,
    solver: IntegerSolver,
    collator: OEISCollator,
    device: str,
    top_k: int,
    show_progress: bool = True
) -> List[Dict]:
    """
    Evaluate all samples and collect results.
    
    Args:
        samples: List of sample dicts
        model_wrapper: Model wrapper
        solver: IntegerSolver instance
        collator: Collator for batch preparation
        device: Device string
        top_k: Number of candidates
        show_progress: Whether to show progress bar
    
    Returns:
        List of result dicts
    """
    results = []
    iterator = tqdm(samples, desc="Evaluating") if show_progress else samples
    
    for sample in iterator:
        try:
            # Prepare batch
            batch = prepare_single_batch(sample["input_seq"], collator, device)
            
            # Run inference
            candidates, last_pos = run_inference_single(
                model_wrapper, batch, solver, top_k
            )
            
            # Compute match rank
            match_rank = compute_match_rank(candidates, sample["target"])
            
            # Determine solver mode and score
            if candidates:
                solver_mode = candidates[0]["method"]
                score_top1 = candidates[0]["score"]
                pred_top1 = candidates[0]["value"]
                
                # Get sign prediction from first candidate
                sign_pred = get_sign_idx(pred_top1)
            else:
                solver_mode = "none"
                score_top1 = None
                pred_top1 = None
                sign_pred = -1
            
            # True sign
            sign_true = get_sign_idx(sample["target"])
            
            results.append({
                "oeis_id": sample["oeis_id"],
                "target": sample["target"],
                "target_str": sample["target_str"],
                "pred_top1": pred_top1,
                "match_rank": match_rank,
                "solver_mode": solver_mode,
                "mag_log10": get_log10_magnitude(sample["target"]),
                "score_top1": score_top1,
                "sign_pred": sign_pred,
                "sign_true": sign_true,
                "magnitude_bucket": get_magnitude_bucket(sample["target"])
            })
            
        except Exception as e:
            logger.warning(f"Error processing {sample['oeis_id']}: {e}")
            results.append({
                "oeis_id": sample["oeis_id"],
                "target": sample["target"],
                "target_str": sample["target_str"],
                "pred_top1": None,
                "match_rank": -1,
                "solver_mode": "error",
                "mag_log10": get_log10_magnitude(sample["target"]),
                "score_top1": None,
                "sign_pred": -1,
                "sign_true": get_sign_idx(sample["target"]),
                "magnitude_bucket": get_magnitude_bucket(sample["target"])
            })
    
    return results


# ============================================================
# Metrics Computation
# ============================================================


def compute_overall_metrics(results: List[Dict], top_k: int) -> Dict[str, Any]:
    """
    Compute overall accuracy metrics.
    
    Args:
        results: List of result dicts
        top_k: Top-K threshold
    
    Returns:
        Dict with overall metrics
    """
    total = len(results)
    if total == 0:
        return {
            "total_samples": 0,
            "top1_acc": 0.0,
            "top5_acc": 0.0,
            "sign_acc": 0.0,
            "valid_rate": 0.0
        }
    
    # Top-1 accuracy
    top1_correct = sum(1 for r in results if r["match_rank"] == 1)
    top1_acc = (top1_correct / total) * 100
    
    # Top-K accuracy
    topk_correct = sum(1 for r in results if 1 <= r["match_rank"] <= top_k)
    topk_acc = (topk_correct / total) * 100
    
    # Sign accuracy
    sign_correct = sum(
        1 for r in results 
        if r["sign_pred"] >= 0 and r["sign_pred"] == r["sign_true"]
    )
    valid_for_sign = sum(1 for r in results if r["sign_pred"] >= 0)
    sign_acc = (sign_correct / valid_for_sign * 100) if valid_for_sign > 0 else 0.0
    
    # Valid rate (solver returned candidates)
    valid_count = sum(1 for r in results if r["solver_mode"] != "none" and r["solver_mode"] != "error")
    valid_rate = (valid_count / total) * 100
    
    return {
        "total_samples": total,
        "top1_acc": round(top1_acc, 2),
        f"top{top_k}_acc": round(topk_acc, 2),
        "sign_acc": round(sign_acc, 2),
        "valid_rate": round(valid_rate, 2)
    }


def compute_magnitude_breakdown(results: List[Dict], top_k: int) -> pd.DataFrame:
    """
    Compute metrics broken down by magnitude bucket.
    
    Args:
        results: List of result dicts
        top_k: Top-K threshold
    
    Returns:
        DataFrame with per-bucket metrics
    """
    buckets = {}
    
    for r in results:
        bucket = r["magnitude_bucket"]
        if bucket not in buckets:
            buckets[bucket] = {"total": 0, "top1": 0, "topk": 0}
        
        buckets[bucket]["total"] += 1
        if r["match_rank"] == 1:
            buckets[bucket]["top1"] += 1
        if 1 <= r["match_rank"] <= top_k:
            buckets[bucket]["topk"] += 1
    
    rows = []
    for bucket_name in ["Small", "Medium", "Large", "Huge", "Astronomical"]:
        if bucket_name in buckets:
            b = buckets[bucket_name]
            rows.append({
                "bucket": bucket_name,
                "count": b["total"],
                "top1_acc": round((b["top1"] / b["total"]) * 100, 2) if b["total"] > 0 else 0.0,
                f"top{top_k}_acc": round((b["topk"] / b["total"]) * 100, 2) if b["total"] > 0 else 0.0,
                "top1_count": b["top1"],
                f"top{top_k}_count": b["topk"]
            })
    
    return pd.DataFrame(rows)


def compute_mode_breakdown(results: List[Dict], top_k: int) -> pd.DataFrame:
    """
    Compute metrics broken down by solver mode.
    
    Args:
        results: List of result dicts
        top_k: Top-K threshold
    
    Returns:
        DataFrame with per-mode metrics
    """
    modes = {}
    total_all = len(results)
    
    for r in results:
        mode = r["solver_mode"]
        if mode not in modes:
            modes[mode] = {"total": 0, "top1": 0, "topk": 0}
        
        modes[mode]["total"] += 1
        if r["match_rank"] == 1:
            modes[mode]["top1"] += 1
        if 1 <= r["match_rank"] <= top_k:
            modes[mode]["topk"] += 1
    
    rows = []
    mode_order = ["dense", "sieve", "crt", "zero", "none", "error"]
    
    for mode in mode_order:
        if mode in modes:
            m = modes[mode]
            rows.append({
                "mode": mode,
                "count": m["total"],
                "usage_rate": round((m["total"] / total_all) * 100, 2) if total_all > 0 else 0.0,
                "top1_acc": round((m["top1"] / m["total"]) * 100, 2) if m["total"] > 0 else 0.0,
                f"top{top_k}_acc": round((m["topk"] / m["total"]) * 100, 2) if m["total"] > 0 else 0.0
            })
    
    return pd.DataFrame(rows)


# ============================================================
# Output Functions
# ============================================================


def save_results_csv(results: List[Dict], output_path: Path) -> None:
    """
    Save detailed results to CSV.
    
    Args:
        results: List of result dicts
        output_path: Output file path
    """
    df = pd.DataFrame(results)
    
    # Select and order columns
    columns = [
        "oeis_id", "target", "target_str", "pred_top1", "match_rank",
        "solver_mode", "mag_log10", "score_top1", "sign_pred", "sign_true"
    ]
    
    df = df[[c for c in columns if c in df.columns]]
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")


def save_summary_json(
    overall: Dict,
    magnitude_df: pd.DataFrame,
    mode_df: pd.DataFrame,
    execution_time: float,
    output_path: Path
) -> None:
    """
    Save summary statistics to JSON.
    
    Args:
        overall: Overall metrics dict
        magnitude_df: Magnitude breakdown DataFrame
        mode_df: Mode breakdown DataFrame
        execution_time: Total execution time in seconds
        output_path: Output file path
    """
    summary = {
        "overall": overall,
        "by_magnitude": magnitude_df.set_index("bucket").to_dict(orient="index"),
        "by_mode": mode_df.set_index("mode").to_dict(orient="index"),
        "execution": {
            "total_time_sec": round(execution_time, 2),
            "avg_time_per_sample_sec": round(
                execution_time / overall["total_samples"], 3
            ) if overall["total_samples"] > 0 else 0.0
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved summary to {output_path}")


def save_config_json(args: argparse.Namespace, output_path: Path) -> None:
    """
    Save analysis configuration to JSON.
    
    Args:
        args: Parsed command line arguments
        output_path: Output file path
    """
    config_dict = {
        "checkpoint": str(args.checkpoint),
        "split_type": args.split_type,
        "split_name": args.split_name,
        "max_samples": args.max_samples,
        "top_k": args.top_k,
        "filter_magnitude": args.filter_magnitude,
        "device": args.device,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Saved config to {output_path}")


# ============================================================
# CLI and Main
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate IntSeqBERT + Solver for exact match accuracy"
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split_type", type=str, required=True,
        help="Split type (e.g., std, easy)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--split_name", type=str, default="test",
        help="Split name (train, val, test)"
    )
    parser.add_argument(
        "--data_root", type=str, default=config.DATA_ROOT,
        help="Data root directory"
    )
    parser.add_argument(
        "--max_samples", type=int, default=1000,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--top_k", type=int, default=config.SOLVER_TOP_K_DEFAULT,
        help="Number of candidates from solver"
    )
    parser.add_argument(
        "--filter_magnitude", type=str, default=None,
        choices=["small", "medium", "large", "huge", "astronomical"],
        help="Filter by magnitude bucket"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (cuda, cpu, auto)"
    )
    
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    """Setup logging configuration."""
    log_path = output_dir / "analyze_solver.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    logger.info("=" * 50)
    logger.info("Solver Evaluation")
    logger.info("=" * 50)
    
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model, model_wrapper = load_model_from_checkpoint(checkpoint_path, device)
    
    # Create solver
    solver = IntegerSolver()
    
    # Create collator (no masking needed for inference)
    collator = OEISCollator(mask_prob=0.0)
    
    # Load split IDs
    data_root = Path(args.data_root)
    split_path = data_root / config.SPLIT_DIR_NAME / args.split_type / f"{args.split_name}.txt"
    
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    
    split_ids = load_split_ids(split_path)
    logger.info(f"Loaded {len(split_ids)} IDs from {split_path}")
    
    # Load test samples
    jsonl_path = data_root / config.JSONL_FILENAME
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    
    samples = load_test_samples(
        jsonl_path, split_ids, args.max_samples, args.filter_magnitude
    )
    logger.info(f"Loaded {len(samples)} samples")
    
    if not samples:
        logger.error("No samples to evaluate!")
        return
    
    # Evaluate
    logger.info("-" * 50)
    start_time = time.time()
    
    results = evaluate_samples(
        samples, model_wrapper, solver, collator, device, args.top_k
    )
    
    execution_time = time.time() - start_time
    
    # Compute metrics
    overall = compute_overall_metrics(results, args.top_k)
    magnitude_df = compute_magnitude_breakdown(results, args.top_k)
    mode_df = compute_mode_breakdown(results, args.top_k)
    
    # Log summary
    logger.info("-" * 50)
    logger.info(f"Total samples: {overall['total_samples']}")
    logger.info(f"Top-1 Accuracy: {overall['top1_acc']:.2f}%")
    logger.info(f"Top-{args.top_k} Accuracy: {overall.get(f'top{args.top_k}_acc', 0):.2f}%")
    logger.info(f"Sign Accuracy: {overall['sign_acc']:.2f}%")
    logger.info(f"Valid Rate: {overall['valid_rate']:.2f}%")
    logger.info(f"Execution Time: {execution_time:.2f}s")
    
    # Log magnitude breakdown
    logger.info("-" * 50)
    logger.info("By Magnitude:")
    for _, row in magnitude_df.iterrows():
        logger.info(
            f"  {row['bucket']}: {row['count']} samples, "
            f"Top-1 {row['top1_acc']:.1f}%"
        )
    
    # Log mode breakdown
    logger.info("-" * 50)
    logger.info("By Solver Mode:")
    for _, row in mode_df.iterrows():
        logger.info(
            f"  {row['mode']}: {row['count']} ({row['usage_rate']:.1f}%), "
            f"Top-1 {row['top1_acc']:.1f}%"
        )
    
    # Save outputs
    logger.info("-" * 50)
    save_results_csv(results, output_dir / "solver_results.csv")
    save_summary_json(overall, magnitude_df, mode_df, execution_time, output_dir / "summary.json")
    magnitude_df.to_csv(output_dir / "magnitude_breakdown.csv", index=False)
    mode_df.to_csv(output_dir / "mode_breakdown.csv", index=False)
    save_config_json(args, output_dir / "analysis_config.json")
    
    logger.info("=" * 50)
    logger.info("Done!")


if __name__ == "__main__":
    main()
