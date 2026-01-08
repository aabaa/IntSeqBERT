"""
evaluate_final.py:
Final evaluation script for IntSeqBERT on the strict dataset.
"""

import json
import math
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from tqdm import tqdm
import torch

from intseq_bert import solver
from intseq_bert import loader


def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Final Evaluation for IntSeqBERT")
    
    # Model & Data paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--features_dir", type=str, required=True, help="Path to features directory (for splitting)")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to raw data_clean_strict.jsonl")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json")
    
    # Splitting parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed used in training")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    
    # Solver parameters
    parser.add_argument("--beam_width", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples (debug)")
    
    return parser.parse_args()


def get_test_ids_from_loader(
    features_dir: str,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Set[str]:
    """
    Use loader.py to reproduce the exact test split used during training.
    Returns set of OEIS IDs (e.g. "A000001") in the test set.
    """
    _, _, test_ds = loader.load_and_split_data(
        features_dir=features_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    # feature files are named "A000001.pt" -> stem is "A000001"
    return {p.stem for p in test_ds.feature_files}


def load_sequences_by_ids(jsonl_path: str, target_ids: Set[str], verbose: bool = True) -> List[Dict]:
    """
    Load raw sequences from JSONL for specific IDs.
    Strictly follows JSONL format: {"oeis_id": "...", "sequence": [...]}
    """
    data = []
    
    file_iter = open(jsonl_path, 'r', encoding='utf-8')
    if verbose:
        file_iter = tqdm(file_iter, desc="Scanning JSONL")
    
    try:
        for line in file_iter:
            try:
                record = json.loads(line)
                
                # Use strict keys from JSONL
                oid = record.get('oeis_id')
                
                if oid and oid in target_ids:
                    # Keep the record as is (or filter keys if memory is tight)
                    # We assume record has 'sequence' as per schema
                    data.append(record)
                    
            except json.JSONDecodeError:
                continue
    finally:
        if hasattr(file_iter, 'close'):
            file_iter.close()
    
    return data


def load_test_sequences(args) -> List[Dict]:
    """Load matching test sequences."""
    print(f"🔍 Reproducing test split using loader...")
    test_ids = get_test_ids_from_loader(
        args.features_dir,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    print(f"✅ Identified {len(test_ids)} sequences in the official Test Set.")
    
    print(f"📖 Loading raw sequences from {args.jsonl_path}...")
    test_data = load_sequences_by_ids(args.jsonl_path, test_ids)
    print(f"✅ Loaded {len(test_data)} raw sequences matching the test set.")
    
    return test_data


def calculate_metrics(target: int, candidates: List[Any], pred_mag: float) -> Dict[str, Any]:
    """Calculate evaluation metrics."""
    cand_values = [c[0] for c in candidates] if candidates else []
    
    metrics = {
        "top1": False,
        "top5": False,
        "top10": False,
        "mag_error": 0.0,
        "target_log_mag": 0.0
    }
    
    if cand_values:
        if target == cand_values[0]:
            metrics["top1"] = True
        if target in cand_values[:5]:
            metrics["top5"] = True
        if target in cand_values[:10]:
            metrics["top10"] = True
            
    try:
        log_tgt = math.log10(abs(target) + 1)
        log_pred = math.log10(abs(pred_mag) + 1)
        metrics["mag_error"] = abs(log_tgt - log_pred)
        metrics["target_log_mag"] = log_tgt
    except ValueError:
        pass
        
    return metrics


def update_results(
    results: Dict[str, Any],
    metrics: Dict[str, Any],
    record: Dict,
    output: Dict,
    log_sample: bool = False
) -> None:
    """Update results dictionary."""
    # Summary stats
    results["summary"]["total"] += 1
    if metrics["top1"]: results["summary"]["correct_top1"] += 1
    if metrics["top5"]: results["summary"]["correct_top5"] += 1
    if metrics["top10"]: results["summary"]["correct_top10"] += 1
    results["summary"]["total_mag_error"] += metrics["mag_error"]
    
    # Bucket analysis
    bucket = int(metrics["target_log_mag"])
    if bucket not in results["details_by_magnitude"]:
        results["details_by_magnitude"][bucket] = {"total": 0, "correct": 0}
    results["details_by_magnitude"][bucket]["total"] += 1
    if metrics["top1"]:
        results["details_by_magnitude"][bucket]["correct"] += 1
    
    # Logging
    if log_sample:
        # Use JSONL key 'oeis_id' for logging
        log_entry = {
            "oeis_id": record.get("oeis_id"),
            "target": record.get("sequence")[-1],
            "pred_top1": output['candidates'][0][0] if output['candidates'] else None,
            "candidates": [c[0] for c in output['candidates'][:3]],
            "correct": metrics["top1"],
            "mag_error": round(metrics["mag_error"], 4)
        }
        results["logs"].append(log_entry)


def create_empty_results(config: Optional[Dict] = None) -> Dict[str, Any]:
    return {
        "config": config or {},
        "summary": {
            "total": 0,
            "correct_top1": 0,
            "correct_top5": 0,
            "correct_top10": 0,
            "total_mag_error": 0.0
        },
        "details_by_magnitude": {},
        "logs": []
    }


def main():
    args = setup_args()
    
    # 1. Load Data
    test_data = load_test_sequences(args)
    
    if args.limit:
        test_data = test_data[:args.limit]
        print(f"⚠️ Limiting evaluation to first {args.limit} samples.")

    # 2. Initialize Solver
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Initializing Solver on {device}...")
    solver_instance = solver.IntSeqSolver(model_path=args.model_path, device=device)
    
    # 3. Evaluation Loop
    results = create_empty_results(vars(args))
    
    print("🚀 Starting Evaluation...")
    start_time = time.time()
    
    for i, record in enumerate(tqdm(test_data)):
        try:
            # Use 'sequence' directly (list of ints)
            full_seq = record.get('sequence')
            
            if not full_seq or len(full_seq) < 5:
                continue
            
            input_seq = full_seq[:-1]
            target = full_seq[-1]
            
            # Solve
            output = solver_instance.solve(
                input_seq, 
                top_k=args.top_k, 
                beam_width=args.beam_width
            )
            
            # Metrics
            metrics = calculate_metrics(target, output['candidates'], output['predicted_magnitude'])
            
            # Update
            log_sample = not metrics["top1"] or (i % 100 == 0)
            update_results(results, metrics, record, output, log_sample)
            
            # Periodic Save
            if (i + 1) % 500 == 0:
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)

        except Exception as e:
            # print(f"Error processing {record.get('oeis_id')}: {e}")
            continue

    # Final Report
    total = results["summary"]["total"]
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("🏆 FINAL RESULTS (Held-out Test Set)")
    print("="*50)
    print(f"Processed: {total} samples in {elapsed/60:.1f} min")
    
    if total > 0:
        acc1 = results["summary"]["correct_top1"] / total * 100
        acc5 = results["summary"]["correct_top5"] / total * 100
        mae = results["summary"]["total_mag_error"] / total
        
        print(f"Top-1 Accuracy : {acc1:.2f}%")
        print(f"Top-5 Accuracy : {acc5:.2f}%")
        print(f"Avg Mag Error  : {mae:.4f}")
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()