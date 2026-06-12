# Implementation Specification: `src/intseq_bert/analysis/analyze_solver.py`

## Table of Contents

1. [Overview](#1-overview)
2. [Dependencies](#2-dependencies)
3. [Command-Line Arguments](#3-command-line-arguments)
4. [Processing Flow](#4-processing-flow)
5. [Analysis Metrics](#5-analysis-metrics)
6. [Output Files](#6-output-files)
7. [Implementation Details](#7-implementation-details)
8. [Error Handling](#8-error-handling)

---

## 1. Overview

This script combines a trained model with `solver.py`, reconstructs the value of the next term for test data, and measures **exact-match accuracy**.

### Evaluation Points

1. **Exact match:** Whether the predicted integer exactly matches the target.
2. **Top-k accuracy:** Whether the correct answer appears within the configured top-k candidates.
3. **Performance by solver mode:** Performance for dense, sieve, and CRT modes.
4. **Performance by magnitude:** Which numeric ranges are solved well, from small values to huge integers.

---

## 2. Dependencies

### Libraries

```python
import torch
import pandas as pd
import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
```

### Internal Modules

```python
from intseq_bert import config
from intseq_bert.solver import IntegerSolver, VanillaSolver
from intseq_bert.analysis.common import create_model_wrapper, ModelWrapper
from intseq_bert.features import process_sequence
from intseq_bert.collator import OEISCollator
```

### Configuration (`config.py`)

| Constant | Value | Purpose |
|------|------|------|
| `MOD_RANGE` | `list(range(2, 102))` | List of moduli |
| `MAGNITUDE_BUCKETS` | list | Thresholds for magnitude buckets |
| `SOLVER_TOP_K_DEFAULT` | 5 | Default number of candidates |
| `VANILLA_MASK_TOKEN_ID` | 1 | Mask token ID for the Vanilla Transformer |
| `VANILLA_SPECIAL_TOKENS_OFFSET` | 3 | Offset between tokens and integers |

---

## 3. Command-Line Arguments

```bash
python -m intseq_bert.analysis.analyze_solver \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --split_name test \
    --output_dir results/solver_analysis \
    --max_samples 1000 \
    --top_k 5
```

### Argument List

| Argument | Type | Required | Default | Description |
|------|------|------|-----------|------|
| `--checkpoint` | str | yes | - | Model checkpoint path |
| `--model_type` | str | | `intseq` | Model type (`intseq`, `vanilla`, `ablation`) |
| `--split_type` | str | yes | - | Split type, e.g. `std` or `easy` |
| `--split_name` | str | | `test` | Split name (`train`, `val`, `test`) |
| `--output_dir` | str | yes | - | Output directory |
| `--data_root` | str | | `config.DATA_ROOT` | Data root directory |
| `--max_samples` | int | | `1000` | Maximum number of samples to evaluate |
| `--top_k` | int | | `5` | Number of candidates returned by the solver |
| `--filter_magnitude` | str | | `None` | Test only a specific magnitude range (`small`, `medium`, `large`, `huge`, `astronomical`) |
| `--device` | str | | `auto` | Device (`cuda`, `cpu`, `auto`) |

---

## 4. Processing Flow

### 4.1. Main Flow

```text
1. Parse arguments and configure logging.
2. Load the model, restoring configuration from the checkpoint.
3. Load test data from JSONL.
4. Run the inference loop sample by sample.
5. Aggregate results.
6. Generate output files.
```

### 4.2. Step 1: Data Preparation

The normal `DataLoader` converts values into tensors such as magnitude and modulo features, but exact-match evaluation requires the **raw integer before conversion**.

For this reason, read JSONL directly and perform the following:

1. Split the final term from the sequence as the target.
2. Tensorize the input sequence using `features.process_sequence()`.
3. Keep the target as a Python `int`.

> **Sequence-length assumption:** In preprocessing (`preprocess.py`), `config.MIN_SEQUENCE_LENGTH = 10` applies an explicit filter. Therefore, the std test split does not contain sequences shorter than 10 terms. Observed statistics: minimum 10 terms, median 36 terms, mean 42.5 terms.
> The `len(seq) < 2` filter remains as a code-level guard, but it is unnecessary for the actual dataset.
> Solver input context is therefore always guaranteed to contain **at least 9 terms**, and the extremely difficult case of predicting from no context, i.e. one term only, is not included in this evaluation.

```python
def load_test_samples(
    jsonl_path: Path,
    split_ids: List[str],
    max_samples: int
) -> List[Dict]:
    """
    Returns:
        List of dicts with keys:
          - oeis_id: str
          - input_seq: List[int] (without target)
          - target: int (ground-truth integer)
          - target_str: str (string representation of the ground truth)
    """
```

### 4.3. Step 2: Inference Loop

Run the following for each sample:

```python
for sample in tqdm(samples):
    # 1. Append a dummy next-term slot and create features
    seq_with_dummy = sample["input_seq"] + [0]
    features_dict = process_sequence(seq_with_dummy)

    # 2. Convert to batch format (B=1)
    batch = collator([features_dict])

    # 3. Explicitly mask the appended final position to avoid leakage
    batch["mag_inputs"][:, -1, :config.MAG_RAW_DIM] = 0.0
    batch["mag_inputs"][:, -1, -1] = 1.0
    batch["mod_inputs"][:, -1, :] = 0.0

    # 4. Forward pass through the model wrapper
    with torch.no_grad():
        predictions = model_wrapper.predict(batch)

    # 5. Extract solver parameters from the appended final position, i.e. next-term prediction
    last_pos = batch["attention_mask"].sum(dim=1).item() - 1
    args = IntegerSolver.from_model_output(
        predictions, position=last_pos, model=model_wrapper.model
    )

    # 6. Solve
    candidates = solver.solve(*args, top_k=top_k)

    # 7. Judge match rank
    match_rank = -1
    for rank, cand in enumerate(candidates, 1):
        if cand["value"] == sample["target"]:
            match_rank = rank
            break

    # 8. Record result
    results.append({
        "oeis_id": sample["oeis_id"],
        "target": sample["target"],
        "target_str": sample["target_str"],
        "pred_top1": candidates[0]["value"] if candidates else None,
        "match_rank": match_rank,
        "solver_mode": candidates[0]["method"] if candidates else "none",
        "mag_log10": get_log10_magnitude(sample["target"]),
        "score_top1": candidates[0]["score"] if candidates else None,
        "sign_pred": get_sign_idx(candidates[0]["value"]) if candidates else None,
        "sign_true": get_sign_idx(sample["target"])
    })
```

> **Note:** Masking
>
> After appending a dummy token (`0`) to the end of the input sequence, explicitly mask that position.
> This prevents data leakage and ensures that the model predicts without observing the dummy value.
>
> ```python
> # Masking, using the same convention as collator.py
> batch["mag_inputs"][:, -1, :config.MAG_RAW_DIM] = 0.0  # Zero out content
> batch["mag_inputs"][:, -1, -1] = 1.0  # Set is_masked flag to 1
> batch["mod_inputs"][:, -1, :] = 0.0  # Zero out sin/cos features
> ```

### 4.4. Step 3: Aggregate Results

Aggregate all sample-level results and compute summary statistics.

---

## 5. Analysis Metrics

### 5.1. Overall Metrics

| Metric | Formula | Description |
|------|----------|------|
| `top1_acc` | `(match_rank == 1).mean() x 100` | Top-1 exact match accuracy (%) |
| `top{top_k}_acc` | `(1 <= match_rank <= top_k).mean() x 100` | Fraction where the answer appears in the top-k candidates (%) |
| `sign_acc` | `(sign_pred == sign_true).mean() x 100` | Sign prediction accuracy (%) |
| `valid_rate` | `(match_rank != -1 or candidates not empty)` | Rate at which the solver returns a solution |

### 5.2. By Magnitude

Classification based on `config.MAGNITUDE_BUCKETS`:

| Bucket | Range (log10) | Numeric Range |
|----------|-------------|---------|
| Small | 0 ~ 2 | 1 ~ 100 |
| Medium | 2 ~ 5 | 100 ~ 100,000 |
| Large | 5 ~ 20 | `10^5 ~ 10^20` |
| Huge | 20 ~ 50 | `10^20 ~ 10^50` |
| Astronomical | 50+ | `10^50+` |

Compute top-1 accuracy and top-k accuracy for each bucket.

### 5.3. By Solver Mode

| Mode | Description |
|--------|------|
| `dense` | Mode A: exhaustive search |
| `sieve` | Mode AB: anchor sieve |
| `crt` | Mode B: sparse CRT |
| `vanilla_lm` | Vanilla Transformer, LM head only |
| `zero` | Immediate zero return |
| `none` | No solution |

Compute usage rate and accuracy for each mode.

### 5.4. Vanilla Transformer Support

When `model_type=vanilla`, use the following special handling:

1. **Solver switch:** Use `VanillaSolver` instead of `IntegerSolver`.
2. **Token ID masking:** Set the final position of `token_ids` to `VANILLA_MASK_TOKEN_ID`.
3. **Inference logic:** Predict token IDs directly from `lm_head` logits and convert them to integers.
4. **UNK handling:** If an out-of-vocabulary token (`UNK`) is predicted, mark `is_unk=True`.

```python
# Inference flow for the Vanilla Transformer
if model_type == "vanilla":
    solver = VanillaSolver()
    logits = VanillaSolver.from_model_output(
        predictions, position=last_pos, batch_idx=0
    )  # (vocab_size,)
    candidates = solver.solve(logits, top_k)
    # candidates: [{"value": int | None, "score": float, "is_unk": bool}, ...]
```

---

## 6. Output Files

### 6.1. Directory Structure

```text
results/solver_analysis/
├── solver_results.csv        # Detailed result for every sample
├── summary.json              # Aggregated summary
├── magnitude_breakdown.csv   # Aggregation by magnitude bucket
├── mode_breakdown.csv        # Aggregation by solver mode
└── analysis_config.json      # Run configuration
```

### 6.2. `solver_results.csv`

Detailed results for individual samples.

| Column | Type | Description |
|--------|------|------|
| `oeis_id` | str | OEIS ID |
| `target` | int | Ground-truth integer |
| `target_str` | str | String representation of the ground truth, for very large numbers |
| `pred_top1` | int | Top-1 predicted value |
| `match_rank` | int | Rank of the ground truth, `1..top_k` or `-1` for incorrect |
| `solver_mode` | str | Solver mode used |
| `mag_log10` | float | Target magnitude in `log10` |
| `score_top1` | float | Top-1 score |
| `sign_pred` | int | Predicted sign (0/1/2) |
| `sign_true` | int | Ground-truth sign (0/1/2) |

```csv
oeis_id,target,target_str,pred_top1,match_rank,solver_mode,mag_log10,score_top1,sign_pred,sign_true
A000045,13,13,13,1,dense,1.114,-0.05,0,0
A000040,101,101,99,-1,dense,2.004,-1.20,0,0
A123456,12345678901234567890,12345678901234567890,12345678901234567890,1,crt,19.091,-0.01,0,0
```

### 6.3. `summary.json`

The example below assumes the default `--top_k 5`; for other values, the dynamic key is `top{top_k}_acc`.

```json
{
  "overall": {
    "total_samples": 1000,
    "top1_acc": 45.2,
    "top5_acc": 62.8,
    "sign_acc": 98.5,
    "valid_rate": 99.1
  },
  "by_magnitude": {
    "Small": {"count": 450, "top1_acc": 72.3, "top5_acc": 85.1},
    "Medium": {"count": 320, "top1_acc": 48.5, "top5_acc": 65.2},
    "Large": {"count": 180, "top1_acc": 22.1, "top5_acc": 38.5},
    "Huge": {"count": 45, "top1_acc": 8.9, "top5_acc": 15.6},
    "Astronomical": {"count": 5, "top1_acc": 0.0, "top5_acc": 0.0}
  },
  "by_mode": {
    "dense": {"count": 850, "top1_acc": 52.1},
    "sieve": {"count": 100, "top1_acc": 28.0},
    "crt": {"count": 45, "top1_acc": 8.9},
    "zero": {"count": 5, "top1_acc": 100.0}
  },
  "execution": {
    "total_time_sec": 245.3,
    "avg_time_per_sample_sec": 0.245
  }
}
```

### 6.4. `magnitude_breakdown.csv`

The example below assumes the default `--top_k 5`; for other values, the dynamic columns are `top{top_k}_acc` and `top{top_k}_count`.

```csv
bucket,count,top1_acc,top5_acc,top1_count,top5_count
Small,450,72.3,85.1,325,383
Medium,320,48.5,65.2,155,209
Large,180,22.1,38.5,40,69
Huge,45,8.9,15.6,4,7
Astronomical,5,0.0,0.0,0,0
```

### 6.5. `mode_breakdown.csv`

The example below assumes the default `--top_k 5`; for other values, the dynamic column is `top{top_k}_acc`.

```csv
mode,count,usage_rate,top1_acc,top5_acc
dense,850,85.0,52.1,68.5
sieve,100,10.0,28.0,45.0
crt,45,4.5,8.9,15.6
zero,5,0.5,100.0,100.0
```

### 6.6. `analysis_config.json`

```json
{
  "checkpoint": "checkpoints/intseq_std/best_model.pt",
  "split_type": "std",
  "split_name": "test",
  "max_samples": 1000,
  "top_k": 5,
  "filter_magnitude": null,
  "device": "cuda",
  "timestamp": "2026-01-19 16:00:00"
}
```

---

## 7. Implementation Details

### 7.1. Retrieving Ground-Truth Data

Read the JSONL file directly, tensorize with logic equivalent to `process_sequence()`, and keep the **final term, i.e. the target, as a Python `int`**.

> **Important:** Loading through the existing `DataLoader` can lose very large integers through float conversion, so direct JSONL loading is used.

```python
def load_test_samples(jsonl_path: Path, split_ids: Set[str], max_samples: int):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["oeis_id"] not in split_ids:
                continue

            seq = record["sequence"]
            if len(seq) < 2:
                continue

            target = seq[-1]
            input_seq = seq[:-1]

            samples.append({
                "oeis_id": record["oeis_id"],
                "input_seq": input_seq,
                "target": target,
                "target_str": str(target)
            })

            if len(samples) >= max_samples:
                break

    return samples
```

### 7.2. Restoring Model Parameters

Restore `d_model`, `nhead`, and `num_layers` from the checkpoint.

```python
def load_model_from_checkpoint(model_type: str, checkpoint_path: Path, device: str):
    """Load the model using create_model_wrapper."""
    from intseq_bert.analysis.common import create_model_wrapper
    return create_model_wrapper(model_type, str(checkpoint_path), device)
```

### 7.3. Performance Estimates

| Number of Samples | Estimated Time (GPU) | Estimated Time (CPU) |
|-----------|---------------|---------------|
| 100 | ~30 sec | ~2 min |
| 1000 | ~5 min | ~20 min |
| All samples (2500) | ~12 min | ~50 min |

---

## 8. Error Handling

| Situation | Handling |
|------|------|
| Checkpoint does not exist | Raise `FileNotFoundError` |
| JSONL does not exist | Raise `FileNotFoundError` |
| Split file does not exist | Raise `FileNotFoundError` |
| Solver returns an empty list | Record `match_rank = -1`, `solver_mode = "none"` |
| Sequence is too short (`< 2`) | Skip |
| `log10` overflow on huge numbers | Guard `math.log10`, falling back to string length |

---

## 9. Usage Examples

### Basic Usage

```bash
python -m intseq_bert.analysis.analyze_solver \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --output_dir results/solver_analysis
```

### Test Only Small Numbers

```bash
python -m intseq_bert.analysis.analyze_solver \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --output_dir results/solver_small \
    --filter_magnitude small
```

### Full Evaluation

```bash
python -m intseq_bert.analysis.analyze_solver \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --output_dir results/solver_full \
    --max_samples 999999
```
