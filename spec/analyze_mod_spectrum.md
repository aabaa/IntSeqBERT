# Implementation Specification: `src/intseq_bert/analysis/analyze_mod_spectrum.py`

## 1. Overview

This script performs **modulo-spectrum analysis** for a trained model. It compares all moduli `m` from 2 to 101 under equal conditions and ranks the structures the model handles well. It supports **IntSeqBERT**, **Vanilla Transformer**, and **Ablation (No-Mod)** model types.

### Key Features

1. **Global Ranking:** Compute NIG (Normalized Information Gain) for each modulus `m` and rank them.
2. **Tag-stratified analysis:** Stratified analysis by OEIS tag.
3. **Bootstrap CI:** 95% confidence interval estimation for statistical significance.

---

## 2. Dependencies

### Libraries

```python
import torch
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple
from torch.utils.data import DataLoader
```

### Internal Modules

```python
from intseq_bert import config
from intseq_bert.loader import load_dataset, OEISDataset
from intseq_bert.collator import OEISCollator
from intseq_bert.analysis.common import (
    create_model_wrapper,
    ModelWrapper,
    split_mod_logits,
)
```

---

## 3. Command-Line Arguments (CLI)

```bash
python -m intseq_bert.analysis.analyze_mod_spectrum \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --split_name test \
    --output_dir results/analysis \
    --model_type intseq \
    --jsonl_path data/oeis/data.jsonl \
    --batch_size 64 \
    --bootstrap_samples 1000
```

### Argument List

| Argument | Type | Required | Default | Description |
|------|------|------|-----------|------|
| `--checkpoint` | str | yes | - | Checkpoint path |
| `--split_type` | str | yes | - | Split type, e.g. `std` or `easy` |
| `--split_name` | str | | `test` | Split name (`train`, `val`, `test`) |
| `--output_dir` | str | yes | - | Output directory |
| `--model_type` | str | | `intseq` | Model type (`intseq`, `vanilla`, `ablation`) |
| `--jsonl_path` | str | | `data/oeis/data.jsonl` | OEIS JSONL path for tag metadata |
| `--batch_size` | int | | `64` | Batch size |
| `--bootstrap_samples` | int | | `1000` | Number of bootstrap samples |
| `--seed` | int | | `42` | Random seed for bootstrap resampling |
| `--quiet` | flag | | `False` | Suppress progress bars where supported |
| `--device` | str | | `auto` | Device (`cuda`, `cpu`, `auto`) |

---

## 4. Model Abstraction

Use the shared `ModelWrapper` utilities from `analysis/common.py` to handle different model types uniformly.

### 4.1. `ModelWrapper` (Abstract Base)

```python
class ModelWrapper(ABC):
    """Abstract model wrapper."""

    @abstractmethod
    def predict(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                "mag_mu": (B, L),           # Magnitude predicted mean
                "mag_log_var": (B, L),      # Magnitude uncertainty
                "sign_logits": (B, L, 3),   # Sign logits
                "mod_logits": (B, L, ~5150) # Concatenated modulo logits
            }
        """
        pass

    @abstractmethod
    def get_mod_log_probs(self, mod_logits: torch.Tensor) -> List[torch.Tensor]:
        """Return per-modulus log probabilities."""
        pass
```

### 4.2. `IntSeqWrapper`

```python
class IntSeqWrapper(ModelWrapper):
    """Wrapper for IntSeqForPreTraining."""

    def __init__(self, checkpoint_path: str, device: str):
        self.model = IntSeqForPreTraining.from_checkpoint(checkpoint_path)
        self.model.to(device).eval()
        self.device = device

    def predict(self, batch: Dict) -> Dict:
        with torch.no_grad():
            outputs = self.model(
                mag_features=batch["mag_inputs"].to(self.device),
                mod_features=batch["mod_inputs"].to(self.device),
                src_key_padding_mask=(batch["attention_mask"] == 0).to(self.device)
            )
        return outputs["predictions"]

    def get_mod_log_probs(self, mod_logits: torch.Tensor) -> List[torch.Tensor]:
        split_logits = split_mod_logits(mod_logits)
        return [F.log_softmax(logits, dim=-1) for logits in split_logits]
```

### 4.3. `VanillaWrapper`

```python
class VanillaWrapper(ModelWrapper):
    """Wrapper for VanillaTransformerForPreTraining."""

    def __init__(self, checkpoint_path: str, device: str):
        self.model = VanillaTransformerForPreTraining.from_checkpoint(checkpoint_path)
        self.model.to(device).eval()
        self.device = device

    def predict(self, batch: Dict) -> Dict:
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["token_ids"].to(self.device),
                src_key_padding_mask=(batch["attention_mask"] == 0).to(self.device)
            )
        return outputs["predictions"]

    def get_mod_log_probs(self, mod_logits: torch.Tensor) -> List[torch.Tensor]:
        split_logits = split_mod_logits(mod_logits)
        return [F.log_softmax(logits, dim=-1) for logits in split_logits]
```

### 4.4. Factory Function

```python
def create_model_wrapper(
    model_type: str,
    checkpoint_path: str,
    device: str
) -> ModelWrapper:
    """Create the appropriate wrapper for a model type."""
    if model_type == "intseq":
        return IntSeqWrapper(checkpoint_path, device)
    elif model_type == "vanilla":
        return VanillaWrapper(checkpoint_path, device)
    elif model_type == "ablation":
        return AblationWrapper(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
```

---

## 5. Evaluation Metrics

### 5.1. Normalized Information Gain (NIG)

Because the number of classes, and thus difficulty, differs by modulus, use a metric normalized by maximum entropy.

```python
def compute_nig(ce_loss: float, modulus: int) -> float:
    """
    Formula: R(m) = 1.0 - (Loss / log(m))

    Args:
        ce_loss: Mean cross-entropy loss
        modulus: Modulus m

    Returns:
        NIG score (1.0 = perfect, 0.0 = random, < 0 = worse than random)
    """
    max_entropy = np.log(modulus)
    return 1.0 - (ce_loss / max_entropy)
```

### 5.2. Per-Modulus Metrics

Compute the following for each modulus:

| Metric | Computation | Description |
|------|------|------|
| `accuracy` | `(pred == target).mean()` | Classification accuracy (%) |
| `ce_loss` | `CrossEntropy(logits, targets).mean()` | Mean cross-entropy loss |
| `nig_score` | `1 - ce_loss / log(m)` | Normalized Information Gain |

---

## 6. Processing Flow

### 6.1. Main Flow

```text
1. Parse arguments and configure logging.
2. Create model wrapper with create_model_wrapper.
3. Prepare dataset and DataLoader.
4. Run streaming inference with `StreamingEvaluator.process_batch`.
5. Compute per-modulus metrics with `compute_mod_metrics_from_stats`.
6. Estimate bootstrap confidence intervals with `bootstrap_ci_from_stats`.
7. Run tag-stratified analysis with `tag_stratified_analysis_from_stats`.
8. Generate output files.
```

### 6.2. Streaming Evaluation for Memory Efficiency

For large datasets, e.g. 30k sequences, keeping all predictions in memory (`logits: 20003 x 128 x 5150`) consumes over 70 GB and causes OOM. Use **Streaming Evaluation** instead.

1. **Batch-wise aggregation:**
   - Process predictions (`mod_logits`) and labels (`mod_labels`) one batch at a time.
   - For each batch, compute and accumulate only the loss sum (`loss_sum`), accuracy count (`acc_sum`), and valid counts (`counts`).
   - Discard the large logits tensor immediately after processing the batch.

2. **Deferred computation:**
   - After all batches are processed, compute overall mean loss, accuracy, and NIG from accumulated sufficient statistics.
   - This keeps memory usage approximately constant with respect to dataset size.

```python
class StreamingEvaluator:
    """
    Evaluates model metrics batch-by-batch to avoid OOM.
    Stores per-sample statistics instead of full logits.
    """
    def process_batch(self, preds: Dict, batch: Dict):
        # Compute batch statistics and append them to self.results
        pass

    def finalize(self) -> Dict[str, torch.Tensor]:
        # Merge and return statistics from all batches
        pass
```

> **Important: Collator output compatibility**
>
> Data loading must be identical between training (`train.py`) and analysis.
> In particular, confirm that `OEISCollator` returns `mod_labels` for all 100 moduli.

### 6.3. `compute_mod_metrics_from_stats`

Compute global metrics for each modulus from accumulated streaming statistics.

```python
def compute_mod_metrics_from_stats(
    stats: Dict[str, torch.Tensor]
) -> pd.DataFrame:
    """
    Returns:
        DataFrame with columns: [modulus, accuracy, ce_loss, nig_score]
    """
    loss_sum = stats["loss_sum"]  # (N, 100)
    acc_sum = stats["acc_sum"]    # (N, 100)
    counts = stats["counts"]      # (N,)
    # Aggregate over sequences, then compute cross-entropy, accuracy, and NIG per modulus.
```

### 6.4. `bootstrap_ci_from_stats`

Estimate 95% confidence intervals for NIG scores by resampling sequence-level statistics.

```python
def bootstrap_ci_from_stats(
    stats: Dict[str, torch.Tensor],
    n_samples: int = 1000,
    ci_level: float = 0.95,
    seed: int = None,
    quiet: bool = False
) -> pd.DataFrame:
    """
    Returns:
        DataFrame with columns: [modulus, nig_lower, nig_upper]
    """
    # Resample rows of stats["loss_sum"] and stats["counts"],
    # recompute NIG, then return percentile bounds.
```

---

## 7. Tag-Stratified Analysis

### 7.1. Load OEIS Tags

```python
def load_oeis_tags(jsonl_path: str) -> Dict[str, List[str]]:
    """
    Returns:
        {oeis_id: [tag1, tag2, ...], ...}
    """
    id_to_tags = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            id_to_tags[record["oeis_id"]] = record.get("keywords", [])
    return id_to_tags
```

### 7.2. `tag_stratified_analysis_from_stats`

```python
def tag_stratified_analysis_from_stats(
    stats: Dict[str, torch.Tensor],
    id_to_tags: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Returns:
        DataFrame with columns: [tag, count, overall_acc, non_base10_acc, nig_score, top_modulus]
    """
    # tag -> sequence-index list
    tag_to_indices = defaultdict(list)
    for i, oeis_id in enumerate(stats["oeis_ids"]):
        for tag in id_to_tags.get(oeis_id, []):
            tag_to_indices[tag].append(i)

    results = []
    for tag, indices in tag_to_indices.items():
        if len(indices) < 10:  # Minimum 10 samples
            continue

        indices_t = torch.tensor(indices, dtype=torch.long)
        tag_counts = stats["counts"][indices_t].sum().item()
        if tag_counts == 0:
            continue

        tag_loss = stats["loss_sum"][indices_t].sum(dim=0) / tag_counts
        tag_acc = (stats["acc_sum"][indices_t].sum(dim=0) / tag_counts) * 100
        metrics = pd.DataFrame([
            {
                "modulus": m,
                "accuracy": tag_acc[i].item(),
                "nig_score": compute_nig(tag_loss[i].item(), m),
            }
            for i, m in enumerate(config.MOD_RANGE)
        ])
        top_row = metrics.loc[metrics["nig_score"].idxmax()]

        results.append({
            "tag": tag,
            "count": len(indices),
            "overall_acc": metrics["accuracy"].mean(),
            "non_base10_acc": _compute_non_base10_acc(metrics),
            "nig_score": metrics["nig_score"].mean(),
            "top_modulus": int(top_row["modulus"])
        })

    return pd.DataFrame(results).sort_values("nig_score", ascending=False)


def _compute_non_base10_acc(metrics: pd.DataFrame) -> float:
    """
    Mean accuracy after excluding base-10-related moduli (10, 20, 50, 100).

    Terminology:
    - "Trivial solution": a case where |y| < m and residue computation is unnecessary.
    - "Base-dependent": a modulus derived from decimal notation, such as mod 10 or 100.

    This function excludes the latter, i.e. base-dependent moduli.
    It is used as a metric for measuring number-theoretic structure understanding.
    """
    base10_related_mods = {10, 20, 50, 100}  # Base-10-related moduli
    non_base10 = metrics[~metrics["modulus"].isin(base10_related_mods)]
    return non_base10["accuracy"].mean()
```

---

## 8. Output Files

### 8.1. Directory Structure

```text
results/analysis/mod/
├── mod_spectrum_ranking.csv      # NIG ranking by modulus
├── mod_spectrum_with_ci.csv      # With bootstrap confidence intervals
├── tag_performance.csv           # Tag-stratified analysis
├── analysis_config.json          # Run configuration
└── figures/
    └── mod_spectrum_bar.png      # Optional bar chart
```

### 8.2. `mod_spectrum_ranking.csv`

```csv
rank,modulus,accuracy,ce_loss,nig_score,interpretation
1,2,92.5,0.104,0.85,Parity (Odd/Even)
2,3,78.2,0.243,0.78,Ternary Pattern
3,10,42.1,0.645,0.72,Base-10 Pattern
...
```

### 8.3. `mod_spectrum_with_ci.csv`

```csv
modulus,nig_lower,nig_upper
2,0.83,0.87
3,0.75,0.81
...
```

### 8.4. `tag_performance.csv`

Extended tag-wise metrics.

| Column | Description |
|--------|------|
| `tag` | Tag name |
| `count` | Number of samples |
| `overall_acc` | Mean accuracy across all moduli |
| `non_base10_acc` | Mean accuracy excluding base-10 moduli |
| `nig_score` | Mean NIG score |
| `top_modulus` | Highest-scoring modulus |
| `acc_mod_2`, `acc_mod_3`, `acc_mod_5`, `acc_mod_10`, `acc_mod_100` | Individual accuracies for key moduli |
| `base10_bias` | Decimal-base bias (`acc_mod_10` - `non_base10_acc`) |
| `top_5_mods_nig` | Top 5 moduli by NIG, e.g. `"2(0.85); 3(0.78); ..."` |
| `worst_5_mods_nig` | Bottom 5 moduli by NIG |
| `mag_mse` | Magnitude MSE, reserved for future extension and currently N/A |
| `mag_acc` | Magnitude accuracy, reserved for future extension and currently N/A |

```csv
tag,count,overall_acc,non_base10_acc,nig_score,top_modulus,acc_mod_2,acc_mod_3,acc_mod_5,acc_mod_10,acc_mod_100,base10_bias,top_5_mods_nig,worst_5_mods_nig,mag_mse,mag_acc
mult,850,65.2,60.5,0.68,2,92.5,78.2,65.0,42.1,35.0,18.4,"2(0.85); 3(0.78); 4(0.72); 6(0.70); 8(0.68)","98(0.10); 99(0.12); 97(0.15); 101(0.18); 95(0.20)",,
prime,400,55.0,52.0,0.60,2,88.0,70.5,55.0,38.0,28.0,14.0,"2(0.80); 3(0.68); ...",...,,
...
```

### 8.5. `analysis_config.json`

```json
{
  "checkpoint": "checkpoints/intseq_std/best_model.pt",
  "model_type": "intseq",
  "split_type": "std",
  "split_name": "test",
  "bootstrap_samples": 1000,
  "seed": 42
}
```

---

## 9. Implementation Notes

### 9.1. Interpretation Mapping

Automatically assign interpretations to moduli with high NIG.

```python
INTERPRETATION_MAP = {
    # Basic periodicity
    2: "Parity (Odd/Even)",
    3: "Mod-3 (digit-sum residue)",
    4: "Last 2 Bits",
    5: "Last Digit (Base-5)",
    6: "LCM(2,3) - 2 & 3 Combined",
    7: "Prime",
    8: "Last 3 Bits",
    9: "Mod-9 (Digital Root)",

    # Base-10 related, as an indicator of notation dependence
    10: "Base-10 (Last Digit)",
    20: "Base-10 Multiple",
    50: "Base-10 Multiple",
    100: "Base-10 (Last 2 Digits)",

    # Highly composite numbers
    12: "Highly Composite (LCM(3,4))",
    24: "Highly Composite",
    60: "Sexagesimal Base",

    # Large primes
    101: "Large Prime (Near 100)",
    97: "Large Prime",
}

def get_interpretation(modulus: int) -> str:
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
```

### 9.2. Error Handling

| Situation | Handling |
|------|------|
| Checkpoint does not exist | `FileNotFoundError` |
| Unknown model type | `ValueError` |
| Empty dataset | `ValueError` |
| CUDA OOM | Automatically reduce batch size or exit with an error |
