# Implementation Specification: `src/intseq_bert/analysis/analyze_cases.py`

## 1. Overview

This script generates **case study visualizations** for representative integer sequences. It visualizes internal model behavior, including uncertainty and periodic patterns, to check whether the model is learning structure rather than memorizing examples. It supports **IntSeqBERT**, **Vanilla Transformer**, and **Ablation (No-Mod)** model types.

### Key Features

1. **Magnitude and uncertainty plot:** Growth trajectory with predictive uncertainty.
2. **Sign-probability plot:** Sign-class probability transitions.
3. **Modulo-spectrum heatmap:** Periodicity fingerprints by modulus.
4. **Attention / Summary Panel:** Attention heatmap for supported models; summary panel otherwise.

---

## 2. Dependencies

### Libraries

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple
```

### Internal Modules

```python
from intseq_bert import config
from intseq_bert.analysis.common import (
    ModelWrapper,
    create_model_wrapper,
    split_mod_logits,
    get_mod_index,
    LOG_VAR_CLIP_MIN,
    LOG_VAR_CLIP_MAX,
)
```

---

## 3. Command-Line Arguments (CLI)

```bash
python -m intseq_bert.analysis.analyze_cases \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045,A000040,A000290 \
    --output_dir results/case_studies \
    --model_type intseq \
    --features_dir data/oeis/features
```

### Argument List

| Argument | Type | Required | Default | Description |
|------|------|------|-----------|------|
| `--checkpoint` | str | yes | - | Checkpoint path |
| `--oeis_ids` | str | yes | - | Comma-separated list of OEIS IDs |
| `--output_dir` | str | yes | - | Output directory |
| `--model_type` | str | | `intseq` | Model type (`intseq`, `vanilla`, `ablation`) |
| `--features_dir` | str | | `data/oeis/features` | Feature directory |
| `--jsonl_path` | str | | `None` | JSONL fallback path when `.pt` files are unavailable |
| `--device` | str | | `auto` | Device |
| `--figsize` | str | | `12,10` | Figure size (`width,height`) |
| `--dpi` | int | | `150` | Output resolution |

---

## 4. Target Sequences (Archetypes)

Default set of representative sequences. This can be overridden from the command line.

```python
DEFAULT_ARCHETYPES = {
    "linear_recurrence": "A000045",   # Fibonacci
    "polynomial": "A000290",          # Squares (n^2)
    "sign_oscillation": "A033999",    # Alternating (-1)^n
    "number_theory": "A000040",       # Primes
    "super_growth": "A000142",        # Factorial (n!)
}
```

| Category | OEIS ID | Sequence Name | Purpose |
|---------|---------|--------|-----------|
| Linear Recurrence | A000045 | Fibonacci | Check periodicity stripes |
| Polynomial | A000290 | Squares | Check tracking of growth curves |
| Sign Oscillation | A033999 | Alternating | Check separation of sign oscillation patterns |
| Number Theory | A000040 | Primes | Check whether uncertainty is honest |
| Super Growth | A000142 | Factorial | Check tracking of rapid growth |

---

## 5. Visualization Panels

Use a 2x2 layout containing four panels in one image.

### 5.1. Panel 1: Magnitude and Uncertainty

```python
def plot_magnitude_uncertainty(
    ax: plt.Axes,
    positions: np.ndarray,      # (L,)
    ground_truth: np.ndarray,   # (L,) model magnitude scale
    pred_mu: np.ndarray,        # (L,)
    pred_sigma: np.ndarray,     # (L,) = sqrt(exp(log_var))
    mask: np.ndarray            # (L,) prediction target positions
):
    """
    Visualize the growth trajectory and uncertainty.

    - Blue solid line: ground-truth magnitude target
    - Red dashed line: predicted mean mu
    - Red band: uncertainty band of +/-2 sigma
    """
    ax.plot(positions, ground_truth, 'b-', label='Ground Truth', linewidth=2)
    ax.plot(positions[mask], pred_mu[mask], 'r--', label='Predicted mu', linewidth=1.5)

    # Uncertainty band (masked positions only)
    ax.fill_between(
        positions[mask],
        pred_mu[mask] - 2 * pred_sigma[mask],
        pred_mu[mask] + 2 * pred_sigma[mask],
        color='red', alpha=0.2, label='+/-2 sigma'
    )

    ax.set_xlabel('Position n')
    ax.set_ylabel('log₁₀(|x|)')
    ax.set_title('Magnitude & Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
```

### 5.2. Panel 2: Sign Probability

```python
def plot_sign_probability(
    ax: plt.Axes,
    positions: np.ndarray,      # (L,)
    sign_probs: np.ndarray,     # (L, 3) [P(+), P(-), P(0)]
    ground_truth_sign: np.ndarray  # (L,) 0=+, 1=-, 2=0
):
    """
    Stacked area chart of sign-class probabilities.

    - Blue: Positive
    - Red: Negative
    - Gray: Zero
    """
    ax.stackplot(
        positions,
        sign_probs[:, 0],  # Positive
        sign_probs[:, 1],  # Negative
        sign_probs[:, 2],  # Zero
        labels=['Positive', 'Negative', 'Zero'],
        colors=['#2196F3', '#F44336', '#9E9E9E'],
        alpha=0.8
    )

    # Ground-truth markers
    for i, sign in enumerate(ground_truth_sign):
        marker_y = 0.95 if sign == 0 else (0.5 if sign == 1 else 0.05)
        ax.plot(positions[i], marker_y, 'ko', markersize=3)

    ax.set_xlabel('Position n')
    ax.set_ylabel('Probability')
    ax.set_title('Sign Probability')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
```

### 5.3. Panel 3: Modulo-Spectrum Heatmap

```python
def plot_modulo_heatmap(
    ax: plt.Axes,
    positions: np.ndarray,          # (L,)
    mod_confidences: np.ndarray,    # (L, num_display_mods)
    display_mods: List[int],        # Moduli to display
    ground_truth_mod: np.ndarray,   # (L, num_display_mods) ground-truth residues
    fig=None
):
    """
    Heatmap of periodicity fingerprints.

    - X-axis: Position n
    - Y-axis: Modulus m
    - Color: Predicted probability assigned to the ground-truth class
    """
    im = ax.imshow(
        mod_confidences.T,
        aspect='auto',
        cmap='RdYlGn',
        vmin=0, vmax=1,
        origin='lower'
    )

    ax.set_xlabel('Position n')
    ax.set_ylabel('Modulus m')
    ax.set_title('Modulo Spectrum (Confidence on GT)')

    ax.set_yticks(range(len(display_mods)))
    ax.set_yticklabels(display_mods)

    plt.colorbar(im, ax=ax, label='P(correct)')
```

### 5.4. Panel 4: Attention Heatmap (Optional)

```python
def plot_attention_heatmap(
    ax: plt.Axes,
    attention_weights: np.ndarray,  # (L, L) averaged attention weights
    positions: np.ndarray,
    fig=None
):
    """
    Heatmap of attention patterns.

    - X-axis: Key position n'
    - Y-axis: Query position n
    - Color: Attention weight
    """
    im = ax.imshow(
        attention_weights,
        aspect='auto',
        cmap='Blues',
        vmin=0
    )

    ax.set_xlabel("Key Position n'")
    ax.set_ylabel('Query Position n')
    ax.set_title('Attention Pattern (Avg over heads)')
    plt.colorbar(im, ax=ax, label='Weight')
```

---

## 6. Model Abstraction Extensions

Use the shared `ModelWrapper` defined in `analysis/common.py`. It already provides default `predict_with_details()` and `supports_attention()` methods for case-study code paths.

### 6.1. Additional Interface

```python
class ModelWrapper(ABC):
    # ... existing methods from analysis/common.py

    def predict_with_details(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Default implementation delegates to predict()."""
        return self.predict(batch)

    def supports_attention(self) -> bool:
        """Return whether this wrapper exposes attention weights."""
        return False
```

### 6.2. Current Wrapper Behavior

Case-study inference uses `create_model_wrapper()` from `analysis/common.py`.
The default `predict_with_details()` method returns the same prediction dictionary as `predict()`, and `supports_attention()` returns `False` unless a wrapper explicitly overrides it.

Dedicated attention extraction is handled by `analyze_attention.py`, which patches transformer layers and registers hooks before inference.

---

## 7. Processing Flow

### 7.1. Main Flow

```text
1. Parse arguments and configure logging.
2. Create a model wrapper.
3. For each OEIS ID:
   a. Load the feature file.
   b. Run inference with predict_with_details.
   c. Generate a four-panel figure with generate_case_figure.
   d. Save the PNG.
```

### 7.2. `load_single_sequence` with Fallbacks

If a `.pt` file is unavailable, provide a fallback that generates features on the fly from raw data. This makes it fast to visualize an interesting sequence immediately.

```python
def load_single_sequence(
    oeis_id: str,
    features_dir: Path,
    raw_data_path: Optional[Path] = None,
    jsonl_path: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    """
    Load features for a single sequence and convert them into batch format.

    Priority:
    1. Fast load from features_dir/{oeis_id}.pt if it exists.
    2. Search jsonl_path for the record and convert on the fly.
    3. Search raw_data_path (stripped.txt) and convert on the fly.

    Returns:
        {
            "mag_inputs": (1, L, 5),
            "mod_inputs": (1, L, 200),
            "mod_targets": (1, L, 100),
            "token_ids": (1, L),
            "attention_mask": (1, L),
            "oeis_id": str
        }
    """
    # 1. Fast load from an existing .pt file
    pt_path = features_dir / f"{oeis_id}.pt"
    if pt_path.exists():
        data = torch.load(pt_path)
        mag_tensor = data["mag_features"]
        if mag_tensor.size(-1) == config.MAG_RAW_DIM:
            is_masked = torch.zeros(mag_tensor.size(0), 1)
            mag_tensor = torch.cat([mag_tensor, is_masked], dim=-1)
        result = {
            "mag_inputs": mag_tensor.unsqueeze(0),
            "mod_inputs": data["mod_features"].unsqueeze(0),
            "mod_targets": data["mod_integers"].unsqueeze(0),
            "attention_mask": torch.ones(1, data["mag_features"].size(0)),
            "oeis_id": oeis_id
        }
        _add_token_ids(result)
        return result

    # 2. Convert on the fly from JSONL
    if jsonl_path and jsonl_path.exists():
        record = _find_record_in_jsonl(oeis_id, jsonl_path)
        if record:
            return _convert_record_to_features(record)

    # 3. Convert on the fly from raw text
    if raw_data_path and raw_data_path.exists():
        sequence = _find_sequence_in_raw(oeis_id, raw_data_path)
        if sequence:
            return _convert_sequence_to_features(oeis_id, sequence)

    raise FileNotFoundError(
        f"Feature file not found: {pt_path}. "
        f"Provide --jsonl_path or --raw_data_path for on-the-fly conversion."
    )


def _find_record_in_jsonl(oeis_id: str, jsonl_path: Path) -> Optional[Dict]:
    """Find the record with the specified ID in JSONL."""
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record.get("oeis_id") == oeis_id:
                return record
    return None


def _convert_record_to_features(record: Dict) -> Dict[str, torch.Tensor]:
    """Convert a JSONL record into feature tensors."""
    from intseq_bert import config
    from intseq_bert.features import process_sequence

    sequence = record["sequence"]
    features = process_sequence(sequence)
    mag_tensor = features[config.KEY_MAG_FEATURES]
    if mag_tensor.size(-1) == config.MAG_RAW_DIM:
        is_masked = torch.zeros(mag_tensor.size(0), 1)
        mag_tensor = torch.cat([mag_tensor, is_masked], dim=-1)

    result = {
        "mag_inputs": mag_tensor.unsqueeze(0),
        "mod_inputs": features[config.KEY_MOD_FEATURES].unsqueeze(0),
        "mod_targets": features[config.KEY_MOD_INTEGERS].unsqueeze(0),
        "attention_mask": torch.ones(1, mag_tensor.size(0)),
        "oeis_id": record["oeis_id"]
    }
    _add_token_ids(result)
    return result
```

### 7.3. `generate_case_figure`

```python
# Default display_mods sorted by structure:
# - Primes, i.e. number-theoretic structure, first.
# - Composites and base-10-related moduli later.
# This makes it easy to spot base-10 bias, such as strong colors only at mod 10 or 100.
DEFAULT_DISPLAY_MODS = [
    # Primes (Number Theory)
    2, 3, 5, 7, 11, 13,
    # Composites / highly composite numbers
    4, 6, 12,
    # Base-10 related, for bias detection
    10, 100
]

def generate_case_figure(
    oeis_id: str,
    model: ModelWrapper,
    batch: Dict,
    output_path: Path,
    display_mods: List[int] = None,  # Use DEFAULT_DISPLAY_MODS when None
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 150
):
    if display_mods is None:
        display_mods = DEFAULT_DISPLAY_MODS
    """
    Generate a four-panel case study figure.
    """
    # Inference
    preds = model.predict_with_details(batch)

    # Extract ground truth
    gt_mag = batch["mag_inputs"][0, :, 0].numpy()  # model magnitude scale
    gt_sign = batch["mag_inputs"][0, :, 1:4].argmax(dim=-1).numpy()

    # Extract predictions
    pred_mu = preds["mag_mu"][0].cpu().numpy()
    pred_sigma = np.sqrt(np.exp(preds["mag_log_var"][0].cpu().numpy()))
    sign_probs = F.softmax(preds["sign_logits"][0], dim=-1).cpu().numpy()

    # Compute modulo confidence
    mod_confidences = _compute_mod_confidences(
        preds["mod_logits"][0],
        batch["mod_targets"][0],
        display_mods
    )

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Case Study: {oeis_id}', fontsize=14, fontweight='bold')

    L = gt_mag.shape[0]
    positions = np.arange(L)
    mask = np.ones(L, dtype=bool)  # Show all positions

    # Panel 1: magnitude and uncertainty
    plot_magnitude_uncertainty(axes[0, 0], positions, gt_mag, pred_mu, pred_sigma, mask)

    # Panel 2: sign probability
    plot_sign_probability(axes[0, 1], positions, sign_probs, gt_sign)

    # Panel 3: modulo heatmap
    plot_modulo_heatmap(axes[1, 0], positions, mod_confidences, display_mods, None, fig=fig)

    # Panel 4: Attention or Summary
    if model.supports_attention() and "attention_weights" in preds:
        plot_attention_heatmap(axes[1, 1], preds["attention_weights"].cpu().numpy(), positions, fig=fig)
    else:
        _plot_summary_metrics(axes[1, 1], preds, batch)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved: {output_path}")
```

### 7.4. `_compute_mod_confidences`

```python
def _compute_mod_confidences(
    mod_logits: torch.Tensor,      # (L, ~5150)
    mod_targets: torch.Tensor,     # (L, 100)
    display_mods: List[int]
) -> np.ndarray:
    """
    Compute the predicted probability assigned to the ground-truth class at each position.

    Returns:
        (L, len(display_mods))
    """
    split_logits = split_mod_logits(mod_logits)  # List of (L, m)

    confidences = []
    for m in display_mods:
        idx = get_mod_index(m)
        logits_m = split_logits[idx]  # (L, m)
        probs_m = F.softmax(logits_m, dim=-1)  # (L, m)
        targets_m = mod_targets[:, idx].long()  # (L,)

        # Avoid invalid gather from IGNORE_INDEX or out-of-range labels
        valid_mask = (targets_m >= 0) & (targets_m < m)
        safe_targets = targets_m.clone()
        safe_targets[~valid_mask] = 0
        conf_m = probs_m.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        conf_m[~valid_mask] = 0.0
        confidences.append(conf_m.cpu().numpy())

    return np.stack(confidences, axis=1)  # (L, len(display_mods))
```

---

## 8. Output Files

### 8.1. Directory Structure

```text
<output_dir>/
├── A000045.png
├── A000040.png
├── A000290.png
├── A033999.png
└── A000142.png
```

### 8.2. Figure File Naming

```python
def get_output_filename(oeis_id: str) -> str:
    return f"{oeis_id}.png"
```

---

## 9. Error Handling

| Situation | Handling |
|------|------|
| Feature file does not exist | `FileNotFoundError` plus skip and continue |
| Unsupported model output | Leave the corresponding panel blank or show a message |
| Matplotlib error | Log the error and continue |

---

## 10. Usage Examples

### Single-Model Case Study

```bash
# Default five sequences
python -m intseq_bert.analysis.analyze_cases \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045,A000040,A000290,A033999,A000142 \
    --output_dir results/case_studies \
    --model_type intseq
```

### Custom Sequences

```bash
python -m intseq_bert.analysis.analyze_cases \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000001,A000002,A000003 \
    --output_dir results/custom_cases
```
