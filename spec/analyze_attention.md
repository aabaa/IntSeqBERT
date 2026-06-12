# Implementation Specification: `src/intseq_bert/analysis/analyze_attention.py`

## 1. Overview

This script specializes in **attention pattern visualization** for Transformer models. It analyzes which positions a trained model attends to during prediction and helps validate whether the model understands sequence structure.

> **Note:** This script is optional. `analyze_cases.py` also provides a simpler attention visualization.
> This script performs more detailed layer-wise and head-wise analysis.

### Key Features

1. **Layer-wise Attention:** Visualize attention patterns for each encoder layer.
2. **Head-wise Analysis:** Analyze specialization of individual heads.
3. **Aggregated View:** Summary view averaged across all layers and heads.
4. **Recurrence Detection:** Detect attention patterns toward neighboring terms.

---

## 2. Dependencies

### Libraries

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
```

### Internal Modules

```python
from intseq_bert.analysis.common import create_model_wrapper, ModelWrapper
from intseq_bert.analysis.analyze_cases import load_single_sequence
```

---

## 3. Command-Line Arguments (CLI)

```bash
python -m intseq_bert.analysis.analyze_attention \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045,A000142 \
    --output_dir results/attention_analysis \
    --model_type intseq \
    --layer_ids all \
    --head_ids all
```

### Argument List

| Argument | Type | Required | Default | Description |
|------|------|------|-----------|------|
| `--checkpoint` | str | yes | - | Checkpoint path |
| `--oeis_ids` | str | yes | - | Comma-separated list of OEIS IDs |
| `--output_dir` | str | yes | - | Output directory |
| `--model_type` | str | | `intseq` | Model type (`intseq`, `vanilla`, `ablation`) |
| `--features_dir` | str | | `data/oeis/features` | Feature directory |
| `--layer_ids` | str | | `all` | Layers to visualize (`all` or comma-separated IDs) |
| `--head_ids` | str | | `all` | Reserved argument; currently not used to filter drawn heads |
| `--device` | str | | `auto` | Device |
| `--figsize` | str | | `16,12` | Figure size |
| `--dpi` | int | | `150` | Output resolution |

---

## 4. Attention Extraction

### 4.1. `AttentionExtractor`

Collect attention weights from all layers using forward hooks.

```python
class AttentionExtractor:
    """Extract attention weights from a Transformer encoder."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attention_weights = []
        self.hooks = []

    def register_hooks(self):
        """Register hooks on self_attn for all EncoderLayer modules."""
        for layer in self._get_encoder_layers():
            hook = layer.self_attn.register_forward_hook(self._hook_fn)
            self.hooks.append(hook)

    def _hook_fn(self, module, input, output):
        """Save attention weights from output[1]."""
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]  # (B, num_heads, L, L)
            if attn_weights is not None:
                self.attention_weights.append(attn_weights.detach().cpu())

    def _get_encoder_layers(self):
        """Find EncoderLayer modules according to model type."""
        if hasattr(self.model, 'bert'):
            return self.model.bert.encoder.layers
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'encoder'):
            return self.model.backbone.encoder.layers
        elif hasattr(self.model, 'encoder'):
            if hasattr(self.model.encoder, 'layers'):
                return self.model.encoder.layers
            elif hasattr(self.model.encoder, 'encoder'):
                return self.model.encoder.encoder.layers
        else:
            raise ValueError("Cannot find encoder layers")

    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        """Clear collected weights."""
        self.attention_weights = []

    def get_attention_tensor(self) -> torch.Tensor:
        """
        Returns:
            (num_layers, B, num_heads, L, L)
        """
        return torch.stack(self.attention_weights, dim=0)
```

### 4.2. Usage with Padding Trimming

OEIS sequences are often only 30 to 50 terms long, but they are padded up to `config.MAX_SEQUENCE_LENGTH` (128). If the heatmap is drawn directly, meaningful patterns appear only in the upper-left portion and the remaining area is mostly blank.

**To draw only the valid region, compute `valid_len` and pass it to the visualization functions.**

```python
_patch_attention_layers(model.model)
extractor = AttentionExtractor(model.model)
extractor.register_hooks()

batch = load_single_sequence(oeis_id, features_dir)

# Get valid length, excluding padding
valid_len = batch["attention_mask"].sum().item()

with torch.no_grad():
    outputs = model.predict(batch)

attention = extractor.get_attention_tensor()  # (num_layers, 1, num_heads, L, L)
extractor.remove_hooks()

# Pass valid_len to visualization
plot_layerwise_attention(attention[:, 0], output_path, oeis_id, valid_len=valid_len)
```

---

## 5. Visualization

### 5.1. Layer-wise Grid with Padding Trimming

```python
def plot_layerwise_attention(
    attention: torch.Tensor,      # (num_layers, num_heads, L, L)
    output_path: Path,
    oeis_id: str,
    valid_len: Optional[int] = None,  # Valid length; trim when provided
    layer_ids: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Show average attention for each layer in a grid.

    Layout: 2 rows x (num_layers / 2) columns

    Note:
        If valid_len is provided, draw only the valid region excluding padding.
        This makes detailed patterns visible even for shorter sequences of 30 to 50 terms.
    """
    num_layers = attention.size(0)
    if layer_ids is None:
        layer_ids = list(range(num_layers))

    # Average heads in each layer
    layer_avg = attention.mean(dim=1)  # (num_layers, L, L)

    # Padding trim: crop if valid_len is provided
    if valid_len is not None:
        layer_avg = layer_avg[:, :valid_len, :valid_len]

    ncols = min(4, len(layer_ids))
    nrows = (len(layer_ids) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, layer_id in enumerate(layer_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        im = ax.imshow(layer_avg[layer_id].numpy(), cmap='Blues', vmin=0)
        ax.set_title(f'Layer {layer_id}')
        ax.set_xlabel("Key Pos")
        ax.set_ylabel("Query Pos")

    # Hide unused axes
    for idx in range(len(layer_ids), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    title = f'Layer-wise Attention: {oeis_id}'
    if valid_len is not None:
        title += f' (L={valid_len})'
    fig.suptitle(title, fontsize=14)
    plt.colorbar(im, ax=axes, shrink=0.6, label='Attention Weight')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

### 5.2. Head-wise Analysis

Analyze the specialization of each head.

```python
def plot_headwise_attention(
    attention: torch.Tensor,      # (num_layers, num_heads, L, L)
    layer_id: int,
    output_path: Path,
    oeis_id: str,
    valid_len: Optional[int] = None  # For padding trimming
):
    """
    Show all heads in a specified layer in a grid.
    """
    num_heads = attention.size(1)
    layer_attn = attention[layer_id]  # (num_heads, L, L)

    # Padding trim
    if valid_len is not None:
        layer_attn = layer_attn[:, :valid_len, :valid_len]

    ncols = min(4, num_heads)
    nrows = (num_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.atleast_2d(axes)

    for head_id in range(num_heads):
        row, col = divmod(head_id, ncols)
        ax = axes[row, col]

        im = ax.imshow(layer_attn[head_id].numpy(), cmap='Blues', vmin=0)
        ax.set_title(f'Head {head_id}')

    fig.suptitle(f'{oeis_id} - Layer {layer_id} Heads', fontsize=14)
    plt.colorbar(im, ax=axes, shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

### 5.3. Aggregated Summary

```python
def plot_aggregated_attention(
    attention: torch.Tensor,      # (num_layers, num_heads, L, L)
    output_path: Path,
    oeis_id: str,
    valid_len: Optional[int] = None  # For padding trimming
):
    """
    Mean attention over all layers and heads, plus a horizontal profile.
    """
    # Average across all layers and heads
    avg_attn = attention.mean(dim=(0, 1)).numpy()  # (L, L)

    # Padding trim
    if valid_len is not None:
        avg_attn = avg_attn[:valid_len, :valid_len]

    L = avg_attn.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Heatmap
    im = axes[0].imshow(avg_attn, cmap='Blues', vmin=0)
    axes[0].set_title(f'Aggregated Attention (L={L})')
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")
    plt.colorbar(im, ax=axes[0])

    # Panel 2: Horizontal profile, i.e. max-attention key for each query
    max_key_pos = avg_attn.argmax(axis=1)
    relative_pos = max_key_pos - np.arange(L)  # Relative position

    axes[1].bar(range(L), relative_pos, color='steelblue', alpha=0.7)
    axes[1].axhline(y=-1, color='red', linestyle='--', label='n-1 (prev)')
    axes[1].axhline(y=-2, color='orange', linestyle='--', label='n-2')
    axes[1].set_xlabel('Query Position n')
    axes[1].set_ylabel('Relative Key Position (max attn)')
    axes[1].set_title('Attention Focus Offset')
    axes[1].legend()

    fig.suptitle(f'Attention Analysis: {oeis_id}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

---

## 6. Recurrence Pattern Detection

### 6.1. Quantifying Attention to Neighboring Terms

For recurrence sequences such as Fibonacci, the relation `a_n = f(a_{n-1}, a_{n-2})` suggests stronger attention to `n-1` and `n-2`.

```python
def analyze_recurrence_pattern(
    attention: torch.Tensor      # (num_layers, num_heads, L, L)
) -> Dict[str, float]:
    """
    Quantify attention toward recurrence patterns.

    Returns:
        {
            "prev_1_ratio": float,  # Mean attention ratio to n-1
            "prev_2_ratio": float,  # Mean attention ratio to n-2
            "diagonal_ratio": float,  # Diagonal/self-attention ratio
            "total_local_ratio": float,  # Total ratio for |offset| <= 2
        }
    """
    avg_attn = attention.mean(dim=(0, 1)).numpy()  # (L, L)
    L = avg_attn.shape[0]

    prev_1_sum = 0
    prev_2_sum = 0
    diag_sum = 0
    total = 0

    for q in range(L):
        row_sum = avg_attn[q].sum()
        total += row_sum

        # Diagonal/self-attention
        diag_sum += avg_attn[q, q]

        # n-1
        if q >= 1:
            prev_1_sum += avg_attn[q, q - 1]

        # n-2
        if q >= 2:
            prev_2_sum += avg_attn[q, q - 2]

    # Local ratio (|offset| <= 2)
    local_sum = 0
    for q in range(L):
        for offset in range(-2, 3):
            k = q + offset
            if 0 <= k < L:
                local_sum += avg_attn[q, k]

    return {
        "prev_1_ratio": prev_1_sum / total if total > 0 else 0,
        "prev_2_ratio": prev_2_sum / total if total > 0 else 0,
        "diagonal_ratio": diag_sum / total if total > 0 else 0,
        "total_local_ratio": local_sum / total if total > 0 else 0,
    }
```

### 6.2. Alignment with Expected Patterns

```python
EXPECTED_PATTERNS = {
    "A000045": {"type": "linear_recurrence", "recurrence_depth": 2},  # Fibonacci: a_n = a_{n-1} + a_{n-2}
    "A000142": {"type": "linear_recurrence", "recurrence_depth": 1},  # Factorial: a_n = n * a_{n-1}
    "A000040": {"type": "non_local", "recurrence_depth": None},        # Primes: no local pattern
}

def check_pattern_alignment(
    oeis_id: str,
    recurrence_stats: Dict[str, float]
) -> str:
    """
    Check consistency with the expected pattern.

    Returns:
        "ALIGNED" | "MISALIGNED" | "UNKNOWN"
    """
    if oeis_id not in EXPECTED_PATTERNS:
        return "UNKNOWN"

    expected = EXPECTED_PATTERNS[oeis_id]

    if expected["type"] == "linear_recurrence":
        # Attention to neighboring terms should be strong
        if recurrence_stats["total_local_ratio"] > 0.5:
            return "ALIGNED"
        else:
            return "MISALIGNED"

    elif expected["type"] == "non_local":
        # No specific local pattern should dominate
        if recurrence_stats["total_local_ratio"] < 0.3:
            return "ALIGNED"
        else:
            return "MISALIGNED"

    return "UNKNOWN"
```

---

## 7. Processing Flow

### 7.1. Main Flow

```text
1. Parse arguments and configure logging.
2. Create model wrapper.
3. Monkey-patch `TransformerEncoderLayer._sa_block` to force `need_weights=True`.
4. Initialize AttentionExtractor.
5. For each OEIS ID:
   a. Load features.
   b. Extract attention.
   c. Generate visualizations:
      - Layer-wise grid
      - Head-wise view for the final layer
      - Aggregated summary
   d. Analyze recurrence pattern.
6. Write summary CSV.
```

### 7.2. Main Function

```python
def main(args):
    # Setup
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model_wrapper(args.model_type, args.checkpoint, device)
    _patch_attention_layers(model.model)

    extractor = AttentionExtractor(model.model)
    extractor.register_hooks()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for oeis_id in args.oeis_ids.split(","):
        logging.info(f"Processing: {oeis_id}")

        try:
            batch = load_single_sequence(oeis_id, Path(args.features_dir))

            # Get valid length, excluding padding
            valid_len = int(batch["attention_mask"].sum().item())

            # Forward pass; attention is collected via hooks
            extractor.clear()
            _ = model.predict(batch)

            attention = extractor.get_attention_tensor()[:, 0]  # Remove batch dim
            # attention: (num_layers, num_heads, L, L)

            # Visualizations, trimmed to valid length
            plot_layerwise_attention(
                attention,
                output_dir / f"{oeis_id}_layerwise.png",
                oeis_id,
                valid_len=valid_len
            )

            plot_headwise_attention(
                attention,
                layer_id=-1,  # Last layer
                output_path=output_dir / f"{oeis_id}_heads_last.png",
                oeis_id=oeis_id,
                valid_len=valid_len
            )

            plot_aggregated_attention(
                attention,
                output_dir / f"{oeis_id}_aggregated.png",
                oeis_id,
                valid_len=valid_len
            )

            # Recurrence analysis
            stats = analyze_recurrence_pattern(attention)
            alignment = check_pattern_alignment(oeis_id, stats)

            results.append({
                "oeis_id": oeis_id,
                **stats,
                "pattern_alignment": alignment
            })

        except Exception as e:
            logging.error(f"Error processing {oeis_id}: {e}")
            continue

    extractor.remove_hooks()

    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "attention_summary.csv", index=False)
    logging.info(f"Saved summary to {output_dir / 'attention_summary.csv'}")
```

---

## 8. Output Files

### 8.1. Directory Structure

```text
results/analysis/attention/
├── A000045_layerwise.png        # Attention grid for all layers
├── A000045_heads_last.png       # Heads in the final layer
├── A000045_aggregated.png       # Aggregated view plus recurrence analysis
├── A000142_layerwise.png
├── A000142_heads_last.png
├── A000142_aggregated.png
└── attention_summary.csv        # Recurrence pattern statistics
```

### 8.2. `attention_summary.csv`

```csv
oeis_id,prev_1_ratio,prev_2_ratio,diagonal_ratio,total_local_ratio,pattern_alignment
A000045,0.25,0.18,0.12,0.65,ALIGNED
A000142,0.35,0.08,0.15,0.62,ALIGNED
A000040,0.10,0.08,0.20,0.42,ALIGNED
```

---

## 9. Limitations

| Limitation | Description |
|------|------|
| PyTorch `TransformerEncoder` | Requires `need_weights=True`; the standard implementation uses `False`, so this script monkey-patches it |
| Vanilla Transformer | Can be supported with the same hook approach |
| Memory | Attention tensors (`L x L`) become large for long sequences |

---

## 10. Usage Examples

### Basic Usage

```bash
python -m intseq_bert.analysis.analyze_attention \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045,A000142,A000040 \
    --output_dir results/attention
```

### Specific Layers Only

```bash
python -m intseq_bert.analysis.analyze_attention \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045 \
    --output_dir results/attention \
    --layer_ids 0,3,5
```
