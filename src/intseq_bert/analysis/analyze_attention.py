"""
analyze_attention.py:
Attention pattern visualization and analysis for IntSeqBERT.

Performs layer-wise, head-wise, and aggregated attention analysis to verify
model understanding of sequence structure (e.g., recurrence patterns).
"""

import torch
import numpy as np
import pandas as pd
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from intseq_bert.analysis.common import (
    ModelWrapper,
    create_model_wrapper,
)
from intseq_bert.analysis.analyze_cases import load_single_sequence


# ==========================================
# Constants
# ==========================================

EXPECTED_PATTERNS = {
    "A000045": {"type": "linear_recurrence", "recurrence_depth": 2},  # Fibonacci: a_n = a_{n-1} + a_{n-2}
    "A000142": {"type": "linear_recurrence", "recurrence_depth": 1},  # Factorial: a_n = n * a_{n-1}
    "A000040": {"type": "non_local", "recurrence_depth": None},        # Primes: no local pattern
}


# ==========================================
# AttentionExtractor Class
# ==========================================

class AttentionExtractor:
    """Extract Attention Weights from Transformer Encoder layers."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attention_weights: List[torch.Tensor] = []
        self.hooks: List = []
    
    def register_hooks(self):
        """Register forward hooks on all EncoderLayer self_attn modules."""
        for layer in self._get_encoder_layers():
            hook = layer.self_attn.register_forward_hook(self._hook_fn)
            self.hooks.append(hook)
    
    def _hook_fn(self, module, input, output):
        """Store attention weights from output[1]."""
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
            if attn_weights is not None:
                logging.debug(f"Captured attention shape: {attn_weights.shape}")
                self.attention_weights.append(attn_weights.detach().cpu())
    
    def _get_encoder_layers(self):
        """Get encoder layers based on model type."""
        if hasattr(self.model, 'bert'):
            return self.model.bert.encoder.layers
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'encoder'):
            return self.model.backbone.encoder.layers
        elif hasattr(self.model, 'encoder'):
            return self.model.encoder.encoder.layers
        else:
            raise ValueError("Cannot find encoder layers in model")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear(self):
        """Clear collected attention weights."""
        self.attention_weights = []
    
    def get_attention_tensor(self) -> torch.Tensor:
        """
        Return stacked attention tensor.
        
        Returns:
            (num_layers, B, num_heads, L, L)
        """
        if not self.attention_weights:
            raise ValueError("No attention weights collected. Run forward pass first.")
        return torch.stack(self.attention_weights, dim=0)


# ==========================================
# Visualization Functions
# ==========================================

def plot_layerwise_attention(
    attention: torch.Tensor,
    output_path: Path,
    oeis_id: str,
    valid_len: Optional[int] = None,
    layer_ids: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150
):
    """
    Plot layer-wise average attention as a grid.
    
    Args:
        attention: (num_layers, num_heads, L, L)
        valid_len: If specified, trim to valid_len x valid_len
        layer_ids: Specific layers to plot (default: all)
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualization")
    
    num_layers = attention.size(0)
    if layer_ids is None:
        layer_ids = list(range(num_layers))
    
    # Average over heads
    layer_avg = attention.mean(dim=1)  # (num_layers, L, L)
    
    # Trim to valid length
    if valid_len is not None:
        layer_avg = layer_avg[:, :valid_len, :valid_len]
    
    ncols = min(4, len(layer_ids))
    nrows = (len(layer_ids) + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    im = None
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
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label='Attention Weight')
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved: {output_path}")


def plot_headwise_attention(
    attention: torch.Tensor,
    layer_id: int,
    output_path: Path,
    oeis_id: str,
    valid_len: Optional[int] = None,
    dpi: int = 150
):
    """
    Plot all heads for a given layer.
    
    Args:
        attention: (num_layers, num_heads, L, L)
        layer_id: Layer to visualize (-1 for last layer)
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualization")
    
    num_heads = attention.size(1)
    layer_attn = attention[layer_id]  # (num_heads, L, L)
    
    if valid_len is not None:
        layer_attn = layer_attn[:, :valid_len, :valid_len]
    
    ncols = min(4, num_heads)
    nrows = (num_heads + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.atleast_2d(axes)
    
    im = None
    for head_id in range(num_heads):
        row, col = divmod(head_id, ncols)
        ax = axes[row, col]
        
        im = ax.imshow(layer_attn[head_id].numpy(), cmap='Blues', vmin=0)
        ax.set_title(f'Head {head_id}')
        ax.set_xlabel("Key Pos")
        ax.set_ylabel("Query Pos")
    
    # Hide unused axes
    for idx in range(num_heads, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')
    
    actual_layer = layer_id if layer_id >= 0 else attention.size(0) + layer_id
    fig.suptitle(f'{oeis_id} - Layer {actual_layer} Heads', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label='Attention Weight')
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved: {output_path}")


def plot_aggregated_attention(
    attention: torch.Tensor,
    output_path: Path,
    oeis_id: str,
    valid_len: Optional[int] = None,
    dpi: int = 150
):
    """
    Plot aggregated attention (all layers, all heads average) with horizontal profile.
    
    Args:
        attention: (num_layers, num_heads, L, L)
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualization")
    
    # Average over all layers and heads
    avg_attn = attention.mean(dim=(0, 1)).numpy()  # (L, L)
    
    if valid_len is not None:
        avg_attn = avg_attn[:valid_len, :valid_len]
    
    L = avg_attn.shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Heatmap
    im = axes[0].imshow(avg_attn, cmap='Blues', vmin=0)
    axes[0].set_title(f'Aggregated Attention (L={L})')
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")
    fig.colorbar(im, ax=axes[0], label='Attention Weight')
    
    # Panel 2: Horizontal Profile (relative position of max attention)
    max_key_pos = avg_attn.argmax(axis=1)
    relative_pos = max_key_pos - np.arange(L)
    
    axes[1].bar(range(L), relative_pos, color='steelblue', alpha=0.7)
    axes[1].axhline(y=-1, color='red', linestyle='--', label='n-1 (prev)')
    axes[1].axhline(y=-2, color='orange', linestyle='--', label='n-2')
    axes[1].set_xlabel('Query Position n')
    axes[1].set_ylabel('Relative Key Position (max attn)')
    axes[1].set_title('Attention Focus Offset')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Attention Analysis: {oeis_id}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved: {output_path}")


# ==========================================
# Recurrence Pattern Analysis
# ==========================================

def analyze_recurrence_pattern(attention: torch.Tensor) -> Dict[str, float]:
    """
    Quantify attention to adjacent positions (recurrence pattern detection).
    
    Args:
        attention: (num_layers, num_heads, L, L)
    
    Returns:
        {
            "prev_1_ratio": ratio of attention to n-1,
            "prev_2_ratio": ratio of attention to n-2,
            "diagonal_ratio": ratio of self-attention,
            "total_local_ratio": ratio within |offset| <= 2
        }
    """
    avg_attn = attention.mean(dim=(0, 1)).numpy()  # (L, L)
    L = avg_attn.shape[0]
    
    prev_1_sum = 0.0
    prev_2_sum = 0.0
    diag_sum = 0.0
    total = 0.0
    
    for q in range(L):
        row_sum = avg_attn[q].sum()
        total += row_sum
        
        # Diagonal (self-attention)
        diag_sum += avg_attn[q, q]
        
        # n-1
        if q >= 1:
            prev_1_sum += avg_attn[q, q - 1]
        
        # n-2
        if q >= 2:
            prev_2_sum += avg_attn[q, q - 2]
    
    # Local ratio (|offset| <= 2)
    local_sum = 0.0
    for q in range(L):
        for offset in range(-2, 3):
            k = q + offset
            if 0 <= k < L:
                local_sum += avg_attn[q, k]
    
    return {
        "prev_1_ratio": prev_1_sum / total if total > 0 else 0.0,
        "prev_2_ratio": prev_2_sum / total if total > 0 else 0.0,
        "diagonal_ratio": diag_sum / total if total > 0 else 0.0,
        "total_local_ratio": local_sum / total if total > 0 else 0.0,
    }


def check_pattern_alignment(
    oeis_id: str,
    recurrence_stats: Dict[str, float]
) -> str:
    """
    Check if observed attention pattern aligns with expected pattern.
    
    Returns:
        "ALIGNED" | "MISALIGNED" | "UNKNOWN"
    """
    if oeis_id not in EXPECTED_PATTERNS:
        return "UNKNOWN"
    
    expected = EXPECTED_PATTERNS[oeis_id]
    
    if expected["type"] == "linear_recurrence":
        # Should have high local attention
        if recurrence_stats["total_local_ratio"] > 0.5:
            return "ALIGNED"
        else:
            return "MISALIGNED"
    
    elif expected["type"] == "non_local":
        # Should not have strong local pattern
        if recurrence_stats["total_local_ratio"] < 0.3:
            return "ALIGNED"
        else:
            return "MISALIGNED"
    
    return "UNKNOWN"


# ==========================================
# CLI
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Attention Pattern Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--oeis_ids", type=str, required=True, help="Comma-separated OEIS IDs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_type", type=str, default="intseq", help="Model type")
    parser.add_argument("--features_dir", type=str, default="data/oeis/features", help="Features directory")
    parser.add_argument("--layer_ids", type=str, default="all", help="Layer IDs (comma-separated or 'all')")
    parser.add_argument("--head_ids", type=str, default="all", help="Head IDs (comma-separated or 'all')")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--figsize", type=str, default="16,12", help="Figure size")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    return parser.parse_args()



def _patch_attention_layers(model: torch.nn.Module):
    """
    Monkey-patch TransformerEncoderLayer._sa_block to force need_weights=True.
    Standard PyTorch implementation hardcodes need_weights=False for efficiency.
    """
    import types
    
    def _get_layers(model):
        if hasattr(model, 'bert'):
            return model.bert.encoder.layers
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
             return model.backbone.encoder.layers
        elif hasattr(model, 'encoder'):
             # Check if encoder is directly the TransformerEncoder
             if hasattr(model.encoder, 'layers'):
                 return model.encoder.layers
             # Fallback for nested structure: IntSeqModel -> encoder -> layers
             elif hasattr(model.encoder, 'encoder'):
                 return model.encoder.encoder.layers
        raise ValueError("Cannot find encoder layers to patch")

    layers = _get_layers(model)
    
    def _sa_block_patched(self, x, attn_mask, key_padding_mask, is_causal=False):
        # Force need_weights=True to capture attention
        # Force average_attn_weights=False to get per-head attention
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True,
                           average_attn_weights=False,
                           is_causal=is_causal)[0]
        return self.dropout1(x)

    logging.info(f"Patching {len(layers)} encoder layers to capture attention weights...")
    for layer in layers:
        # Check signature of _sa_block to be safe (PyTorch version compat)
        # Assuming standard signature as implemented above
        # Monkey patch the instance method
        layer._sa_block = types.MethodType(_sa_block_patched, layer)


def main(args=None):
    if args is None:
        args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_dir = Path(args.features_dir)
    
    # Parse figsize
    try:
        figsize = tuple(int(x) for x in args.figsize.split(","))
        if len(figsize) != 2:
            raise ValueError
    except ValueError:
        logging.warning(f"Invalid figsize '{args.figsize}', using default (16, 12)")
        figsize = (16, 12)
    
    # Parse layer_ids
    layer_ids = None
    if args.layer_ids != "all":
        layer_ids = [int(x) for x in args.layer_ids.split(",")]
    
    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    model = create_model_wrapper(args.model_type, args.checkpoint, device)
    
    # Setup attention extractor
    # Patch model to enable attention weights
    _patch_attention_layers(model.model)
    
    # Setup attention extractor
    extractor = AttentionExtractor(model.model)
    extractor.register_hooks()
    
    

    
    results = []
    
    for oeis_id in args.oeis_ids.split(","):
        oeis_id = oeis_id.strip()
        logging.info(f"Processing: {oeis_id}")
        
        try:
            batch = load_single_sequence(oeis_id, features_dir)
            
            # Get valid length (excluding padding)
            valid_len = int(batch["attention_mask"].sum().item())
            
            # Forward pass (attention collected via hook)
            extractor.clear()
            _ = model.predict(batch)
            
            # Get attention: (num_layers, num_heads, L, L)
            attention = extractor.get_attention_tensor()[:, 0]  # Remove batch dim
            
            # Visualizations
            plot_layerwise_attention(
                attention,
                output_dir / f"{oeis_id}_layerwise.png",
                oeis_id,
                valid_len=valid_len,
                layer_ids=layer_ids,
                figsize=figsize,
                dpi=args.dpi
            )
            
            plot_headwise_attention(
                attention,
                layer_id=-1,  # Last layer
                output_path=output_dir / f"{oeis_id}_heads_last.png",
                oeis_id=oeis_id,
                valid_len=valid_len,
                dpi=args.dpi
            )
            
            plot_aggregated_attention(
                attention,
                output_dir / f"{oeis_id}_aggregated.png",
                oeis_id,
                valid_len=valid_len,
                dpi=args.dpi
            )
            
            # Recurrence analysis
            stats = analyze_recurrence_pattern(attention)
            alignment = check_pattern_alignment(oeis_id, stats)
            
            results.append({
                "oeis_id": oeis_id,
                **stats,
                "pattern_alignment": alignment
            })
            
            logging.info(f"  -> local_ratio={stats['total_local_ratio']:.2f}, alignment={alignment}")
            
        except Exception as e:
            logging.error(f"Error processing {oeis_id}: {e}")
            continue
    
    extractor.remove_hooks()
    
    # Save summary CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_dir / "attention_summary.csv", index=False)
        logging.info(f"Saved summary to {output_dir / 'attention_summary.csv'}")
    
    logging.info("Done!")


if __name__ == "__main__":
    main()
