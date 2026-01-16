"""
analyze_cases.py:
Case Study Visualization for IntSeqBERT and comparison models.

Visualizes model predictions on representative sequences to verify
structural understanding rather than memorization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # Will fail gracefully if matplotlib not installed

from intseq_bert import config
from intseq_bert.analysis.common import (
    ModelWrapper,
    create_model_wrapper,
    split_mod_logits,
    get_mod_index,
    LOG_VAR_CLIP_MIN,
    LOG_VAR_CLIP_MAX,
)


# ==========================================
# Constants
# ==========================================

DEFAULT_ARCHETYPES = {
    "linear_recurrence": "A000045",   # Fibonacci
    "polynomial": "A000290",          # Squares (n²)
    "sign_oscillation": "A033999",    # Alternating (-1)^n
    "number_theory": "A000040",       # Primes
    "super_growth": "A000142",        # Factorial (n!)
}

# Structure-sorted display moduli for heatmap visualization
# Primes first, then composites, then Base-10 related (for bias detection)
DEFAULT_DISPLAY_MODS = [
    # Primes (Number Theory)
    2, 3, 5, 7, 11, 13,
    # Composites / Highly Composite
    4, 6, 12,
    # Base-10 related (bias detection)
    10, 100
]


# ==========================================
# Data Loading Functions
# ==========================================

def load_single_sequence(
    oeis_id: str,
    features_dir: Path,
    raw_data_path: Optional[Path] = None,
    jsonl_path: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    """
    Load features for a single sequence.
    
    Priority:
    1. features_dir/{oeis_id}.pt if exists (fast)
    2. jsonl_path - search for record and convert on-the-fly
    3. raw_data_path - search and convert on-the-fly
    
    Returns:
        {
            "mag_inputs": (1, L, MAG_EXTENDED_DIM),
            "mod_inputs": (1, L, MOD_FEATURE_DIM),
            "attention_mask": (1, L),
            "oeis_id": str
        }
    """
    # 1. Try loading from .pt file
    pt_path = features_dir / f"{oeis_id}.pt"
    if pt_path.exists():
        data = torch.load(pt_path, weights_only=False)
        return {
            "mag_inputs": data["mag_features"].unsqueeze(0),
            "mod_inputs": data["mod_features"].unsqueeze(0),
            "attention_mask": torch.ones(1, data["mag_features"].size(0)),
            "oeis_id": oeis_id
        }
    
    # 2. Try JSONL fallback
    if jsonl_path and jsonl_path.exists():
        record = _find_record_in_jsonl(oeis_id, jsonl_path)
        if record:
            return _convert_record_to_features(record)
    
    # 3. Try raw text fallback
    if raw_data_path and raw_data_path.exists():
        sequence = _find_sequence_in_raw(oeis_id, raw_data_path)
        if sequence:
            return _convert_sequence_to_features(oeis_id, sequence)
    
    raise FileNotFoundError(
        f"Feature file not found: {pt_path}. "
        f"Provide --jsonl_path or --raw_data_path for on-the-fly conversion."
    )


def _find_record_in_jsonl(oeis_id: str, jsonl_path: Path) -> Optional[Dict]:
    """Search for a record by OEIS ID in JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("oeis_id") == oeis_id:
                return record
    return None


def _find_sequence_in_raw(oeis_id: str, raw_data_path: Path) -> Optional[List[int]]:
    """Search for a sequence in raw stripped.txt format."""
    with open(raw_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(oeis_id):
                parts = line.strip().split(",")
                if len(parts) > 1:
                    return [int(x) for x in parts[1:] if x.strip()]
    return None


def _convert_record_to_features(record: Dict) -> Dict[str, torch.Tensor]:
    """Convert JSONL record to feature tensors."""
    from intseq_bert.features import extract_features
    
    sequence = record["values"]
    features = extract_features(sequence)
    
    return {
    mag_tensor = torch.tensor(features["mag_features"], dtype=torch.float32)
    # Pad to 5 dims: [log, s+, s-, s0] -> [log, s+, s-, s0, is_masked]
    if mag_tensor.size(-1) == 4:
        padding = torch.zeros(mag_tensor.size(0), 1)
        mag_tensor = torch.cat([mag_tensor, padding], dim=-1)
        
    return {
        "mag_inputs": mag_tensor.unsqueeze(0),
        "mod_inputs": torch.tensor(features["mod_features"], dtype=torch.float32).unsqueeze(0),
        "attention_mask": torch.ones(1, len(sequence)),
        "oeis_id": record["oeis_id"]
    }


def _convert_sequence_to_features(oeis_id: str, sequence: List[int]) -> Dict[str, torch.Tensor]:
    """Convert raw sequence to feature tensors."""
    from intseq_bert.features import extract_features
    
    features = extract_features(sequence)
    
    return {
    mag_tensor = torch.tensor(features["mag_features"], dtype=torch.float32)
    # Pad to 5 dims: [log, s+, s-, s0] -> [log, s+, s-, s0, is_masked]
    if mag_tensor.size(-1) == 4:
        padding = torch.zeros(mag_tensor.size(0), 1)
        mag_tensor = torch.cat([mag_tensor, padding], dim=-1)
        
    return {
        "mag_inputs": mag_tensor.unsqueeze(0),
        "mod_inputs": torch.tensor(features["mod_features"], dtype=torch.float32).unsqueeze(0),
        "attention_mask": torch.ones(1, len(sequence)),
        "oeis_id": oeis_id
    }


# ==========================================
# Visualization Functions
# ==========================================

def plot_magnitude_uncertainty(
    ax,
    positions: np.ndarray,
    ground_truth: np.ndarray,
    pred_mu: np.ndarray,
    pred_sigma: np.ndarray,
    mask: np.ndarray
):
    """
    Plot magnitude prediction with uncertainty band.
    
    - Blue solid: Ground Truth
    - Red dashed: Predicted μ
    - Red band: ±2σ uncertainty
    """
    ax.plot(positions, ground_truth, 'b-', label='Ground Truth', linewidth=2)
    ax.plot(positions[mask], pred_mu[mask], 'r--', label='Predicted μ', linewidth=1.5)
    
    # Uncertainty band
    ax.fill_between(
        positions[mask],
        pred_mu[mask] - 2 * pred_sigma[mask],
        pred_mu[mask] + 2 * pred_sigma[mask],
        color='red', alpha=0.2, label='±2σ'
    )
    
    ax.set_xlabel('Position n')
    ax.set_ylabel('log₁₀(|x|)')
    ax.set_title('Magnitude & Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_sign_probability(
    ax,
    positions: np.ndarray,
    sign_probs: np.ndarray,
    ground_truth_sign: np.ndarray
):
    """
    Plot sign class probabilities as stacked area.
    
    - Blue: Positive
    - Red: Negative
    - Gray: Zero
    """
    ax.stackplot(
        positions,
        sign_probs[:, 0],
        sign_probs[:, 1],
        sign_probs[:, 2],
        labels=['Positive', 'Negative', 'Zero'],
        colors=['#2196F3', '#F44336', '#9E9E9E'],
        alpha=0.8
    )
    
    # Ground truth markers
    for i, sign in enumerate(ground_truth_sign):
        marker_y = 0.95 if sign == 0 else (0.5 if sign == 1 else 0.05)
        ax.plot(positions[i], marker_y, 'ko', markersize=3)
    
    ax.set_xlabel('Position n')
    ax.set_ylabel('Probability')
    ax.set_title('Sign Probability')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)


def plot_modulo_heatmap(
    ax,
    positions: np.ndarray,
    mod_confidences: np.ndarray,
    display_mods: List[int],
    ground_truth_mod: Optional[np.ndarray],
    fig=None
):
    """
    Plot modulo spectrum heatmap.
    
    - X-axis: Position n
    - Y-axis: Modulus m
    - Color: Confidence on ground truth class
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
    
    # Y-axis labels
    ax.set_yticks(range(len(display_mods)))
    ax.set_yticklabels(display_mods)
    
    # Use figure colorbar if available, otherwise fall back to plt
    if fig is not None:
        fig.colorbar(im, ax=ax, label='P(correct)')
    elif plt is not None:
        plt.colorbar(im, ax=ax, label='P(correct)')


def plot_attention_heatmap(
    ax,
    attention_weights: np.ndarray,
    positions: np.ndarray,
    fig=None
):
    """
    Plot attention pattern heatmap.
    
    - X-axis: Key position
    - Y-axis: Query position
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
    
    if fig is not None:
        fig.colorbar(im, ax=ax, label='Weight')
    elif plt is not None:
        plt.colorbar(im, ax=ax, label='Weight')


def _plot_summary_metrics(ax, preds: Dict, batch: Dict):
    """Plot summary metrics when attention is not available."""
    ax.text(0.5, 0.5, 'Summary Metrics\n(Attention N/A)',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.axis('off')


# ==========================================
# Helper Functions
# ==========================================

def _compute_mod_confidences(
    mod_logits: torch.Tensor,
    mod_targets: torch.Tensor,
    display_mods: List[int]
) -> np.ndarray:
    """
    Compute confidence (probability on GT class) for each position and modulus.
    
    Args:
        mod_logits: (L, sum(MOD_RANGE))
        mod_targets: (L, 100) - ground truth remainders
        display_mods: List of moduli to display
    
    Returns:
        (L, len(display_mods)) array of confidences
    """
    split_logits_list = split_mod_logits(mod_logits)
    
    confidences = []
    for m in display_mods:
        idx = get_mod_index(m)
        logits_m = split_logits_list[idx]  # (L, m)
        probs_m = F.softmax(logits_m, dim=-1)  # (L, m)
        targets_m = mod_targets[:, idx].long()  # (L,)
        
        # Get probability of correct class
        conf_m = probs_m.gather(1, targets_m.unsqueeze(1)).squeeze(1)
        confidences.append(conf_m.cpu().numpy())
    
    return np.stack(confidences, axis=1)


# ==========================================
# Main Figure Generation
# ==========================================

def generate_case_figure(
    oeis_id: str,
    model: ModelWrapper,
    batch: Dict,
    output_path: Path,
    display_mods: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 150
):
    """
    Generate 4-panel case study figure.
    
    Panels:
    1. Magnitude & Uncertainty
    2. Sign Probability
    3. Modulo Spectrum Heatmap
    4. Attention or Summary
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualization")
    
    if display_mods is None:
        display_mods = DEFAULT_DISPLAY_MODS
    
    # Run inference
    preds = model.predict_with_details(batch)
    
    # Extract ground truth
    gt_mag = batch["mag_inputs"][0, :, 0].numpy()
    gt_sign = batch["mag_inputs"][0, :, 1:4].argmax(dim=-1).numpy()
    
    # Extract predictions
    pred_mu = preds["mag_mu"][0].cpu().numpy()
    pred_log_var = preds["mag_log_var"][0].cpu().numpy()
    pred_sigma = np.sqrt(np.exp(np.clip(pred_log_var, LOG_VAR_CLIP_MIN, LOG_VAR_CLIP_MAX)))
    sign_probs = F.softmax(preds["sign_logits"][0], dim=-1).cpu().numpy()
    
    # Compute modulo confidences
    mod_confidences = _compute_mod_confidences(
        preds["mod_logits"][0],
        batch["mod_inputs"][0],
        display_mods
    )
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Case Study: {oeis_id}', fontsize=14, fontweight='bold')
    
    L = gt_mag.shape[0]
    positions = np.arange(L)
    mask = np.ones(L, dtype=bool)
    
    # Panel 1: Magnitude & Uncertainty
    plot_magnitude_uncertainty(axes[0, 0], positions, gt_mag, pred_mu, pred_sigma, mask)
    
    # Panel 2: Sign Probability
    plot_sign_probability(axes[0, 1], positions, sign_probs, gt_sign)
    
    # Panel 3: Modulo Heatmap
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


# ==========================================
# CLI
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Case Study Visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--oeis_ids", type=str, required=True, help="Comma-separated OEIS IDs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_type", type=str, default="intseq", help="Model type")
    parser.add_argument("--features_dir", type=str, default="data/oeis/features", help="Features directory")
    parser.add_argument("--jsonl_path", type=str, default=None, help="JSONL path for fallback")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--figsize", type=str, default="12,10", help="Figure size (width,height)")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    
    # Setup
    logging.basicConfig(level=logging.INFO)
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_dir = Path(args.features_dir)
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else None
    
    # Parse figsize
    figsize = tuple(int(x) for x in args.figsize.split(","))
    
    # Load model
    model = create_model_wrapper(args.model_type, args.checkpoint, device)
    
    # Process each sequence
    oeis_ids = args.oeis_ids.split(",")
    for oeis_id in oeis_ids:
        logging.info(f"Processing: {oeis_id}")
        
        try:
            batch = load_single_sequence(oeis_id, features_dir, jsonl_path=jsonl_path)
            output_path = output_dir / f"{oeis_id}.png"
            generate_case_figure(oeis_id, model, batch, output_path, figsize=figsize, dpi=args.dpi)
        except Exception as e:
            logging.error(f"Error processing {oeis_id}: {e}")
            continue
    
    logging.info("Done!")


if __name__ == "__main__":
    main()
