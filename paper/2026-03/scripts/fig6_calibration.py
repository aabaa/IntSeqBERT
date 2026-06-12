"""
Fig.6 calibration curves (uncertainty calibration)
CICM 2026 paper — Figure 6

X-axis: predicted uncertainty sigma (mean over bin)
Y-axis: actual RMSE (empirical error within the bin)
y=x: perfect calibration line
Upper region (red): overconfident (underpredicted sigma);
lower region (blue): over-uncertain (overpredicted sigma).

Three-panel layout (IntSeqBERT / Vanilla / Ablation, Large models).
The x-axis uses a log scale because Vanilla has an extremely wide sigma range (0 -> 46).

Data: checkpoints/large_std/{model}/analysis/magnitude/calibration_data.csv
       checkpoints/large_std/{model}/analysis/magnitude/overall_metrics.csv

Output: paper/2026-03/figures/fig6_calibration.pdf
      paper/2026-03/figures/fig6_calibration.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT      = REPO_ROOT / "checkpoints" / "large_std"
OUT_DIR   = Path(__file__).resolve().parent.parent / "figures"

MODELS  = ["intseq", "vanilla", "ablation"]
LABELS  = {"intseq": "IntSeqBERT", "vanilla": "Vanilla", "ablation": "Ablation"}
COLORS  = {"intseq": "#1f77b4", "vanilla": "#ff7f0e", "ablation": "#2ca02c"}

# ── Data loading ─────────────────────────────────────────────────────────
def load_cal(model: str):
    cal_path  = CKPT / model / "analysis" / "magnitude" / "calibration_data.csv"
    met_path  = CKPT / model / "analysis" / "magnitude" / "overall_metrics.csv"
    cal_df = pd.read_csv(cal_path)
    metrics = pd.read_csv(met_path).iloc[0]
    return cal_df, float(metrics["ece"])

# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.14, wspace=0.32)

for ax, model in zip(axes, MODELS):
    cal_df, ece = load_cal(model)
    x = cal_df["mean_sigma"].values.astype(float)
    y = cal_df["rmse"].values.astype(float)

    # ── Shared axis range: minimum and maximum ───────────────────────────
    x_lo = max(1e-3, x.min() * 0.8)
    x_hi = x.max() * 1.2
    y_lo = 0.0
    y_hi = max(x_hi, y.max() * 1.1)   # Ensure the y=x line remains visible up to x_hi.

    # ── Perfect calibration line (y=x) ──────────────────────────────────
    xs = np.logspace(np.log10(x_lo), np.log10(x_hi), 300)
    ax.plot(xs, xs, color="#cc0000", linestyle="--", linewidth=1.4,
            label="Perfect ($y=x$)", zorder=3)

    # ── Overconfidence / over-uncertainty background regions ────────────
    ax.fill_between(xs, xs, y_hi, color="#fde8e8", alpha=0.55,
                    zorder=0, label="Overconfident")
    ax.fill_between(xs, 0,  xs,  color="#e8edf8", alpha=0.55,
                    zorder=0, label="Over-uncertain")

    # ── Calibration curve ───────────────────────────────────────────────
    ax.plot(x, y, color=COLORS[model], linewidth=1.8,
            marker="o", markersize=6, zorder=4,
            label=LABELS[model])
    # Add white marker outlines.
    ax.scatter(x, y, color=COLORS[model], s=36, edgecolors="white",
               linewidths=0.6, zorder=5)

    # ── ECE annotation ──────────────────────────────────────────────────
    ax.text(0.96, 0.96,
            f"ECE = {ece:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
            zorder=6)

    # ── Axes ────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(LABELS[model], fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Predicted $\sigma$ (log scale)", fontsize=10)
    ax.grid(which="major", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

axes[0].set_ylabel("Actual RMSE", fontsize=10)

# ── Legend (inside the right panel) ──────────────────────────────────────
# Shared legend: show only perfect calibration / overconfident / over-uncertain regions.
handles = [
    plt.Line2D([0], [0], color="#cc0000", linestyle="--", linewidth=1.4,
               label="Perfect calibration ($y=x$)"),
    plt.Rectangle((0, 0), 1, 1, fc="#fde8e8", alpha=0.7,
                  label="Overconfident region"),
    plt.Rectangle((0, 0), 1, 1, fc="#e8edf8", alpha=0.7,
                  label="Over-uncertain region"),
]
axes[2].legend(handles=handles, loc="upper left", fontsize=8.5,
               framealpha=0.9, handlelength=1.5)

# ── Save ─────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"fig6_calibration.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
