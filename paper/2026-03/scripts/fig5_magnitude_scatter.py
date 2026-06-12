"""
Fig.5 Magnitude scatter plot (prediction vs. ground truth, colored by bucket)
CICM 2026 paper — Figure 5

Shows predicted magnitude vs. true magnitude for the three Large model variants
(IntSeqBERT / Vanilla / Ablation) in parallel panels.
Points are colored by magnitude bucket (Small / Medium / Large / Huge / Astronomical).

Steps:
  1. Run inference and save per-sample (gt, pred) values to CSV on the first run
  2. Load the CSV cache and plot

Output: paper/2026-03/figures/fig5_magnitude_scatter.pdf
      paper/2026-03/figures/fig5_magnitude_scatter.png
Data cache: results/2026-03-02/cache/scatter_cache_{model}.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

CKPT = REPO_ROOT / "checkpoints" / "large_std"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
CACHE_DIR = REPO_ROOT / "results" / "2026-03-02" / "cache"

MODELS = ["intseq", "vanilla", "ablation"]
LABELS = {"intseq": "IntSeqBERT", "vanilla": "Vanilla", "ablation": "Ablation"}
MODEL_TYPES = {"intseq": "intseq", "vanilla": "vanilla", "ablation": "ablation"}

# ── Bucket definitions (matches config.MAGNITUDE_BUCKETS) ────────────────
BUCKET_BOUNDS = [
    (0,  2,  "Small"),
    (2,  5,  "Medium"),
    (5,  20, "Large"),
    (20, 50, "Huge"),
    (50, float("inf"), "Astronomical"),
]
BUCKET_ORDER  = ["Small", "Medium", "Large", "Huge", "Astronomical"]
BUCKET_COLORS = {
    "Small":        "#3b82f6",   # blue
    "Medium":       "#10b981",   # green
    "Large":        "#f59e0b",   # yellow-orange
    "Huge":         "#ef4444",   # red
    "Astronomical": "#7c3aed",   # purple
}
BUCKET_MARKERS = {
    "Small":        "o",
    "Medium":       "o",
    "Large":        "s",
    "Huge":         "^",
    "Astronomical": "D",
}

def get_bucket(v: float) -> str:
    for lo, hi, name in BUCKET_BOUNDS:
        if lo <= v < hi:
            return name
    return "Astronomical"


# ── Step 1: inference -> CSV cache ──────────────────────────────────────
def collect_and_cache(model_name: str) -> Path:
    """
    Run model inference and save (gt, pred) values to CSV.
    Skip inference if the CSV already exists.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"scatter_cache_{model_name}.csv"
    if cache_path.exists():
        print(f"[{model_name}] Using cache: {cache_path}")
        return cache_path

    print(f"[{model_name}] Running inference ...")
    import torch
    from intseq_bert.analysis.analyze_magnitude import (
        collect_predictions, create_model_wrapper,
    )
    from intseq_bert.loader import load_dataset
    from intseq_bert.collator import OEISCollator
    from torch.utils.data import DataLoader

    ckpt_dir  = CKPT / model_name
    ckpt_file = str(ckpt_dir / "best_model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model_wrapper(MODEL_TYPES[model_name], ckpt_file, device)

    dataset  = load_dataset(split_type="std", split_name="test")
    collator = OEISCollator()
    loader   = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collator)

    preds = collect_predictions(model, loader)
    gt_mag   = preds["gt_mag"]   # (N, L)
    pred_mag = preds["pred_mag"] # (N, L)
    mask     = preds["mask"]     # (N, L)

    # flatten & mask
    gt_flat   = gt_mag[mask.bool()].numpy().astype(float)
    pred_flat = pred_mag[mask.bool()].numpy().astype(float)

    # buckets
    buckets = [get_bucket(float(v)) for v in gt_flat]

    df = pd.DataFrame({"gt": gt_flat, "pred": pred_flat, "bucket": buckets})
    df.to_csv(cache_path, index=False)
    print(f"[{model_name}] Saved cache: {cache_path}  ({len(df):,} samples)")
    return cache_path


# ── Step 2: plot ────────────────────────────────────────────────────────
SAMPLE_PER_BUCKET = 3000   # Maximum number of sampled points per bucket.

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
fig.subplots_adjust(left=0.06, right=0.88, top=0.96, bottom=0.13, wspace=0.30)

for ax, model_name in zip(axes, MODELS):
    cache_path = collect_and_cache(model_name)
    df = pd.read_csv(cache_path)

    # Sample by bucket for the scatter plot.
    frames = []
    for bname in BUCKET_ORDER:
        sub = df[df["bucket"] == bname]
        if len(sub) == 0:
            continue
        if len(sub) > SAMPLE_PER_BUCKET:
            sub = sub.sample(SAMPLE_PER_BUCKET, random_state=42)
        frames.append(sub)
    plot_df = pd.concat(frames, ignore_index=True)

    # Plot each bucket in reverse order for the legend.
    for bname in reversed(BUCKET_ORDER):
        sub = plot_df[plot_df["bucket"] == bname]
        if len(sub) == 0:
            continue
        alpha = 0.25 if bname in ("Small", "Medium") else 0.55
        size  = 3    if bname in ("Small", "Medium") else 8
        ax.scatter(
            sub["gt"], sub["pred"],
            c=BUCKET_COLORS[bname],
            marker=BUCKET_MARKERS[bname],
            s=size, alpha=alpha, linewidths=0,
            label=bname, zorder=3,
        )

    # y=x diagonal
    vmax = max(plot_df["gt"].max(), plot_df["pred"].max())
    vmin = min(plot_df["gt"].min(), plot_df["pred"].min())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.2, zorder=4, label="$y=x$")

    # R^2 computed on all points.
    df_all = pd.read_csv(cache_path)
    r2 = 1 - np.sum((df_all["gt"] - df_all["pred"])**2) / \
             np.sum((df_all["gt"] - df_all["gt"].mean())**2)
    ax.text(0.04, 0.96, f"$R^2 = {r2:.4f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.85),
            zorder=5)

    ax.set_title(LABELS[model_name], fontsize=12, fontweight="bold")
    ax.set_xlabel("Ground Truth (log$_{10}$ scale)", fontsize=10)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)

axes[0].set_ylabel("Prediction (log$_{10}$ scale)", fontsize=10)

# ── Legend (outside the figure on the right) ─────────────────────────────
# Build bucket legend entries with size and marker information.
import matplotlib.lines as mlines
legend_handles = []
for bname in BUCKET_ORDER:
    h = mlines.Line2D(
        [], [], color=BUCKET_COLORS[bname],
        marker=BUCKET_MARKERS[bname],
        linestyle="None",
        markersize=6 if bname in ("Small", "Medium") else 7,
        label=bname,
    )
    legend_handles.append(h)
legend_handles.append(
    mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="$y=x$")
)

fig.legend(
    handles=legend_handles,
    loc="center right",
    bbox_to_anchor=(1.0, 0.5),
    fontsize=9.5,
    framealpha=0.9,
    title="Bucket",
    title_fontsize=10,
)

# ── Save ─────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"fig5_magnitude_scatter.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
