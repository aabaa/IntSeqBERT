"""
Fig.4 learning-curve script
CICM 2026 paper — Figure 4

Output: paper/2026-03/figures/fig4_learning_curves.pdf
      paper/2026-03/figures/fig4_learning_curves.png

Data source: checkpoints/{size}_std/{model}/history.csv
  Columns: epoch, train_loss, val_loss
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

SIZES   = ["Small", "Middle", "Large"]
SIZE_DIRS = {"Small": "small_std", "Middle": "middle_std", "Large": "large_std"}
MODELS  = ["intseq", "vanilla", "ablation"]
LABELS  = {"intseq": "IntSeqBERT", "vanilla": "Vanilla", "ablation": "Ablation"}
COLORS  = {"intseq": "#1f77b4", "vanilla": "#ff7f0e", "ablation": "#2ca02c"}
STYLES  = {"intseq": "-",        "vanilla": "--",     "ablation": "-."}

# ── Data loading ─────────────────────────────────────────────────────────
def load_history(size_key: str, model: str) -> pd.DataFrame:
    path = CHECKPOINT_ROOT / SIZE_DIRS[size_key] / model / "history.csv"
    df = pd.read_csv(path, usecols=["epoch", "train_loss", "val_loss"])
    return df.sort_values("epoch")


# ── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=False)
fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.13, wspace=0.32)

for ax, size in zip(axes, SIZES):
    for model in MODELS:
        df = load_history(size, model)
        ax.plot(
            df["epoch"], df["val_loss"],
            color=COLORS[model],
            linestyle=STYLES[model],
            linewidth=1.5,
            label=LABELS[model],
        )
    ax.set_title(size, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_xlim(1, 200)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(which="major", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

# Use a y-axis label only on the left panel.
axes[0].set_ylabel("Validation Loss", fontsize=11)

# Place the legend in the upper right of the Large panel.
axes[2].legend(loc="upper right", fontsize=10, framealpha=0.8)

# ── Save ─────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"fig4_learning_curves.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
