"""
Fig.4b Train vs Val 学習曲線（過学習確認用）
CICM 2026 paper — Figure 4b

各パネル = モデルバリアント（IntSeqBERT / Vanilla / Ablation）
各パネル内 = Small / Middle / Large の Train（細点線）+ Val（太実線）

出力: experiment/cicm2026/fig4b_train_val_curves.pdf
      experiment/cicm2026/fig4b_train_val_curves.png

データソース: checkpoints/{size}_std/{model}/history.csv
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── パス設定 ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent

MODELS  = ["intseq", "vanilla", "ablation"]
LABELS  = {"intseq": "IntSeqBERT", "vanilla": "Vanilla", "ablation": "Ablation"}

SIZES    = ["Small", "Middle", "Large"]
SIZE_DIR = {"Small": "small_std", "Middle": "middle_std", "Large": "large_std"}
# サイズ別カラー（識別しやすい3色）
SIZE_COLORS = {"Small": "#2ca02c", "Middle": "#ff7f0e", "Large": "#1f77b4"}

def load_history(size_key: str, model: str) -> pd.DataFrame:
    path = CHECKPOINT_ROOT / SIZE_DIR[size_key] / model / "history.csv"
    df = pd.read_csv(path, usecols=["epoch", "train_loss", "val_loss"])
    return df.sort_values("epoch")

# ── プロット ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
fig.subplots_adjust(left=0.06, right=0.88, top=0.88, bottom=0.13, wspace=0.32)

for ax, model in zip(axes, MODELS):
    for size in SIZES:
        df = load_history(size, model)
        c = SIZE_COLORS[size]
        # Val Loss: 太実線
        ax.plot(df["epoch"], df["val_loss"],
                color=c, linestyle="-", linewidth=2.0,
                label=f"{size} Val")
        # Train Loss: 細点線
        ax.plot(df["epoch"], df["train_loss"],
                color=c, linestyle=":", linewidth=1.0,
                label=f"{size} Train")

    ax.set_title(LABELS[model], fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_xlim(1, 200)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(which="major", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

axes[0].set_ylabel("Loss", fontsize=11)

# ── 凡例（図の右外に配置）────────────────────────────────────────────────
# IntSeqBERT パネルのハンドルを再利用して右外に凡例を出す
handles, raw_labels = axes[0].get_legend_handles_labels()

# カスタム凡例アイテムを構築
from matplotlib.lines import Line2D
legend_items = []
for size in SIZES:
    c = SIZE_COLORS[size]
    legend_items.append(
        Line2D([0], [0], color=c, linewidth=2.0, linestyle="-", label=f"{size} — Val")
    )
    legend_items.append(
        Line2D([0], [0], color=c, linewidth=1.0, linestyle=":", label=f"{size} ··· Train")
    )

fig.legend(
    handles=legend_items,
    loc="center right",
    bbox_to_anchor=(1.0, 0.5),
    fontsize=9.5,
    framealpha=0.9,
    title="Size / Split",
    title_fontsize=10,
)

fig.suptitle(
    "Fig. 4b  Train vs. Validation Loss (solid = Val, dotted = Train)",
    fontsize=12, y=0.97,
)

# ── 保存 ──────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"fig4b_train_val_curves.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
