"""
Fig.6 校正曲線（Uncertainty Calibration）
CICM 2026 paper — Figure 6

X軸: 予測不確かさ σ（Mean over bin）
Y軸: 実際の RMSE（bin 内の実測誤差）
y=x: 完全校正線
上側(赤): 過信（underpredicted σ）、下側(青): 過大推定（overpredicted σ）

3 パネル構成（IntSeqBERT / Vanilla / Ablation、Large モデル）。
Vanilla は σ レンジが極端に広い（0 → 46）ので X 軸をログスケールに統一。

データ: checkpoints/large_std/{model}/analysis/magnitude/calibration_data.csv
       checkpoints/large_std/{model}/analysis/magnitude/overall_metrics.csv

出力: experiment/cicm2026/fig6_calibration.pdf
      experiment/cicm2026/fig6_calibration.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── パス設定 ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT      = REPO_ROOT / "checkpoints" / "large_std"
OUT_DIR   = Path(__file__).resolve().parent

MODELS  = ["intseq", "vanilla", "ablation"]
LABELS  = {"intseq": "IntSeqBERT", "vanilla": "Vanilla", "ablation": "Ablation"}
COLORS  = {"intseq": "#1f77b4", "vanilla": "#ff7f0e", "ablation": "#2ca02c"}

# ── データ読み込み ────────────────────────────────────────────────────────
def load_cal(model: str):
    cal_path  = CKPT / model / "analysis" / "magnitude" / "calibration_data.csv"
    met_path  = CKPT / model / "analysis" / "magnitude" / "overall_metrics.csv"
    cal_df = pd.read_csv(cal_path)
    metrics = pd.read_csv(met_path).iloc[0]
    return cal_df, float(metrics["ece"])

# ── プロット ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.14, wspace=0.32)

for ax, model in zip(axes, MODELS):
    cal_df, ece = load_cal(model)
    x = cal_df["mean_sigma"].values.astype(float)
    y = cal_df["rmse"].values.astype(float)

    # ── 共通軸レンジ: 最小・最大 ─────────────────────────────────────
    x_lo = max(1e-3, x.min() * 0.8)
    x_hi = x.max() * 1.2
    y_lo = 0.0
    y_hi = max(x_hi, y.max() * 1.1)   # y=x 線が見えるよう x_hi まで確保

    # ── 完全校正線 (y=x) ─────────────────────────────────────────────
    xs = np.logspace(np.log10(x_lo), np.log10(x_hi), 300)
    ax.plot(xs, xs, color="#cc0000", linestyle="--", linewidth=1.4,
            label="Perfect ($y=x$)", zorder=3)

    # ── 過信・過大推定の背景 ──────────────────────────────────────────
    ax.fill_between(xs, xs, y_hi, color="#fde8e8", alpha=0.55,
                    zorder=0, label="Overconfident")
    ax.fill_between(xs, 0,  xs,  color="#e8edf8", alpha=0.55,
                    zorder=0, label="Over-uncertain")

    # ── 校正曲線 ─────────────────────────────────────────────────────
    ax.plot(x, y, color=COLORS[model], linewidth=1.8,
            marker="o", markersize=6, zorder=4,
            label=LABELS[model])
    # マーカーだけ白縁
    ax.scatter(x, y, color=COLORS[model], s=36, edgecolors="white",
               linewidths=0.6, zorder=5)

    # ── ECE アノテーション ────────────────────────────────────────────
    ax.text(0.96, 0.96,
            f"ECE = {ece:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
            zorder=6)

    # ── 軸設定 ───────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(LABELS[model], fontsize=12, fontweight="bold")
    ax.set_xlabel(r"Predicted $\sigma$ (log scale)", fontsize=10)
    ax.grid(which="major", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

axes[0].set_ylabel("Actual RMSE", fontsize=10)

# ── 凡例（右パネル内） ────────────────────────────────────────────────────
# 共通凡例: 完全校正線 / 過信 / 過大推定だけ最右パネルに出す
handles = [
    plt.Line2D([0], [0], color="#cc0000", linestyle="--", linewidth=1.4,
               label="Perfect calibration ($y=x$)"),
    plt.Rectangle((0, 0), 1, 1, fc="#fde8e8", alpha=0.7,
                  label="Overconfident region"),
    plt.Rectangle((0, 0), 1, 1, fc="#e8edf8", alpha=0.7,
                  label="Over-uncertain region"),
]
axes[2].legend(handles=handles, loc="lower right", fontsize=8.5,
               framealpha=0.9, handlelength=1.5)

fig.suptitle(
    r"Fig. 6  Uncertainty Calibration: Predicted $\sigma$ vs. Actual RMSE (Large models)",
    fontsize=12, y=0.97,
)

# ── 保存 ──────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"fig6_calibration.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
