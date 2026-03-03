"""
Fig.3 スケーリング折れ線グラフ
CICM 2026 paper — Figure 3

横軸: Small / Middle / Large
縦軸: 3パネル構成 — Mag Acc (%) / MMA (%) / Solver Top-1 (%)
3系列: IntSeqBERT / Vanilla / Ablation

データ: checkpoints/{size}_std/{model}/analysis/ 以下の CSV / JSON
出力: experiment/cicm2026/fig3_scaling.pdf
      experiment/cicm2026/fig3_scaling.png
"""

from pathlib import Path
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── パス設定 ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT      = REPO_ROOT / "checkpoints"
OUT_DIR   = Path(__file__).resolve().parent

SIZES  = ["small", "middle", "large"]
MODELS = ["intseq", "vanilla", "ablation"]

LABELS = {"intseq": "IntSeqBERT", "vanilla": "Vanilla", "ablation": "Ablation"}
COLORS = {"intseq": "#1f77b4", "vanilla": "#ff7f0e", "ablation": "#2ca02c"}
MARKS  = {"intseq": "o",        "vanilla": "s",       "ablation": "^"}
X_LABELS = ["Small\n(6L-256d)", "Middle\n(8L-512d)", "Large\n(12L-768d)"]
X_TICKS  = [0, 1, 2]

# ── データ取得 ────────────────────────────────────────────────────────────
def read_mag_acc(size: str, model: str) -> float:
    """overall_metrics.csv から acc_0.5 を返す（%）"""
    p = CKPT / f"{size}_std" / model / "analysis" / "magnitude" / "overall_metrics.csv"
    with open(p) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return float(row["acc_0.5"])

def read_mma(size: str, model: str) -> float:
    """mod_spectrum_ranking.csv の accuracy 列の平均（%）"""
    p = CKPT / f"{size}_std" / model / "analysis" / "mod_spectrum" / "mod_spectrum_ranking.csv"
    accs = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            accs.append(float(row["accuracy"]))
    return sum(accs) / len(accs)

def read_solver_top1(size: str, model: str) -> float:
    """solver/summary.json から overall top1_acc を返す（%）"""
    p = CKPT / f"{size}_std" / model / "analysis" / "solver" / "summary.json"
    with open(p) as f:
        d = json.load(f)
    return float(d["overall"]["top1_acc"])

# 全データ収集
data = {model: {"mag": [], "mma": [], "solver": []} for model in MODELS}
for model in MODELS:
    for size in SIZES:
        data[model]["mag"].append(read_mag_acc(size, model))
        data[model]["mma"].append(read_mma(size, model))
        data[model]["solver"].append(read_solver_top1(size, model))

# ── プロット ──────────────────────────────────────────────────────────────
METRICS = [
    ("mag",    "Mag Acc (%)",       [84, 97]),
    ("mma",    "MMA (%)",           [20, 57]),
    ("solver", "Solver Top-1 (%)",  [0,  22]),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.18, wspace=0.30)

for ax, (key, ylabel, ylim) in zip(axes, METRICS):
    for model in MODELS:
        y = data[model][key]
        ax.plot(
            X_TICKS, y,
            color=COLORS[model], marker=MARKS[model],
            markersize=8, linewidth=2.0, label=LABELS[model],
            zorder=3,
        )
        # マーカーに白縁
        ax.scatter(X_TICKS, y,
                   color=COLORS[model], s=60,
                   edgecolors="white", linewidths=0.8, zorder=4)

    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_LABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(ylim)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(which="major", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

# 凡例（中央パネル内）
axes[1].legend(loc="lower right", fontsize=9, framealpha=0.9)

# タイトル
axes[0].set_title("Magnitude Accuracy", fontsize=11, fontweight="bold")
axes[1].set_title("Mean Modulo Accuracy (MMA)", fontsize=11, fontweight="bold")
axes[2].set_title("Solver Top-1 Accuracy", fontsize=11, fontweight="bold")

fig.suptitle(
    "Fig. 3  Scaling Behaviour: Small → Middle → Large",
    fontsize=12, y=0.97,
)

# ── 保存 ──────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"fig3_scaling.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
