"""
Fig.2 NIG スペクトル（m = 2..101）
CICM 2026 paper — Figure 2

NIG (Normalized Information Gain) を法 m ごとに折れ線プロット。
  - 3 系列: IntSeqBERT / Vanilla / Ablation（Large モデル）
  - 素数法には薄いグレー背景シェーディング
  - m=96（最高 NIG）・m=2（パリティ）・m=60（バビロニア）をアノテーション

出力: experiment/cicm2026/fig2_nig_spectrum.pdf
      experiment/cicm2026/fig2_nig_spectrum.png

データソース: checkpoints/large_std/{model}/analysis/mod_spectrum/mod_spectrum_ranking.csv
"""

from pathlib import Path
import pandas as pd
from sympy import isprime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

# ── パス設定 ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT = REPO_ROOT / "checkpoints" / "large_std"
OUT_DIR = Path(__file__).resolve().parent

MODELS  = ["intseq", "vanilla", "ablation"]
LABELS  = {"intseq": "IntSeqBERT", "vanilla": "Vanilla", "ablation": "Ablation"}
COLORS  = {"intseq": "#1f77b4", "vanilla": "#ff7f0e", "ablation": "#2ca02c"}
STYLES  = {"intseq": "-",        "vanilla": "--",     "ablation": "-."}
WIDTHS  = {"intseq": 2.0,        "vanilla": 1.6,     "ablation": 1.6}

# ── データ読み込み ────────────────────────────────────────────────────────
def load_nig(model: str) -> pd.DataFrame:
    path = CKPT / model / "analysis" / "mod_spectrum" / "mod_spectrum_ranking.csv"
    df = pd.read_csv(path).sort_values("modulus").reset_index(drop=True)
    return df

dfs = {m: load_nig(m) for m in MODELS}

# CI データ（IntSeq のみ）
ci_path = CKPT / "intseq" / "analysis" / "mod_spectrum" / "mod_spectrum_with_ci.csv"
ci_df = pd.read_csv(ci_path).sort_values("modulus").reset_index(drop=True)

moduli = dfs["intseq"]["modulus"].values  # 2..101
primes = [m for m in moduli if isprime(m)]

# ── プロット ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.2))
fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.14)

# 素数法にグレー背景シェーディング
for p in primes:
    ax.axvspan(p - 0.5, p + 0.5, color="#e0e0e0", linewidth=0, zorder=0)

# 3 モデルの NIG 折れ線
for model in MODELS:
    df = dfs[model]
    ax.plot(df["modulus"], df["nig_score"],
            color=COLORS[model], linestyle=STYLES[model],
            linewidth=WIDTHS[model], label=LABELS[model], zorder=2)

# IntSeqBERT の信頼区間帯
ax.fill_between(
    ci_df["modulus"], ci_df["nig_lower"], ci_df["nig_upper"],
    color=COLORS["intseq"], alpha=0.12, zorder=1, label="IntSeqBERT 95% CI"
)

# ── 注目モジュラスのアノテーション ──────────────────────────────────────
nig96  = dfs["intseq"].loc[dfs["intseq"]["modulus"] == 96,  "nig_score"].values[0]
nig2   = dfs["intseq"].loc[dfs["intseq"]["modulus"] == 2,   "nig_score"].values[0]
nig60  = dfs["intseq"].loc[dfs["intseq"]["modulus"] == 60,  "nig_score"].values[0]

# m=96
ax.annotate(
    f"$m=96$\n({nig96:.3f})",
    xy=(96, nig96), xytext=(88, nig96 + 0.04),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    fontsize=9, color="#333333", ha="center",
)
# m=2
ax.annotate(
    f"$m=2$\n({nig2:.3f})",
    xy=(2, nig2), xytext=(10, nig2 + 0.04),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    fontsize=9, color="#333333", ha="center",
)
# m=60
ax.annotate(
    f"$m=60$\n({nig60:.3f})",
    xy=(60, nig60), xytext=(52, nig60 - 0.05),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    fontsize=9, color="#333333", ha="center",
)

# ── 軸設定 ────────────────────────────────────────────────────────────────
ax.set_xlim(1.5, 101.5)
ax.set_ylim(0.17, 0.68)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(which="major", linestyle=":", linewidth=0.6, alpha=0.7, zorder=1)
ax.set_xlabel("Modulus $m$", fontsize=11)
ax.set_ylabel("Normalized Information Gain (NIG)", fontsize=11)

# 素数シェーディングの凡例エントリ
prime_patch = mpatches.Patch(color="#e0e0e0", label="Prime $m$ (shaded)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + [prime_patch], loc="lower right", fontsize=9.5,
          framealpha=0.9, ncol=2)

fig.suptitle(
    "Fig. 2  NIG Spectrum over Moduli $m = 2, \\ldots, 101$ (Large models)",
    fontsize=12, y=0.97,
)

# ── 保存 ──────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"fig2_nig_spectrum.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
