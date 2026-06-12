"""
Fig.2 NIG spectrum (m = 2..101)
CICM 2026 paper — Figure 2

Plots NIG (Normalized Information Gain) as a line chart by modulus m.
  - Three series: IntSeqBERT / Vanilla / Ablation (Large models)
  - Light gray background shading for prime moduli
  - Annotations for m=96 (highest NIG), m=2 (parity), and m=60 (Babylonian)

Output: paper/2026-03/figures/fig2_nig_spectrum.pdf
      paper/2026-03/figures/fig2_nig_spectrum.png

Data source: checkpoints/large_std/{model}/analysis/mod_spectrum/mod_spectrum_ranking.csv
"""

from pathlib import Path
import pandas as pd
from sympy import isprime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT = REPO_ROOT / "checkpoints" / "large_std"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

MODELS  = ["intseq", "vanilla", "ablation"]
LABELS  = {"intseq": "IntSeqBERT", "vanilla": "Vanilla", "ablation": "Ablation"}
COLORS  = {"intseq": "#1f77b4", "vanilla": "#ff7f0e", "ablation": "#2ca02c"}
STYLES  = {"intseq": "-",        "vanilla": "--",     "ablation": "-."}
WIDTHS  = {"intseq": 2.0,        "vanilla": 1.6,     "ablation": 1.6}

# ── Data loading ─────────────────────────────────────────────────────────
def load_nig(model: str) -> pd.DataFrame:
    path = CKPT / model / "analysis" / "mod_spectrum" / "mod_spectrum_ranking.csv"
    df = pd.read_csv(path).sort_values("modulus").reset_index(drop=True)
    return df

dfs = {m: load_nig(m) for m in MODELS}

# CI data (IntSeq only)
ci_path = CKPT / "intseq" / "analysis" / "mod_spectrum" / "mod_spectrum_with_ci.csv"
ci_df = pd.read_csv(ci_path).sort_values("modulus").reset_index(drop=True)

moduli = dfs["intseq"]["modulus"].values  # 2..101
primes = [m for m in moduli if isprime(m)]

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.2))
fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.14)

# Gray background shading for prime moduli
for p in primes:
    ax.axvspan(p - 0.5, p + 0.5, color="#e0e0e0", linewidth=0, zorder=0)

# NIG lines for the three models
for model in MODELS:
    df = dfs[model]
    ax.plot(df["modulus"], df["nig_score"],
            color=COLORS[model], linestyle=STYLES[model],
            linewidth=WIDTHS[model], label=LABELS[model], zorder=2)

# IntSeqBERT confidence interval band
ax.fill_between(
    ci_df["modulus"], ci_df["nig_lower"], ci_df["nig_upper"],
    color=COLORS["intseq"], alpha=0.12, zorder=1, label="IntSeqBERT 95% CI"
)

# ── Notable modulus annotations ──────────────────────────────────────────
nig96  = dfs["intseq"].loc[dfs["intseq"]["modulus"] == 96,  "nig_score"].values[0]
nig2   = dfs["intseq"].loc[dfs["intseq"]["modulus"] == 2,   "nig_score"].values[0]
nig60  = dfs["intseq"].loc[dfs["intseq"]["modulus"] == 60,  "nig_score"].values[0]

# m=96 (placed downward to stay inside the frame)
ax.annotate(
    f"$m=96$\n({nig96:.3f})",
    xy=(96, nig96), xytext=(88, nig96 + 0.01),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    fontsize=9, color="#333333", ha="center",
)
# m=2 (placed downward to stay inside the frame)
ax.annotate(
    f"$m=2$\n({nig2:.3f})",
    xy=(2, nig2), xytext=(10, nig2 + 0.01),
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

# ── Axes ─────────────────────────────────────────────────────────────────
ax.set_xlim(1.5, 101.5)
ax.set_ylim(0.17, 0.68)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(which="major", linestyle=":", linewidth=0.6, alpha=0.7, zorder=1)
ax.set_xlabel("Modulus $m$", fontsize=11)
ax.set_ylabel("Normalized Information Gain (NIG)", fontsize=11)

# Legend entry for prime shading
prime_patch = mpatches.Patch(color="#e0e0e0", label="Prime $m$ (shaded)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + [prime_patch], loc="lower right", fontsize=9.5,
          framealpha=0.9, ncol=2)

# ── Save ─────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"fig2_nig_spectrum.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
