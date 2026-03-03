"""
Fig.2b NIG vs φ(m)/m 散布図（追加分析）
CICM 2026 paper — supplementary / Section 5.3

NIG と Euler のトーシェント比 φ(m)/m の関係を検証。
  - Pearson / Spearman 相関係数を算出
  - 素数 / 合成数を色分け
  - 注目モジュラスをアノテーション
  - 回帰直線を重ねる

出力: experiment/cicm2026/fig2b_nig_vs_phi.pdf
      experiment/cicm2026/fig2b_nig_vs_phi.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sympy import totient, isprime
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── データ準備 ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT = REPO_ROOT / "checkpoints" / "large_std"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

df = pd.read_csv(CKPT / "intseq" / "analysis" / "mod_spectrum" / "mod_spectrum_ranking.csv")
df = df.sort_values("modulus").reset_index(drop=True)
df["phi_over_n"] = df["modulus"].apply(lambda m: float(totient(m)) / m)
df["is_prime"]   = df["modulus"].apply(isprime)

x = df["phi_over_n"].values.astype(float)
y = df["nig_score"].values.astype(float)

r_p, p_p = stats.pearsonr(x, y)
r_s, p_s = stats.spearmanr(x, y)

# 回帰直線
slope, intercept, *_ = stats.linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = slope * x_line + intercept

# ── プロット ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.0))
fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.13)

# 回帰直線（背景）
ax.plot(x_line, y_line, color="#aaaaaa", linewidth=1.2, linestyle="--",
        zorder=1, label=f"Regression ($r={r_p:.3f}$)")

# 合成数（青系）
comp = df[~df["is_prime"]]
sc_c = ax.scatter(comp["phi_over_n"], comp["nig_score"],
                  c=comp["modulus"], cmap="Blues_r",
                  vmin=2, vmax=105,
                  s=55, zorder=3, edgecolors="none",
                  label="Composite $m$")

# 素数（赤三角）
prim = df[df["is_prime"]]
ax.scatter(prim["phi_over_n"], prim["nig_score"],
           marker="^", color="#e74c3c", s=50, zorder=4,
           edgecolors="none", label="Prime $m$")

# カラーバー（合成数の大きさ）
cbar = fig.colorbar(sc_c, ax=ax, pad=0.02, fraction=0.035)
cbar.set_label("Modulus $m$ (composite)", fontsize=9)

# 注目モジュラスのアノテーション
ANNOTATE = {
    2:  ("m=2\n(parity)",   (-0.12, +0.005)),
    60: ("m=60\n(Babylonian)", (-0.13, -0.018)),
    96: ("m=96",            (+0.015, +0.003)),
    3:  ("m=3",             (+0.010, -0.010)),
    5:  ("m=5",             (+0.010, +0.003)),
}
for m, (label, (dx, dy)) in ANNOTATE.items():
    row = df[df["modulus"] == m].iloc[0]
    ax.annotate(label,
                xy=(row["phi_over_n"], row["nig_score"]),
                xytext=(row["phi_over_n"] + dx, row["nig_score"] + dy),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0),
                fontsize=8.5, color="#222222",
                ha="center")

# 統計情報テキスト
ax.text(0.97, 0.97,
        f"Pearson  $r = {r_p:.3f}$  ($p < 10^{{-28}}$)\n"
        f"Spearman $\\rho = {r_s:.3f}$  ($p < 10^{{-26}}$)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

ax.set_xlabel(r"$\varphi(m)/m$  (Euler totient ratio)", fontsize=11)
ax.set_ylabel("Normalized Information Gain (NIG)", fontsize=11)
ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9)
ax.grid(linestyle=":", linewidth=0.6, alpha=0.6)

fig.suptitle(
    r"Fig. 2b  NIG vs. Euler totient ratio $\varphi(m)/m$ (Large IntSeqBERT)",
    fontsize=11, y=0.97
)

# ── 保存 ──────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"fig2b_nig_vs_phi.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
