"""
Fig.1 IntSeqBERT アーキテクチャ図
CICM 2026 paper — Figure 1

matplotlib の Patches / FancyArrowPatch で描画。
左から右へのデータフローを示す：
  [Input] → [Dual Streams] → [FiLM Fusion] → [Transformer Encoder] → [Output Heads]

出力: experiment/cicm2026/fig1_architecture.pdf
      experiment/cicm2026/fig1_architecture.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT_DIR = Path(__file__).resolve().parent

# ── 色パレット ──────────────────────────────────────────────────────────────
C_MAG   = "#3b82f6"   # 青 — Magnitude stream
C_MOD   = "#f97316"   # オレンジ — Modulo stream
C_FILM  = "#8b5cf6"   # 紫 — FiLM fusion
C_ENC   = "#10b981"   # 緑 — Transformer Encoder
C_HEAD  = "#6b7280"   # グレー — Output Heads
C_INPUT = "#374151"   # ダークグレー — Input
C_BG_MAG  = "#eff6ff"
C_BG_MOD  = "#fff7ed"
C_BG_ENC  = "#ecfdf5"
C_BG_HEAD = "#f9fafb"

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")
fig.patch.set_facecolor("white")

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.15, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad=0,rounding_size={radius}",
                        fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=3)
    ax.add_patch(p)
    return p

def label(ax, x, y, text, size=9, color="black", bold=False, va="center", ha="center", zorder=4):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, zorder=zorder)

def arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.5, style="->", zorder=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=zorder)

def curved_arrow(ax, x1, y1, x2, y2, color, lw=1.5, rad=0.3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)

# ─────────────────────────────────────────────────────────────────────────────
# Section backgrounds
# ─────────────────────────────────────────────────────────────────────────────
# Stream section
box(ax, 1.8, 1.2, 5.0, 4.6, C_BG_MAG, "#93c5fd", lw=1.0, radius=0.3, alpha=0.5)
label(ax, 4.3, 5.65, "Dual-Stream Embedding", size=9.5, bold=True, color="#1d4ed8")

# Encoder section
box(ax, 8.9, 1.2, 3.2, 4.6, C_BG_ENC, "#6ee7b7", lw=1.0, radius=0.3, alpha=0.5)
label(ax, 10.5, 5.65, "Transformer Encoder", size=9.5, bold=True, color="#065f46")

# Output section
box(ax, 12.5, 1.2, 3.1, 4.6, C_BG_HEAD, "#d1d5db", lw=1.0, radius=0.3, alpha=0.5)
label(ax, 14.05, 5.62, "Output Heads", size=9.5, bold=True, color="#374151")

# ─────────────────────────────────────────────────────────────────────────────
# INPUT: sequence tokens
# ─────────────────────────────────────────────────────────────────────────────
# Input box
box(ax, 0.15, 3.0, 1.5, 1.0, "#f3f4f6", C_INPUT, lw=1.8, radius=0.15)
label(ax, 0.90, 3.7, "Input", size=9, bold=True, color=C_INPUT)
label(ax, 0.90, 3.3, r"$x_1, \ldots, x_L$", size=9, color=C_INPUT)

# Arrow: input → feature extraction
arrow(ax, 1.65, 3.5, 1.95, 3.5, color=C_INPUT, lw=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction boxes
# ─────────────────────────────────────────────────────────────────────────────
# Magnitude feature box
box(ax, 1.95, 4.35, 1.9, 0.8, "#dbeafe", C_MAG, lw=1.5, radius=0.12)
label(ax, 2.90, 4.88, "Mag. Features", size=8.5, bold=False, color=C_MAG)
label(ax, 2.90, 4.55, r"$f_\mathrm{mag} \in \mathbb{R}^4$", size=8.5, color=C_MAG)

# Modulo feature box
box(ax, 1.95, 2.85, 1.9, 0.8, "#ffedd5", C_MOD, lw=1.5, radius=0.12)
label(ax, 2.90, 3.38, "Mod. Features", size=8.5, bold=False, color=C_MOD)
label(ax, 2.90, 3.05, r"$f_\mathrm{mod} \in \mathbb{R}^{200}$", size=8.5, color=C_MOD)

# Arrows: input → features
ax.annotate("", xy=(1.95, 4.75), xytext=(1.65, 3.75),
            arrowprops=dict(arrowstyle="->", color=C_MAG, lw=1.4,
                            connectionstyle="arc3,rad=-0.2"), zorder=2)
ax.annotate("", xy=(1.95, 3.25), xytext=(1.65, 3.25),
            arrowprops=dict(arrowstyle="->", color=C_MOD, lw=1.4,
                            connectionstyle="arc3,rad=0.0"), zorder=2)

# ─────────────────────────────────────────────────────────────────────────────
# Projection layers
# ─────────────────────────────────────────────────────────────────────────────
# MLP_mag
box(ax, 4.15, 4.35, 1.5, 0.8, "#bfdbfe", C_MAG, lw=1.5, radius=0.12)
label(ax, 4.90, 4.80, r"MLP$_\mathrm{mag}$", size=9, bold=True, color=C_MAG)
label(ax, 4.90, 4.52, r"$h_\mathrm{mag} \in \mathbb{R}^d$", size=8.5, color=C_MAG)

# W_mod
box(ax, 4.15, 2.85, 1.5, 0.8, "#fed7aa", C_MOD, lw=1.5, radius=0.12)
label(ax, 4.90, 3.30, r"$W_\mathrm{mod}$", size=9, bold=True, color=C_MOD)
label(ax, 4.90, 3.02, r"$h_\mathrm{mod} \in \mathbb{R}^d$", size=8.5, color=C_MOD)

# Arrows: features → projections
arrow(ax, 3.85, 4.75, 4.15, 4.75, color=C_MAG, lw=1.4)
arrow(ax, 3.85, 3.25, 4.15, 3.25, color=C_MOD, lw=1.4)

# ─────────────────────────────────────────────────────────────────────────────
# FiLM Fusion
# ─────────────────────────────────────────────────────────────────────────────
box(ax, 6.05, 3.55, 2.55, 1.9, "#ede9fe", C_FILM, lw=1.8, radius=0.15)
label(ax, 7.32, 5.22, "FiLM Fusion", size=9.5, bold=True, color=C_FILM)
label(ax, 7.32, 4.85, r"$\gamma, \beta = W_\gamma h_\mathrm{mod},\, W_\beta h_\mathrm{mod}$",
      size=8.0, color="#5b21b6")
label(ax, 7.32, 4.52, r"$e_i = (1+\gamma)\odot h_\mathrm{mag} + \beta$",
      size=8.5, bold=True, color=C_FILM)
label(ax, 7.32, 4.17, r"$e_i \in \mathbb{R}^d$", size=8.5, color=C_FILM)
label(ax, 7.32, 3.83, r"$+ \;\mathrm{PE}$", size=8.5, color="#7c3aed")

# Arrows: projections → FiLM
arrow(ax, 5.65, 4.75, 6.05, 4.75, color=C_MAG, lw=1.4)  # h_mag
# h_mod: curved down then into FiLM
ax.annotate("", xy=(6.05, 4.05), xytext=(5.65, 3.25),
            arrowprops=dict(arrowstyle="->", color=C_MOD, lw=1.4,
                            connectionstyle="arc3,rad=-0.3"), zorder=2)

# small label on arrows
ax.text(5.84, 4.84, r"$h_\mathrm{mag}$", fontsize=7.5, color=C_MAG,
        ha="center", va="bottom", zorder=5)
ax.text(5.77, 3.55, r"$h_\mathrm{mod}$", fontsize=7.5, color=C_MOD,
        ha="center", va="bottom", zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# Transformer Encoder
# ─────────────────────────────────────────────────────────────────────────────
# Arrow: FiLM → Encoder
arrow(ax, 8.60, 4.50, 8.90, 4.00, color=C_FILM, lw=1.6)

# Encoder stack (3 layer boxes to suggest depth)
for i, (yoff, alpha_v) in enumerate([(0.0, 1.0), (0.22, 0.85), (0.44, 0.70)]):
    box(ax, 9.05 + i*0.12, 3.25 - yoff, 2.6, 1.5,
        fc="#d1fae5", ec=C_ENC, lw=1.2, radius=0.12, alpha=alpha_v)

# foreground layer labels
label(ax, 10.35, 4.35, r"Pre-LN", size=8.5, bold=True, color=C_ENC)
label(ax, 10.35, 4.02, r"Multi-Head Attn", size=8.5, color=C_ENC)
label(ax, 10.35, 3.72, r"FFN", size=8.5, color=C_ENC)
label(ax, 10.35, 3.40, r"$\times N$ layers", size=8.5, bold=True, color="#065f46")

# self-loop arrow (self-attention indication)
ax.annotate("", xy=(9.05, 4.35), xytext=(9.05, 4.80),
            arrowprops=dict(arrowstyle="->", color=C_ENC, lw=1.2,
                            connectionstyle="arc3,rad=0.5"), zorder=2)

# ─────────────────────────────────────────────────────────────────────────────
# Arrow: Encoder → Heads
# ─────────────────────────────────────────────────────────────────────────────
arrow(ax, 11.78, 4.00, 12.50, 4.00, color=C_ENC, lw=1.6)

# ─────────────────────────────────────────────────────────────────────────────
# Output Heads
# ─────────────────────────────────────────────────────────────────────────────
HEAD_X = 12.65
HEAD_W = 2.75
HEAD_SPECS = [
    (4.40, "#fef3c7", "#d97706", "Magnitude\nRegression", r"$\hat{\mu}, \hat{v} \in \mathbb{R}$"),
    (3.05, "#e0f2fe", "#0284c7", "Sign\nClassification", r"$\hat{s} \in \{+, -, 0\}$"),
    (1.70, "#fce7f3", "#be185d", r"Modulo $\times 100$" + "\nClassification",
     r"$\hat{r}_m \in \{0,\ldots,m{-}1\}$"),
]

for (y0, fc, ec, title, subtitle) in HEAD_SPECS:
    box(ax, HEAD_X, y0, HEAD_W, 1.1, fc=fc, ec=ec, lw=1.5, radius=0.12)
    label(ax, HEAD_X + HEAD_W/2, y0 + 0.78, title, size=8.5, bold=True, color=ec)
    label(ax, HEAD_X + HEAD_W/2, y0 + 0.32, subtitle, size=8.2, color="#374151")

# fan-out arrows from encoder output
for y_head in [4.95, 3.60, 2.25]:
    ax.annotate("", xy=(HEAD_X, y_head), xytext=(12.50, 4.00),
                arrowprops=dict(arrowstyle="->", color=C_HEAD, lw=1.2,
                                connectionstyle="arc3,rad=0.0"), zorder=2)

# ─────────────────────────────────────────────────────────────────────────────
# Title & Legend
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Fig. 1  IntSeqBERT Architecture",
    fontsize=13, y=0.97, fontweight="bold"
)

# Legend patches
legend_items = [
    mpatches.Patch(fc="#dbeafe", ec=C_MAG, lw=1.2, label="Magnitude stream"),
    mpatches.Patch(fc="#ffedd5", ec=C_MOD, lw=1.2, label="Modulo stream"),
    mpatches.Patch(fc="#ede9fe", ec=C_FILM, lw=1.2, label="FiLM fusion"),
    mpatches.Patch(fc="#d1fae5", ec=C_ENC, lw=1.2, label="Transformer Encoder"),
    mpatches.Patch(fc="#f9fafb", ec=C_HEAD, lw=1.2, label="Output heads"),
]
ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.01, 0.01),
          fontsize=8.5, framealpha=0.9, ncol=5,
          handlelength=1.2, handleheight=1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out_path = OUT_DIR / f"fig1_architecture.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)
