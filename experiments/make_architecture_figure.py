#!/usr/bin/env python3
"""
Pipeline architecture figure (Figure 0) for the CAP-Dedup paper.

Style is matched to publication_figures_v2.py:
 - serif font, 9 pt body
 - dark muted palette (#1f6091 blue, #256d2d green) for the two guarantee
   stages; neutral greys for data/processing
 - axes.linewidth 0.8
 - tinted fills with darker edges (no pastel templated look)

Layout (left -> right):
    Raw stream --+--> Siamese encoder phi
                 |
                 +--> Anomaly scorer s(x) --> Stage 1 (green, recall guarantee)
                                                   --> Stage 2 (blue, coverage) --> Retained set
                                                   ^
                                                   |
                                       Calibration set C (tau_alpha)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "figures_paper_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})

# Publication palette
COL_GREEN  = "#256d2d"
COL_BLUE   = "#1f6091"
COL_EDGE   = "#3a3a3a"
COL_PRIMARY = "#1a1a1a"
COL_SECOND  = "#444444"

# Subtle tinted fills (paper-style, not pastel)
FILL_GREEN = "#eaf0e7"
FILL_BLUE  = "#e7eef5"
FILL_GREY  = "#f5f5f5"
FILL_CAL   = "#fafafa"


def box(ax, xy, w, h, label, sublabel=None, *, fill=FILL_GREY,
        edge=COL_EDGE, lw=0.8, fontsize=8.5, fontsize_sub=7.0, ls="-"):
    x, y = xy
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.012,rounding_size=0.05",
                          linewidth=lw, edgecolor=edge, facecolor=fill,
                          linestyle=ls, zorder=2)
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w / 2, cy + 0.17, label, ha="center", va="center",
                fontsize=fontsize, color=COL_PRIMARY, zorder=3)
        ax.text(x + w / 2, cy - 0.20, sublabel, ha="center", va="center",
                fontsize=fontsize_sub, style="italic", color=COL_SECOND, zorder=3)
    else:
        ax.text(x + w / 2, cy, label, ha="center", va="center",
                fontsize=fontsize, color=COL_PRIMARY, zorder=3)


def arrow(ax, p, q, *, color=COL_EDGE, lw=0.9, curve=0.0, label=None,
          label_loc=None, label_color=None):
    arr = FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=10,
                          linewidth=lw, color=color,
                          connectionstyle=f"arc3,rad={curve}", zorder=1)
    ax.add_patch(arr)
    if label:
        if label_loc is None:
            label_loc = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.10)
        ax.text(label_loc[0], label_loc[1], label, ha="center", va="center",
                fontsize=7.2,
                color=label_color or COL_SECOND, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))


def main():
    # Canvas limits are kept proportional to the figure size (2.0 data-units
    # per inch on both axes) so set_aspect("equal") does NOT letterbox -- boxes
    # render at the size the layout intends and text stays inside them.
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9.4)
    ax.set_aspect("equal")
    ax.axis("off")

    H = 1.0  # common box height

    # ---- Input (left) ---------------------------------------------------
    box(ax, (0.40, 4.00), 2.70, H, "Raw stream",
        sublabel=r"$x_1, x_2, \ldots, x_N$", fill=FILL_GREY)

    # ---- Two feature branches -------------------------------------------
    # Siamese encoder (top branch -> feeds the optional coverage fill)
    box(ax, (3.80, 6.30), 3.20, H, "Siamese encoder",
        sublabel=r"$\phi(x)$", fill=FILL_GREY)
    # Anomaly scorer (main line -> feeds the conformal gate)
    box(ax, (3.80, 4.00), 3.20, H, "Anomaly scorer",
        sublabel=r"$s(x)$", fill=FILL_GREY)

    # ---- Stage 1: conformal gate (green, the contribution) --------------
    box(ax, (7.90, 4.00), 4.00, H,
        "Stage 1: conformal gate",
        sublabel=r"$M = \{x : s(x) \geq \tau_\alpha\}$",
        fill=FILL_GREEN, edge=COL_GREEN, lw=1.2,
        fontsize=8.5, fontsize_sub=7.5)

    # ---- Stage 2: coverage fill (OPTIONAL, demoted: dashed/neutral) ------
    # The ablation shows it is replaceable by random in-budget filling, so it
    # is drawn deliberately secondary and is not the source of the guarantee.
    box(ax, (12.70, 4.00), 5.00, H,
        "Stage 2: coverage fill (optional)",
        sublabel=r"$S \supseteq M,\;\; |S| = B$",
        fill=FILL_GREY, edge=COL_SECOND, lw=0.9, ls="--",
        fontsize=8.5, fontsize_sub=7.5)

    # ---- Calibration set (below Stage 1, side-input) --------------------
    box(ax, (7.90, 1.40), 4.00, H,
        "Calibration set $C$",
        sublabel=r"$\lfloor \alpha(n_\mathrm{cal}{+}1) \rfloor$-th rank",
        fill=FILL_CAL)

    # ---- Retained set (output, below Stage 2) ---------------------------
    box(ax, (12.70, 1.40), 5.00, H,
        "Retained set $S$ (to storage)",
        fill=FILL_GREY, fontsize=8.5)

    # ---- Arrows ---------------------------------------------------------
    # raw -> anomaly scorer (main line, straight)
    arrow(ax, (3.10, 4.50), (3.80, 4.50))
    # raw -> Siamese encoder (curving up into the top branch)
    arrow(ax, (3.10, 4.80), (4.30, 6.30), curve=0.25)
    # anomaly scorer -> Stage 1
    arrow(ax, (7.00, 4.50), (7.90, 4.50))
    ax.text(7.45, 5.20, "scores", ha="center", va="center",
            fontsize=7.5, color=COL_SECOND, style="italic", zorder=4)
    # Stage 1 -> Stage 2
    arrow(ax, (11.90, 4.50), (12.70, 4.50))
    ax.text(12.30, 5.20, r"seed $M$", ha="center", va="center",
            fontsize=7.5, color=COL_SECOND, style="italic", zorder=4)
    # Siamese -> Stage 2  (long arc over the top; label at the apex, clear of boxes)
    arrow(ax, (7.00, 6.80), (15.20, 5.00), curve=-0.28)
    ax.text(11.20, 7.70, r"$\phi(\cdot)$  (cosine distance)",
            ha="center", va="center", fontsize=7.5,
            color=COL_SECOND, style="italic", zorder=4)
    # Stage 2 -> retained set (downward)
    arrow(ax, (15.20, 4.00), (15.20, 2.40))
    # calibration -> Stage 1 (upward; supplies the threshold)
    arrow(ax, (9.90, 2.40), (9.90, 4.00))
    ax.text(9.45, 3.20, r"$\tau_\alpha$",
            ha="center", va="center", fontsize=9.0,
            color=COL_GREEN, style="italic", zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    # ---- Title and legend strip ----------------------------------------
    ax.text(9.00, 8.90, "CAP-Dedup pipeline",
            ha="center", va="center", fontsize=11.5, weight="bold",
            color=COL_PRIMARY)
    ax.text(9.00, 0.60,
            "Green: conformal gate, the recall guarantee (Thm. 1) and the contribution.    "
            "Dashed: optional coverage fill, replaceable by random in-budget filling (see ablation).\n"
            "Grey: data / processing.",
            ha="center", va="center", fontsize=7.5, linespacing=1.5,
            color=COL_SECOND, style="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig0_architecture.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/fig0_architecture.pdf + .png")


if __name__ == "__main__":
    main()
