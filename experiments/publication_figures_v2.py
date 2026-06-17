#!/usr/bin/env python3
"""
Publication-quality figures for the CAP-Dedup paper (v2, Q1-journal style).

Design principles (per IEEE IoT-J / TII / TIE figure guidelines):
  - Vector output (PDF) primary, PNG for preview.
  - Limited palette (4 colours max) chosen for printer- and colour-blind safety.
  - Distinct marker + line-style per series so figures remain readable in B&W.
  - Serif font (DejaVu Serif as a Times-equivalent fallback) at 9pt.
  - Subplot labels (a), (b), (c) in upper-left.
  - Captions are self-contained and avoid "the red line shows ..." colour-only
    references. Instead reference the legend by method name.
  - Method names are paper-quality, NOT internal code names.

Regenerates all figures from the rigorous 10-seed CSVs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

logger = logging.getLogger("publication_figures_v2")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "TEP":  ROOT / "results" / "pareto"      / "pareto_sweep_tep.csv",
    "SKAB": ROOT / "results" / "pareto_skab" / "pareto_sweep_skab.csv",
    "SWaT": ROOT / "results" / "pareto_swat" / "pareto_sweep_swat.csv",
}
OUT_DIR = ROOT / "results" / "figures_paper_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Q1-style matplotlib configuration
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    # Use mathtext (no LaTeX required) with serif font
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.3",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Limited palette: 4 colours that survive print + colourblind-safe
# (verified against deuteranopia / protanopia simulations).
# Note: the unpublished precursor method (internal code: "conformal_only") is NOT a
# published baseline and is intentionally excluded from all figures in this
# manuscript to avoid implying a comparison against unpublished work.
PALETTE = {
    "cap_dedup_budget":            "#1a1a1a",  # near-black: headline method
    "cap_dedup":                   "#777777",  # mid-grey:  Mode A variant
    "baseline_stratified_score":   "#1f6091",  # dark blue
    "baseline_kcenter":            "#9c2c2c",  # dark red
    "baseline_facility_location":  "#256d2d",  # dark green
    "baseline_random_uniform":     "#7e57c2",  # purple (rare)
    "baseline_reservoir":          "#a86027",  # brown  (rare)
}

# Marker / line-style per method so figures remain interpretable in monochrome print
STYLE = {
    "cap_dedup_budget":           dict(marker="o", linestyle="-",  ms=4.5),
    "cap_dedup":                  dict(marker="s", linestyle="--", ms=4.0),
    "baseline_stratified_score":  dict(marker="D", linestyle="-.", ms=3.5),
    "baseline_kcenter":           dict(marker="v", linestyle="--", ms=3.5),
    "baseline_facility_location": dict(marker=">", linestyle=":",  ms=3.5),
    "baseline_random_uniform":    dict(marker="x", linestyle="--", ms=3.5),
    "baseline_reservoir":         dict(marker="+", linestyle=":",  ms=3.5),
}

# Paper-quality display names
DISPLAY_NAME = {
    "cap_dedup_budget":           "CAP-Dedup (Mode B)",
    "cap_dedup":                  "CAP-Dedup (Mode A)",
    "baseline_stratified_score":  "Top-K by Score",
    "baseline_kcenter":           r"$k$-Center",
    "baseline_facility_location": "Facility-Location",
    "baseline_random_uniform":    "Uniform Random",
    "baseline_reservoir":         "Reservoir",
}

# Standard ordering for legends (most important first). Precursor "conformal_only"
# is filtered out at plot time even if present in the underlying CSV.
METHOD_ORDER = [
    "cap_dedup_budget", "cap_dedup",
    "baseline_stratified_score", "baseline_kcenter",
    "baseline_facility_location", "baseline_random_uniform", "baseline_reservoir",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pareto_front(points):
    """Upper-right Pareto frontier of (x, y) points where we maximise both."""
    pts = sorted(set(map(tuple, points)), key=lambda p: (-p[0], -p[1]))
    front = []
    best = -np.inf
    for x, y in pts:
        if y > best:
            front.append((x, y))
            best = y
    front.sort()
    return front


def load_dataset_csv(csv_path):
    df = pd.read_csv(csv_path)
    for col in ["scorer", "conformal_target_recall", "coreset_budget", "method"]:
        if col in df.columns:
            df[col] = df[col].fillna("n/a").astype(str).replace("nan", "n/a")
    return df


# ---------------------------------------------------------------------------
# Figure 1 - 3-panel Pareto frontier (TEP / SKAB / SWaT)
# ---------------------------------------------------------------------------

def fig1_pareto_per_dataset():
    """3-panel comparison. Double-column width (7 in), 2 in tall per panel."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=False, dpi=150)
    panel_labels = ["(a)", "(b)", "(c)"]

    handles_collected = {}
    for ax, (ds_name, csv), pl in zip(axes, DATASETS.items(), panel_labels):
        if not csv.exists():
            ax.text(0.5, 0.5, f"{ds_name}: no data", ha="center", va="center",
                    transform=ax.transAxes); continue
        df = load_dataset_csv(csv)

        # Plot Pareto frontier per method (aggregated across seeds/configs)
        methods_present = [m for m in METHOD_ORDER if m in df["method"].unique()]
        for m in methods_present:
            sub = df[df["method"] == m]
            g = sub.groupby(
                ["scorer", "conformal_target_recall", "coreset_budget"], dropna=False
            ).agg(rec=("safety_recall", "mean"),
                  sav=("storage_savings_pct", "mean")).reset_index()
            front = pareto_front(list(zip(g["rec"], g["sav"])))
            if not front:
                continue
            xs = [r * 100 for r, _ in front]
            ys = [s for _, s in front]
            style = STYLE[m].copy()
            color = PALETTE[m]
            line = ax.plot(xs, ys, color=color, **style,
                            label=DISPLAY_NAME[m], alpha=0.95)
            handles_collected[m] = line[0]

        # Reference vertical lines at 90, 95, 99% recall
        for t in (90, 95, 99):
            ax.axvline(t, color="0.5", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Fault recall (%)")
        if ax is axes[0]:
            ax.set_ylabel("Storage savings (%)")
        ax.set_title(f"{pl} {ds_name}")
        ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
        ax.set_xlim(50, 102)
        ax.set_ylim(bottom=-2)

    # Single shared legend below all panels
    ordered_handles = [handles_collected[m] for m in METHOD_ORDER if m in handles_collected]
    ordered_labels = [DISPLAY_NAME[m] for m in METHOD_ORDER if m in handles_collected]
    fig.legend(ordered_handles, ordered_labels, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.10), frameon=True,
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig1_pareto_per_dataset.{ext}", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved fig1 (pdf + png)")


# ---------------------------------------------------------------------------
# Figure 2 - Scorer x Dataset table-style heatmap
# ---------------------------------------------------------------------------

def fig2_scorer_heatmap():
    """Scorer x Dataset table showing storage savings at >=95% recall."""
    scorers_internal = ["bnn_mean", "bnn_combined", "bnn_variance",
                         "ecod", "autoencoder", "isolation_forest"]
    scorer_display = {
        "bnn_mean":         "BNN mean",
        "bnn_combined":     "BNN combined",
        "bnn_variance":     "BNN variance",
        "ecod":             "ECOD",
        "autoencoder":      "Autoencoder",
        "isolation_forest": "Isolation Forest",
    }
    dsets = list(DATASETS.keys())
    data = np.full((len(scorers_internal), len(dsets)), np.nan)
    annot = np.full((len(scorers_internal), len(dsets)), "", dtype=object)

    for j, (ds_name, csv) in enumerate(DATASETS.items()):
        if not csv.exists():
            for i in range(len(scorers_internal)):
                annot[i, j] = "-"
            continue
        df = load_dataset_csv(csv)
        for i, sname in enumerate(scorers_internal):
            sub = df[(df["method"] == "cap_dedup_budget") & (df["scorer"] == sname)]
            g = sub.groupby(
                ["conformal_target_recall", "coreset_budget"], dropna=False
            ).agg(rec=("safety_recall", "mean"),
                  sav=("storage_savings_pct", "mean")).reset_index()
            feas = g[g["rec"] >= 0.95]
            if len(feas) == 0:
                data[i, j] = 0
                annot[i, j] = "n/a"
            else:
                best = feas.sort_values("sav", ascending=False).iloc[0]
                data[i, j] = best["sav"]
                annot[i, j] = f"{best['sav']:.1f}"

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    # Use a sequential colormap that is print-friendly (greys)
    im = ax.imshow(data, cmap="Greys", aspect="auto", vmin=0, vmax=80)
    ax.set_xticks(range(len(dsets)))
    ax.set_xticklabels(dsets)
    ax.set_yticks(range(len(scorers_internal)))
    ax.set_yticklabels([scorer_display[s] for s in scorers_internal])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Anomaly scorer")
    # Annotate cells
    for i in range(len(scorers_internal)):
        for j in range(len(dsets)):
            val = data[i, j]
            color = "white" if (not np.isnan(val) and val > 40) else "black"
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="Savings (%)")
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig2_scorer_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)
    logger.info("saved fig2 (pdf + png)")


# ---------------------------------------------------------------------------
# Figure 3 - Best savings vs recall floor (per dataset)
# ---------------------------------------------------------------------------

def fig3_savings_vs_recall_floor():
    """Achievable savings as the recall guarantee floor is varied."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    recall_floors = np.arange(0.70, 1.001, 0.01)

    dataset_styles = {
        "TEP":  dict(color="#1a1a1a", marker="o", linestyle="-",  ms=4),
        "SKAB": dict(color="#1f6091", marker="s", linestyle="--", ms=4),
        "SWaT": dict(color="#9c2c2c", marker="^", linestyle=":",  ms=4),
    }
    for ds_name, csv in DATASETS.items():
        if not csv.exists():
            continue
        df = load_dataset_csv(csv)
        sub = df[df["method"].isin(["cap_dedup_budget", "cap_dedup"])]
        g = sub.groupby(
            ["method", "scorer", "conformal_target_recall", "coreset_budget"], dropna=False
        ).agg(rec=("safety_recall", "mean"),
              sav=("storage_savings_pct", "mean")).reset_index()
        savs = []
        for rf in recall_floors:
            feas = g[g["rec"] >= rf]
            savs.append(feas["sav"].max() if len(feas) > 0 else np.nan)
        ax.plot(recall_floors * 100, savs, label=ds_name,
                **dataset_styles[ds_name], alpha=0.95)

    ax.set_xlabel("Recall floor (%)")
    ax.set_ylabel("Maximum storage savings (%)")
    ax.set_xlim(70, 100)
    ax.set_ylim(0, 90)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", title="Dataset")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig3_savings_vs_recall.{ext}", bbox_inches="tight")
    plt.close(fig)
    logger.info("saved fig3 (pdf + png)")


# ---------------------------------------------------------------------------
# Figure 4 - Variance comparison on TEP at >=85% recall floor
# ---------------------------------------------------------------------------

def fig4_variance_comparison():
    """Per-seed savings distribution at the >=85% recall floor on TEP for the
    two CAP-Dedup modes and the three feasible literature baselines.

    The purpose of the figure is to show that Mode B (budget-first) has
    near-zero seed-to-seed variance because it deterministically allocates
    the storage budget given a calibration draw, whereas stochastic and
    score-only baselines spread out. No unpublished precursor is shown.
    """
    csv = DATASETS["TEP"]
    if not csv.exists():
        return
    df = load_dataset_csv(csv)

    candidate_methods = [
        "cap_dedup_budget", "cap_dedup",
        "baseline_stratified_score", "baseline_kcenter", "baseline_facility_location",
    ]
    palette_for_box = {
        "cap_dedup_budget":           ("#1a1a1a", ""),
        "cap_dedup":                  ("#777777", "//"),
        "baseline_stratified_score":  ("#1f6091", "xx"),
        "baseline_kcenter":           ("#9c2c2c", ".."),
        "baseline_facility_location": ("#256d2d", "++"),
    }
    data, labels, fills, pats = [], [], [], []
    for m in candidate_methods:
        sub = df[df["method"] == m]
        per_seed = []
        for seed in sub["seed"].unique():
            seed_df = sub[(sub["seed"] == seed) & (sub["safety_recall"] >= 0.85)]
            if len(seed_df) > 0:
                per_seed.append(seed_df["storage_savings_pct"].max())
        if per_seed and len(per_seed) >= 3:
            data.append(per_seed)
            labels.append(DISPLAY_NAME[m].replace(" (", "\n("))
            fc, pat = palette_for_box[m]
            fills.append(fc); pats.append(pat)
    if not data:
        return

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    bp = ax.boxplot(data, tick_labels=labels, widths=0.55,
                    patch_artist=True, showfliers=True)
    for patch, fc, pat in zip(bp["boxes"], fills, pats):
        patch.set_facecolor(fc); patch.set_alpha(0.55)
        patch.set_edgecolor("0.1"); patch.set_hatch(pat)
    for median in bp["medians"]:
        median.set(color="#d77f00", linewidth=1.6)
    # Overlay individual seed values as small jittered dots so deterministic
    # methods (which collapse to a single line) read as a cluster of points
    # rather than a bare orange dash.
    rng = np.random.default_rng(7)
    for i, (d, fc) in enumerate(zip(data, fills), start=1):
        x_jitter = i + rng.uniform(-0.10, 0.10, size=len(d))
        ax.scatter(x_jitter, d, s=12, color=fc, edgecolor="0.15",
                   linewidth=0.5, alpha=0.85, zorder=3)
    ax.set_ylabel("Storage savings (%) at recall $\\geq 85\\%$")
    ax.tick_params(axis="x", labelsize=7.0)
    ax.grid(True, axis="y", alpha=0.4)
    for i, d in enumerate(data, start=1):
        if d:
            sd = np.std(d, ddof=1) if len(d) > 1 else 0.0
            ax.text(i, max(d) + 2.0, f"$\\sigma$={sd:.2f}",
                    ha="center", fontsize=7.0, color="0.2")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig4_variance_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)
    logger.info("saved fig4 (precursor removed)")


# ---------------------------------------------------------------------------
# Figure 5 - Latency / scalability (read from JSON)
# ---------------------------------------------------------------------------

def fig5_latency_scalability():
    import json
    p = ROOT / "results" / "benchmarks" / "latency_scalability.json"
    if not p.exists():
        return
    with open(p) as f:
        bench = json.load(f)
    Ns = [e["N"] for e in bench["sizes"]]
    e2e = [e["end_to_end_median_ms_per_sample"] for e in bench["sizes"]]
    tput = [e["throughput_samples_per_sec"] for e in bench["sizes"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    # Panel (a): latency
    ax1.loglog(Ns, e2e, color="#1a1a1a", marker="o", linestyle="-",
                ms=5, linewidth=1.4)
    ax1.set_xlabel("Test set size $N$")
    ax1.set_ylabel("End-to-end latency (ms/sample)")
    ax1.set_title("(a) Per-sample latency")
    ax1.grid(True, which="both", alpha=0.4)

    # Panel (b): throughput
    ax2.loglog(Ns, tput, color="#1f6091", marker="s", linestyle="--",
                ms=5, linewidth=1.4)
    ax2.set_xlabel("Test set size $N$")
    ax2.set_ylabel("Throughput (samples/s)")
    ax2.set_title("(b) Throughput")
    ax2.grid(True, which="both", alpha=0.4)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig5_latency_scalability.{ext}", bbox_inches="tight")
    plt.close(fig)
    logger.info("saved fig5 (pdf + png)")


# ---------------------------------------------------------------------------
# Figure 6 - Per-fault recall on TEP (read from existing CSV)
# ---------------------------------------------------------------------------

def fig6_per_fault_recall():
    """3-panel per-fault recall bar chart for TEP / SKAB / SWaT, averaged
    over the 4-seed headline-operating-point per-fault runs."""
    # Headline operating point per-fault locations:
    #   TEP  -> results/per_fault/tep/per_fault_seed{42-45}.csv (bnn_mean, budget=0.85)
    #   SKAB -> results/per_fault_headline/skab/per_fault_seed{42-45}.csv
    #   SWaT -> results/per_fault_headline/swat/per_fault_seed{42-45}.csv
    per_fault_dirs = {
        "TEP":  ROOT / "results" / "per_fault" / "tep",
        "SKAB": ROOT / "results" / "per_fault_headline" / "skab",
        "SWaT": ROOT / "results" / "per_fault_headline" / "swat",
    }
    seed_files = lambda d: sorted(d.glob("per_fault_seed*.csv"))

    panels = []
    for ds_name, ds_dir in per_fault_dirs.items():
        files = seed_files(ds_dir)
        if not files:
            continue
        # Average per-fault recall across seeds for each fault_id
        rows = []
        for f in files:
            df = pd.read_csv(f)
            rows.append(df)
        all_df = pd.concat(rows, ignore_index=True)
        # Group by fault_id; preserve original ordering from any seed's file
        agg_rows = []
        for fid in all_df["fault_id"].unique():
            sub = all_df[all_df["fault_id"] == fid]
            agg_rows.append({
                "fault_id": fid,
                "per_fault_recall":     sub["per_fault_recall"].mean(),
                "per_fault_recall_std": sub["per_fault_recall"].std(ddof=1) if len(sub) > 1 else 0.0,
                "n_test_anom":          sub["n_test_anom"].mean(),
                "n_kept":               sub["n_kept"].mean(),
            })
        pf = pd.DataFrame(agg_rows).sort_values("per_fault_recall", ascending=False).reset_index(drop=True)
        # Aggregate recall = mean(n_kept) / mean(n_test_anom), averaged over seeds
        agg = float(pf["n_kept"].sum() / pf["n_test_anom"].sum()) if pf["n_test_anom"].sum() else 0.0
        panels.append((ds_name, pf, agg))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.6), sharey=True,
                              gridspec_kw={"wspace": 0.18})
    if len(panels) == 1:
        axes = [axes]
    panel_labels = ["(a)", "(b)", "(c)"]
    for ax, (ds_name, pf, agg), pl in zip(axes, panels, panel_labels):
        xs = np.arange(len(pf))
        # Two-tone with hatching for below-aggregate
        colors = ["0.25" if r >= agg else "0.7" for r in pf["per_fault_recall"]]
        patterns = ["" if r >= agg else "////" for r in pf["per_fault_recall"]]
        yerr = (pf["per_fault_recall_std"] * 100.0).to_numpy() if "per_fault_recall_std" in pf.columns else None
        bars = ax.bar(xs, pf["per_fault_recall"] * 100, color=colors,
                      edgecolor="0.1", linewidth=0.5, width=0.85,
                      yerr=yerr, error_kw={"elinewidth": 0.5, "ecolor": "0.3", "capsize": 1.5} if yerr is not None else None)
        for bar, pat in zip(bars, patterns):
            bar.set_hatch(pat)
        ax.axhline(agg * 100, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(xs)
        if ds_name == "TEP":
            # TEP fault_id is numeric (1..20)
            ax.set_xticklabels([str(int(float(fid))) for fid in pf["fault_id"]],
                               rotation=0, fontsize=7)
            ax.set_xlabel("TEP fault ID")
        elif ds_name == "SKAB":
            # SKAB has many anomaly episodes - show only every Nth label
            n_show = min(8, len(pf))
            ticks_keep = np.linspace(0, len(pf) - 1, n_show).astype(int)
            ax.set_xticks(ticks_keep)
            ax.set_xticklabels([str(int(float(pf["fault_id"].iloc[i]))) for i in ticks_keep],
                               rotation=0, fontsize=7)
            ax.set_xlabel("SKAB episode ID")
        else:  # SWaT
            # 6 attack names can be long (e.g., MV201_P101); use a small
            # rotation so adjacent labels never collide on the printed page.
            short = [str(s).replace("MV201_P101", "MV201/P101") for s in pf["fault_id"]]
            ax.set_xticklabels(short, rotation=22, ha="right", fontsize=7.0)
            ax.set_xlabel("SWaT attack")
        ax.set_title(f"{pl} {ds_name}\n(aggregate = {agg*100:.1f}%)", fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", alpha=0.4, linewidth=0.4)
        if ax is axes[0]:
            ax.set_ylabel("Per-fault recall (%)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig6_per_fault_recall.{ext}", bbox_inches="tight")
    plt.close(fig)
    logger.info("saved fig6 (3-panel pdf + png)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=== Generating Q1-style publication figures (v2) ===")
    fig1_pareto_per_dataset()
    fig2_scorer_heatmap()
    fig3_savings_vs_recall_floor()
    fig4_variance_comparison()
    fig5_latency_scalability()
    fig6_per_fault_recall()
    logger.info(f"All figures saved to: {OUT_DIR}")
    logger.info("Both .pdf (vector) and .png (preview) produced for each.")


if __name__ == "__main__":
    main()
