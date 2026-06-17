#!/usr/bin/env python3
"""
Analyze the Pareto sweep CSV produced by pareto_sweep_tep.py.

Generates:
  - per-scorer Pareto frontier table (recall vs savings @ 90/95/99% recall)
  - winning scorer identification
  - operating points "achievable at >= 95% recall with maximum savings"
  - a comparison plot with one line per scorer

Usage:
  python experiments/analyze_pareto_results.py            # uses default CSV
  python experiments/analyze_pareto_results.py --csv path/to/sweep.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def pareto_front(points):
    """(recall, savings) points -> upper-right frontier sorted by recall asc."""
    pts = sorted(set(map(tuple, points)), key=lambda p: (-p[0], -p[1]))
    front = []
    best = -np.inf
    for r, s in pts:
        if s > best:
            front.append((r, s))
            best = s
    front.sort()
    return front


def savings_at_recall(front, target):
    feasible = [s for r, s in front if r >= target]
    return max(feasible) if feasible else None


def per_seed_knee(df_subset, targets=(0.90, 0.95, 0.99)):
    """Return {target: list_of_per_seed_savings} for the given subset."""
    out = {t: [] for t in targets}
    for seed in df_subset["seed"].unique():
        seed_df = df_subset[df_subset["seed"] == seed]
        pts = list(zip(seed_df["safety_recall"], seed_df["storage_savings_pct"]))
        f = pareto_front(pts)
        for t in targets:
            v = savings_at_recall(f, t)
            if v is not None:
                out[t].append(v)
    return out


def best_operating_point(df_subset, target_recall=0.95):
    """Find the single best (recall, savings) point that meets target_recall,
    aggregated across seeds (median savings at the highest-savings combo that
    achieves target_recall on >= half the seeds).
    """
    if len(df_subset) == 0:
        return None
    n_seeds = df_subset["seed"].nunique()
    # Group by every column that defines a distinct operating point
    candidate_cols = ["method", "tau_low", "tau_high", "theta",
                      "scorer", "conformal_target_recall", "coreset_budget"]
    group_cols = [c for c in candidate_cols if c in df_subset.columns]
    g = df_subset.groupby(group_cols).agg(
        n=("safety_recall", "size"),
        median_recall=("safety_recall", "median"),
        mean_recall=("safety_recall", "mean"),
        std_recall=("safety_recall", "std"),
        median_savings=("storage_savings_pct", "median"),
        mean_savings=("storage_savings_pct", "mean"),
        std_savings=("storage_savings_pct", "std"),
    ).reset_index()
    feasible = g[g["median_recall"] >= target_recall]
    if len(feasible) == 0:
        return None
    return feasible.sort_values("median_savings", ascending=False).iloc[0].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results" / "pareto" / "pareto_sweep_tep.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "pareto" / "analysis_summary.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    # Coerce any NaN in object columns to "n/a" BEFORE printing/sorting
    for col in ["scorer", "conformal_target_recall", "coreset_budget", "method"]:
        if col in df.columns:
            df[col] = df[col].fillna("n/a").astype(str).replace("nan", "n/a")
    print(f"Scorers: {sorted(df['scorer'].unique())}")
    print(f"Conformal targets: {sorted(df['conformal_target_recall'].unique())}")
    print()

    summary = {
        "csv": args.csv,
        "n_seeds": int(df["seed"].nunique()),
        "n_rows": int(len(df)),
        "scorers": {},
        "best_overall": {},
    }

    # ---- Per-method breakdown (conformal_only vs cap_dedup) ----
    if "method" in df.columns:
        print("=" * 95)
        print(f"{'method':<14} {'n_pts':>7} {'sav@90R':>10} {'sav@95R':>10} {'sav@99R':>10}")
        print("=" * 95)
        for m in sorted(df["method"].unique()):
            sub_m = df[df["method"] == m]
            knee_m = per_seed_knee(sub_m)
            def fmt(arr):
                if not arr: return "  n/a"
                return f"{np.mean(arr):5.1f}+/-{np.std(arr):3.1f}"
            print(f"{m:<14} {len(sub_m):>7} {fmt(knee_m[0.90]):>10} "
                  f"{fmt(knee_m[0.95]):>10} {fmt(knee_m[0.99]):>10}")
        print()

    # ---- Per-scorer analysis ----
    print("=" * 95)
    print(f"{'scorer':<22} {'n_pts':>6} {'sav@90R':>10} {'sav@95R':>10} {'sav@99R':>10} "
          f"{'best_op_recall':>16} {'best_op_savings':>16}")
    print("=" * 95)
    for sname in sorted(df["scorer"].unique()):
        sub = df[df["scorer"] == sname]
        knee = per_seed_knee(sub)
        bop95 = best_operating_point(sub, 0.95)
        bop90 = best_operating_point(sub, 0.90)

        def fmt(arr):
            if not arr:
                return "  n/a"
            return f"{np.mean(arr):5.1f}+/-{np.std(arr):3.1f}"

        bop_str = ("n/a", "n/a")
        if bop95 is not None:
            bop_str = (f"{bop95['median_recall']*100:6.2f}%",
                       f"{bop95['median_savings']:6.2f}%")
        elif bop90 is not None:
            bop_str = (f"{bop90['median_recall']*100:6.2f}%",
                       f"{bop90['median_savings']:6.2f}% (@90R)")
        print(f"{sname:<22} {len(sub):>6} {fmt(knee[0.90]):>10} {fmt(knee[0.95]):>10} "
              f"{fmt(knee[0.99]):>10} {bop_str[0]:>16} {bop_str[1]:>16}")

        summary["scorers"][sname] = {
            "n_points": int(len(sub)),
            "knee": {
                f"savings_at_recall_{int(t*100)}": {
                    "values": knee[t],
                    "mean": float(np.mean(knee[t])) if knee[t] else None,
                    "std": float(np.std(knee[t])) if knee[t] else None,
                    "n_feasible_seeds": len(knee[t]),
                }
                for t in (0.90, 0.95, 0.99)
            },
            "best_operating_point_at_95_recall": bop95,
        }
    print()

    # ---- Best overall ----
    print("=" * 95)
    print("BEST OVERALL OPERATING POINT @ >=95% recall (median across seeds)")
    print("=" * 95)
    bop = best_operating_point(df, 0.95)
    if bop is None:
        print("NO operating point achieves median recall >= 95% across seeds.")
        bop_lower = best_operating_point(df, 0.90)
        if bop_lower:
            print(f"\nBest @ >=90%R instead: scorer={bop_lower.get('conformal_target_recall')}, "
                  f"recall={bop_lower['median_recall']*100:.2f}%, "
                  f"savings={bop_lower['median_savings']:.2f}%")
    else:
        print(f"  tau_low={bop['tau_low']}, tau_high={bop['tau_high']}, theta={bop['theta']}")
        print(f"  conformal_target={bop['conformal_target_recall']}")
        print(f"  median fault recall: {bop['median_recall']*100:.2f}%")
        print(f"  median storage savings: {bop['median_savings']:.2f}% +/- {bop['std_savings']:.2f}%")
    summary["best_overall"] = bop

    # ---- Save summary ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {out_path}")

    # ---- Comparison plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {
            "off": "lightgray", "bnn_mean": "#1f77b4", "bnn_variance": "#17becf",
            "bnn_combined": "#9467bd", "isolation_forest": "#d62728",
            "autoencoder": "#2ca02c",
        }
        fig, ax = plt.subplots(figsize=(12, 7))
        for sname in sorted(df["scorer"].unique()):
            sub = df[df["scorer"] == sname]
            color = colors.get(sname, None)
            ax.scatter(sub["safety_recall"] * 100, sub["storage_savings_pct"],
                       s=20, c=color, alpha=0.45, edgecolors="none",
                       label=f"{sname} (n={len(sub)})")
            # Per-scorer aggregate frontier
            pts = list(zip(sub["safety_recall"], sub["storage_savings_pct"]))
            front = pareto_front(pts)
            if front:
                ax.plot([r*100 for r,_ in front], [s for _,s in front],
                        "-", color=color, linewidth=1.8, alpha=0.85)
        for t in (0.90, 0.95, 0.99):
            ax.axvline(t * 100, color="red", linestyle=":", alpha=0.4)
        ax.set_xlabel("Fault Recall (%)", fontsize=12)
        ax.set_ylabel("Storage Savings (%)", fontsize=12)
        ax.set_title(f"CAP-Dedup + Conformal Layer 0: Pareto frontier by scorer "
                     f"({df['seed'].nunique()} seeds, {len(df)} points)",
                     fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_xlim(0, 102)
        plot_path = out_path.parent / "pareto_by_scorer.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Plot saved: {plot_path}")
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
