#!/usr/bin/env python3
"""
Cross-dataset summary table for the CAP-Dedup paper.

Reads the per-dataset sweep CSVs (results/pareto/pareto_sweep_tep.csv and
results/pareto_skab/pareto_sweep_skab.csv) and produces:

  1. A unified per-(dataset, scorer, mode) summary table with the Pareto-knee
     numbers at 95/90/85/80% recall floors.
  2. JSON dump for the paper's results section.

Usage:
  python experiments/cross_dataset_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATASET_CSVS = {
    "TEP":  ROOT / "results" / "pareto"      / "pareto_sweep_tep.csv",
    "SKAB": ROOT / "results" / "pareto_skab" / "pareto_sweep_skab.csv",
    "SWaT": ROOT / "results" / "pareto_swat" / "pareto_sweep_swat.csv",
}


def pareto_front(points):
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


def per_seed_knee(df_subset, targets=(0.80, 0.85, 0.90, 0.95, 0.99)):
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


def fmt_knee(values: List[float]) -> str:
    if not values:
        return "    n/a"
    return f"{np.mean(values):5.2f}+/-{np.std(values):4.2f}"


def best_op_point(df_subset, target_recall=0.95):
    """Return the operating point (config) with max median savings at >= recall floor."""
    if len(df_subset) == 0:
        return None
    cols = ["method", "scorer", "tau_low", "tau_high", "theta",
            "conformal_target_recall", "coreset_budget"]
    cols = [c for c in cols if c in df_subset.columns]
    g = df_subset.groupby(cols, dropna=False).agg(
        mean_recall=("safety_recall", "mean"),
        std_recall=("safety_recall", "std"),
        mean_savings=("storage_savings_pct", "mean"),
        std_savings=("storage_savings_pct", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    # Require full seed support
    full = g[g["n_seeds"] >= g["n_seeds"].max()]
    feasible = full[full["mean_recall"] >= target_recall]
    if len(feasible) == 0:
        return None
    return feasible.sort_values("mean_savings", ascending=False).iloc[0].to_dict()


def main():
    output = {
        "datasets": {},
    }

    print("=" * 96)
    print(f"{'CROSS-DATASET CAP-DEDUP PARETO SUMMARY':^96}")
    print("=" * 96)

    for dset, csv in DATASET_CSVS.items():
        if not csv.exists():
            print(f"\n[{dset}] csv missing: {csv}  -- skipped")
            continue
        df = pd.read_csv(csv)
        # Coerce object cols with possible NaN to string "n/a" before sorting
        for col in ["scorer", "conformal_target_recall", "coreset_budget", "method"]:
            if col in df.columns:
                df[col] = df[col].fillna("n/a").astype(str).replace("nan", "n/a")
        n_seeds = df["seed"].nunique()
        n_rows = len(df)
        scorers = sorted(df["scorer"].unique())
        methods = sorted(df["method"].unique()) if "method" in df.columns else ["conformal_only"]
        print(f"\n[{dset}]  n_seeds={n_seeds}, n_rows={n_rows}, scorers={len(scorers)}, methods={methods}")
        print("-" * 96)
        print(f"{'method':<18} {'scorer':<18} {'sav@80R':>11} {'sav@85R':>11} "
              f"{'sav@90R':>11} {'sav@95R':>11} {'sav@99R':>11}")
        print("-" * 96)

        ds_out = {"n_seeds": int(n_seeds), "scorers": {}, "best_at_recall_floors": {}}

        # By scorer (overall, all methods combined)
        for sname in scorers:
            sub_s = df[df["scorer"] == sname]
            knee = per_seed_knee(sub_s)
            print(f"{'all':<18} {sname:<18} {fmt_knee(knee[0.80]):>11} {fmt_knee(knee[0.85]):>11} "
                  f"{fmt_knee(knee[0.90]):>11} {fmt_knee(knee[0.95]):>11} {fmt_knee(knee[0.99]):>11}")
            ds_out["scorers"][sname] = {
                str(int(t * 100)): {"mean": float(np.mean(v)) if v else None,
                                    "std": float(np.std(v)) if v else None,
                                    "n_feasible_seeds": len(v)}
                for t, v in knee.items()
            }

        # Best operating point per recall floor (across ALL methods/scorers)
        print(f"\n  Best operating point per recall floor (across ALL config):")
        for tgt in [0.95, 0.90, 0.85, 0.80]:
            bop = best_op_point(df, target_recall=tgt)
            if bop is None:
                print(f"    recall >= {int(tgt*100)}%: NO feasible point")
                ds_out["best_at_recall_floors"][str(int(tgt * 100))] = None
            else:
                print(f"    recall >= {int(tgt*100)}%: method={bop.get('method','?'):<18} "
                      f"scorer={bop['scorer']:<18} "
                      f"-> recall={bop['mean_recall']:.3f}+/-{bop['std_recall']:.3f}, "
                      f"savings={bop['mean_savings']:5.2f}+/-{bop['std_savings']:4.2f}%")
                ds_out["best_at_recall_floors"][str(int(tgt * 100))] = {
                    "method": str(bop.get("method", "?")),
                    "scorer": str(bop["scorer"]),
                    "config": {
                        k: (str(bop.get(k)) if bop.get(k) is not None else None)
                        for k in ["tau_low", "tau_high", "theta",
                                  "conformal_target_recall", "coreset_budget"]
                    },
                    "mean_recall": float(bop["mean_recall"]),
                    "std_recall": float(bop["std_recall"]) if not pd.isna(bop["std_recall"]) else None,
                    "mean_savings": float(bop["mean_savings"]),
                    "std_savings": float(bop["std_savings"]) if not pd.isna(bop["std_savings"]) else None,
                }
        output["datasets"][dset] = ds_out

    out_path = ROOT / "results" / "cross_dataset_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON summary saved: {out_path}")


if __name__ == "__main__":
    main()
