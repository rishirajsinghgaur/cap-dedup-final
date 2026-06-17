#!/usr/bin/env python3
"""
Component-wise ablation table for CAP-Dedup, built from existing sweep CSVs.

Produces a clean Q1-style ablation that lists each architectural choice and
shows performance with/without it across all 3 datasets at matched recall
floors. No new experiments needed - all data is in:
    results/pareto/pareto_sweep_tep.csv
    results/pareto_skab/pareto_sweep_skab.csv
    results/pareto_swat/pareto_sweep_swat.csv

Ablations covered:
  - Full CAP-Dedup (Mode B + ECOD/bnn_combined + coreset)
  - Remove conformal Layer 0           = conformal_only (existing CAP-Dedup)
  - Remove submodular coreset          = Mode A (cap_dedup strict)
  - Use random sampling                = baseline_random_uniform / reservoir
  - Use scorer-only (no coreset)       = baseline_stratified_score
  - Use coreset-only (no priority)     = baseline_kcenter / facility_location
  - Swap scorer: 6 options             = scorer breakdown
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("ablation_table")
ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "TEP":  ROOT / "results" / "pareto"      / "pareto_sweep_tep.csv",
    "SKAB": ROOT / "results" / "pareto_skab" / "pareto_sweep_skab.csv",
    "SWaT": ROOT / "results" / "pareto_swat" / "pareto_sweep_swat.csv",
}


def best_at(df, target_recall):
    """For each (method, scorer, ...) config, compute MEAN recall/savings across
    seeds, filter feasible (median_recall >= target_recall), pick max savings.
    Returns (mean_recall, mean_savings, std_savings, n_feasible_seeds) or Nones."""
    if len(df) == 0:
        return (None,) * 4
    cols = ["method", "scorer", "tau_low", "tau_high", "theta",
            "conformal_target_recall", "coreset_budget"]
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(cols, dropna=False).agg(
        mean_recall=("safety_recall", "mean"),
        mean_savings=("storage_savings_pct", "mean"),
        std_savings=("storage_savings_pct", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    g = g[g["n_seeds"] >= g["n_seeds"].max() - 1]  # near-full seed support
    feas = g[g["mean_recall"] >= target_recall]
    if len(feas) == 0:
        return None, None, None, 0
    b = feas.sort_values("mean_savings", ascending=False).iloc[0]
    return (float(b["mean_recall"]), float(b["mean_savings"]),
            float(b["std_savings"]) if not pd.isna(b["std_savings"]) else 0.0,
            int(b["n_seeds"]))


def fmt(triple):
    rec, sav, std, n = triple
    if rec is None:
        return "    n/a"
    return f"{sav:5.1f}+/-{std:4.1f}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Build the master ablation table at recall=0.95 (the safety-critical floor)
    ablations = [
        # (label, filter_func) -- filter_func takes a DataFrame and returns the subset
        ("Full CAP-Dedup (Mode B + coreset + priority + best scorer)",
         lambda d: d[d["method"] == "cap_dedup_budget"]),
        ("- Replace conformal+coreset with CAP-Dedup (Mode B off, conformal_only only)",
         lambda d: d[(d["method"] == "conformal_only") & (d["scorer"] == "off")]),
        ("- Remove coreset stage (Mode A strict, conformal only)",
         lambda d: d[d["method"] == "cap_dedup"]),
        ("- Replace coreset with k-center (no anomaly priority)",
         lambda d: d[d["method"] == "baseline_kcenter"]),
        ("- Replace coreset with facility-location (no anomaly priority)",
         lambda d: d[d["method"] == "baseline_facility_location"]),
        ("- Replace coreset with stratified-by-score (no diversity)",
         lambda d: d[d["method"] == "baseline_stratified_score"]),
        ("- Replace selection with random uniform sampling",
         lambda d: d[d["method"] == "baseline_random_uniform"]),
        ("- Replace selection with reservoir sampling",
         lambda d: d[d["method"] == "baseline_reservoir"]),
    ]

    # Scorer ablation (Mode B only, what scorer wins)
    scorer_ablations = [
        ("bnn_mean", "Mode B + BNN mean prediction (cheap; re-uses BNN ensemble)"),
        ("bnn_variance", "Mode B + BNN epistemic uncertainty"),
        ("bnn_combined", "Mode B + BNN mean + variance combined"),
        ("isolation_forest", "Mode B + Isolation Forest"),
        ("autoencoder", "Mode B + Autoencoder reconstruction error"),
        ("ecod", "Mode B + ECOD (KDD'22 SOTA, paper recommendation)"),
    ]

    out = {"ablations_by_dataset": {}, "scorer_ablations_by_dataset": {}}

    print("=" * 130)
    print(f"{'COMPONENT-WISE ABLATION TABLE  -  10-seed mean +/- std savings % per (dataset, recall floor)':^130}")
    print("=" * 130)
    print(f"{'Configuration':<60} "
          f"{'TEP@90R':>10} {'TEP@95R':>10} | "
          f"{'SKAB@90R':>10} {'SKAB@95R':>10} | "
          f"{'SWaT@90R':>10} {'SWaT@95R':>10}")
    print("-" * 130)
    for label, filt in ablations:
        row = {"label": label}
        cells = []
        for ds_name, csv_path in DATASETS.items():
            if not csv_path.exists():
                cells.extend(["n/a", "n/a"]); row[ds_name] = {}; continue
            df = pd.read_csv(csv_path)
            for col in ["scorer", "conformal_target_recall", "coreset_budget", "method"]:
                if col in df.columns:
                    df[col] = df[col].fillna("n/a").astype(str).replace("nan", "n/a")
            sub = filt(df)
            row[ds_name] = {}
            for tgt in [0.90, 0.95]:
                triple = best_at(sub, tgt)
                cells.append(fmt(triple))
                row[ds_name][f"recall_{int(tgt*100)}"] = {
                    "mean_recall": triple[0], "mean_savings": triple[1],
                    "std_savings": triple[2], "n_seeds": triple[3],
                }
        # short label
        sl = label[:60]
        print(f"{sl:<60} "
              f"{cells[0]:>10} {cells[1]:>10} | "
              f"{cells[2]:>10} {cells[3]:>10} | "
              f"{cells[4]:>10} {cells[5]:>10}")
        out["ablations_by_dataset"][label] = row

    print()
    print("=" * 110)
    print(f"{'SCORER ABLATION  -  Mode B (cap_dedup_budget) at >=95% recall':^110}")
    print("=" * 110)
    print(f"{'Scorer':<70} {'TEP':>12} {'SKAB':>12} {'SWaT':>12}")
    print("-" * 110)
    for scorer_name, label in scorer_ablations:
        row = {"scorer": scorer_name, "label": label}
        cells = []
        for ds_name, csv_path in DATASETS.items():
            if not csv_path.exists():
                cells.append("n/a"); row[ds_name] = None; continue
            df = pd.read_csv(csv_path)
            for col in ["scorer", "conformal_target_recall", "coreset_budget", "method"]:
                if col in df.columns:
                    df[col] = df[col].fillna("n/a").astype(str).replace("nan", "n/a")
            sub = df[(df["method"] == "cap_dedup_budget") & (df["scorer"] == scorer_name)]
            triple = best_at(sub, 0.95)
            cells.append(fmt(triple))
            row[ds_name] = {
                "mean_recall": triple[0], "mean_savings": triple[1],
                "std_savings": triple[2], "n_seeds": triple[3],
            }
        print(f"{label:<70} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}")
        out["scorer_ablations_by_dataset"][scorer_name] = row

    print()
    # ALSO produce a multi-recall view for the headline operating points
    print("=" * 110)
    print(f"{'HEADLINE OPERATING POINTS  -  Full CAP-Dedup at multiple recall floors':^110}")
    print("=" * 110)
    print(f"{'Recall floor':<15} {'TEP':>12} {'SKAB':>12} {'SWaT':>12}")
    print("-" * 60)
    out["multi_recall"] = {}
    for tgt in [0.99, 0.95, 0.90, 0.85, 0.80]:
        cells = []
        rec_entry = {}
        for ds_name, csv_path in DATASETS.items():
            df = pd.read_csv(csv_path)
            for col in ["scorer", "conformal_target_recall", "coreset_budget", "method"]:
                if col in df.columns:
                    df[col] = df[col].fillna("n/a").astype(str).replace("nan", "n/a")
            # Best across ALL CAP-Dedup configs (Mode A + Mode B)
            sub = df[df["method"].isin(["cap_dedup", "cap_dedup_budget"])]
            triple = best_at(sub, tgt)
            cells.append(fmt(triple))
            rec_entry[ds_name] = {"mean_recall": triple[0], "mean_savings": triple[1],
                                   "std_savings": triple[2], "n_seeds": triple[3]}
        out["multi_recall"][f"recall_{int(tgt*100)}"] = rec_entry
        print(f"{f'>={int(tgt*100)}%':<15} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}")

    out_path = ROOT / "results" / "ablation_table.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON saved: {out_path}")


if __name__ == "__main__":
    main()
