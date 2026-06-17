#!/usr/bin/env python3
"""
Compute 95% confidence intervals (t-distribution) for the headline
CAP-Dedup operating points on TEP, SKAB, SWaT. Outputs both the
old-style mean +/- std and an explicit 95% CI half-width.

Output:
    results/headline_with_ci.csv
    results/headline_with_ci.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "TEP":  ROOT / "results" / "pareto"      / "pareto_sweep_tep.csv",
    "SKAB": ROOT / "results" / "pareto_skab" / "pareto_sweep_skab.csv",
    "SWaT": ROOT / "results" / "pareto_swat" / "pareto_sweep_swat.csv",
}
RECALL_FLOORS = [0.99, 0.95, 0.90, 0.85, 0.80]


def best_per_seed(df: pd.DataFrame, recall_floor: float) -> np.ndarray:
    cap = df[df["method"].isin(["cap_dedup", "cap_dedup_budget"])]
    out = []
    for seed in cap["seed"].unique():
        s = cap[(cap["seed"] == seed) & (cap["safety_recall"] >= recall_floor)]
        if len(s) > 0:
            out.append(float(s["storage_savings_pct"].max()))
    return np.asarray(out, dtype=float)


def main():
    rows = []
    for ds_name, p in DATASETS.items():
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for floor in RECALL_FLOORS:
            arr = best_per_seed(df, floor)
            n = len(arr)
            if n == 0:
                rows.append({"dataset": ds_name, "recall_floor": floor,
                              "n_feasible": 0, "infeasible": True})
                continue
            if n == 1:
                rows.append({"dataset": ds_name, "recall_floor": floor,
                              "n_feasible": 1, "mean_savings_pct": float(arr[0]),
                              "std_savings_pct": float("nan"),
                              "ci95_half_width": float("nan"),
                              "ci95_low": float(arr[0]), "ci95_high": float(arr[0])})
                continue
            m, s = float(arr.mean()), float(arr.std(ddof=1))
            sem = s / np.sqrt(n)
            half = float(stats.t.ppf(0.975, df=n - 1) * sem)
            rows.append({
                "dataset": ds_name, "recall_floor": floor,
                "n_feasible": int(n),
                "mean_savings_pct": m,
                "std_savings_pct":  s,
                "ci95_half_width":  half,
                "ci95_low":         m - half,
                "ci95_high":        m + half,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(ROOT / "results" / "headline_with_ci.csv", index=False)
    with open(ROOT / "results" / "headline_with_ci.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("=" * 80)
    print(f"{'95% CIs (t-distribution) for CAP-Dedup headline':^80}")
    print("=" * 80)
    print(f"{'Dataset':<6} {'Floor':<7} {'n':<3} {'Mean':>7} {'Std':>6} "
          f"{'95% CI':<22} {'Half-width':>10}")
    for r in rows:
        if r.get("infeasible"):
            print(f"{r['dataset']:<6} >={int(r['recall_floor']*100):<3}%   "
                  f" 0  infeasible")
            continue
        if r["n_feasible"] == 1:
            print(f"{r['dataset']:<6} >={int(r['recall_floor']*100):<3}%   "
                  f" 1  {r['mean_savings_pct']:>6.2f}% (only one seed feasible)")
            continue
        print(f"{r['dataset']:<6} >={int(r['recall_floor']*100):<3}%  "
              f"{r['n_feasible']:>2}  {r['mean_savings_pct']:>6.2f}% "
              f"{r['std_savings_pct']:>5.2f}% "
              f"[{r['ci95_low']:>5.2f},{r['ci95_high']:>5.2f}]%  "
              f"{r['ci95_half_width']:>5.2f}")
    print(f"\nSaved CSV: {ROOT/'results'/'headline_with_ci.csv'}")
    print(f"Saved JSON: {ROOT/'results'/'headline_with_ci.json'}")


if __name__ == "__main__":
    main()
