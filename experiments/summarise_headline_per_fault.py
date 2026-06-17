#!/usr/bin/env python3
"""
Summarise the headline-operating-point per-fault recall for TEP/SKAB/SWaT
across 4 seeds. Reports aggregate recall, per-fault min/mean/std/max, and
the 3 weakest faults by name.

Reads from:
    results/per_fault/tep/  (already at headline operating point)
    results/per_fault_headline/skab/  (re-run at headline op point)
    results/per_fault_headline/swat/  (re-run at headline op point)

Output:
    results/per_fault_headline/summary.json
    results/per_fault_headline/summary.csv  (one row per dataset)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "TEP":  ROOT / "results" / "per_fault" / "tep",
    "SKAB": ROOT / "results" / "per_fault_headline" / "skab",
    "SWaT": ROOT / "results" / "per_fault_headline" / "swat",
}


def aggregate_dataset(ds_dir: Path) -> dict:
    rows = []
    summaries = []
    for csv in sorted(ds_dir.glob("per_fault_seed*.csv")):
        m = re.search(r"per_fault_seed(\d+)\.csv", csv.name)
        if not m:
            continue
        seed = int(m.group(1))
        df = pd.read_csv(csv)
        df["seed"] = seed
        rows.append(df)
        sj = ds_dir / f"summary_seed{seed}.json"
        if sj.exists():
            with open(sj) as f:
                summaries.append(json.load(f))
    if not rows:
        return {"n_seeds": 0}
    all_df = pd.concat(rows, ignore_index=True)
    # Average per-fault recall across seeds for each fault_id
    by_fault = all_df.groupby("fault_id").agg(
        per_fault_recall_mean=("per_fault_recall", "mean"),
        per_fault_recall_std=("per_fault_recall", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        n_seeds=("per_fault_recall", "count"),
    ).reset_index()
    n_seeds = len(summaries)
    agg_recalls = [s["aggregate_recall"] for s in summaries]
    agg_savings = [s["aggregate_savings_pct"] for s in summaries]
    out = {
        "n_seeds":              n_seeds,
        "n_faults":             int(by_fault.shape[0]),
        "scorer":               summaries[0]["scorer"] if summaries else "?",
        "priority_frac":        summaries[0]["priority_frac"] if summaries else None,
        "budget_frac":          summaries[0]["budget_frac"] if summaries else None,
        "aggregate_recall_mean":  float(np.mean(agg_recalls)),
        "aggregate_recall_std":   float(np.std(agg_recalls, ddof=1)) if len(agg_recalls) > 1 else 0.0,
        "aggregate_savings_mean": float(np.mean(agg_savings)),
        "aggregate_savings_std":  float(np.std(agg_savings, ddof=1)) if len(agg_savings) > 1 else 0.0,
        "per_fault_min_recall":  float(by_fault["per_fault_recall_mean"].min()),
        "per_fault_mean_recall": float(by_fault["per_fault_recall_mean"].mean()),
        "per_fault_max_recall":  float(by_fault["per_fault_recall_mean"].max()),
    }
    weakest = by_fault.nsmallest(3, "per_fault_recall_mean")[["fault_id", "per_fault_recall_mean", "per_fault_recall_std"]]
    out["weakest_3"] = [
        {"fault_id": str(r["fault_id"]),
          "recall_mean": float(r["per_fault_recall_mean"]),
          "recall_std":  float(r["per_fault_recall_std"])}
        for _, r in weakest.iterrows()
    ]
    return out


def main():
    out = {"per_dataset": {}}
    for ds_name, ds_dir in DATASETS.items():
        info = aggregate_dataset(ds_dir)
        out["per_dataset"][ds_name] = info
        print(f"\n[{ds_name}] n_seeds={info.get('n_seeds')}, "
              f"scorer={info.get('scorer')}, "
              f"savings={info.get('aggregate_savings_mean'):.1f}%, "
              f"aggregate_recall={info.get('aggregate_recall_mean'):.3f} +/- "
              f"{info.get('aggregate_recall_std'):.3f}")
        for w in info.get("weakest_3", []):
            print(f"    weakest: fault {w['fault_id']:>8}  "
                  f"recall = {w['recall_mean']:.3f} +/- {w['recall_std']:.3f}")

    out_dir = ROOT / "results" / "per_fault_headline"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    # CSV form
    csv_rows = [{"dataset": ds, **info} for ds, info in out["per_dataset"].items()
                 if info.get("n_seeds")]
    pd.DataFrame(csv_rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved: {out_dir/'summary.json'} and .csv")


if __name__ == "__main__":
    main()
