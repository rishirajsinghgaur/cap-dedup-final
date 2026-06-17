#!/usr/bin/env python3
"""
TEP sample-size stability ablation.

a robustness request
14.9M-row TEP release is a source of potential bias. This ablation runs
the framework at three sample sizes S in {5_000, 10_000, 20_000} on three
distinct modelling seeds {42, 43, 44}, recording the best feasible savings
at the ≥95% recall floor for each (S, seed). If the headline number is
stable across S, the 10k choice is defended on robustness grounds (not
just on memory/time grounds).

Output:
    results/tep_sample_size_ablation_runs/sweep_S{N}_seed{K}.csv
    results/tep_sample_size_ablation.csv
    results/tep_sample_size_ablation.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SWEEP_SCRIPT = ROOT / "experiments" / "pareto_sweep_tep.py"
MAIN_CSV = ROOT / "results" / "pareto" / "pareto_sweep_tep.csv"
MAIN_BAK = ROOT / "results" / "pareto" / "pareto_sweep_tep.csv.bak_pre_ablation"

SIZES = [5_000, 10_000, 20_000]
SEEDS = [42, 43, 44]
RECALL_FLOOR = 0.95


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "tep_sample_size_ablation_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the main 10-seed TEP CSV so the ablation does not clobber it
    if MAIN_CSV.exists() and not MAIN_BAK.exists():
        shutil.copy2(MAIN_CSV, MAIN_BAK)
        print(f"Preserved main TEP CSV: {MAIN_BAK}")

    summary_rows = []
    for size in SIZES:
        for seed in SEEDS:
            tag = f"S{size}_seed{seed}"
            tagged_csv = runs_dir / f"sweep_{tag}.csv"
            if tagged_csv.exists():
                print(f"[{tag}] already done, skipping")
                df = pd.read_csv(tagged_csv)
            else:
                print(f"\n[{tag}] running TEP sweep with sample-size={size} seed_start={seed}")
                cmd = [
                    sys.executable, str(SWEEP_SCRIPT),
                    "--sample-size", str(size),
                    "--seeds", "1",
                    "--seed-start", str(seed),
                ]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"  WARN: sweep failed for {tag}: {exc}")
                    continue
                if not MAIN_CSV.exists():
                    print(f"  WARN: no CSV produced for {tag}")
                    continue
                shutil.copy2(MAIN_CSV, tagged_csv)
                df = pd.read_csv(tagged_csv)
                print(f"  saved {tagged_csv} ({len(df)} rows)")

            cap = df[df["method"] == "cap_dedup_budget"]
            feas = cap[cap["safety_recall"] >= RECALL_FLOOR]
            if len(feas) == 0:
                summary_rows.append({
                    "sample_size": size, "seed": seed,
                    "best_savings_pct": float("nan"),
                    "feasible_at_95": False,
                    "best_recall": float(cap["safety_recall"].max()) if len(cap) > 0 else float("nan"),
                })
            else:
                summary_rows.append({
                    "sample_size": size, "seed": seed,
                    "best_savings_pct": float(feas["storage_savings_pct"].max()),
                    "feasible_at_95": True,
                    "best_recall": float(feas["safety_recall"].max()),
                })

    # Restore the main 10-seed TEP CSV so downstream scripts find it
    if MAIN_BAK.exists():
        shutil.copy2(MAIN_BAK, MAIN_CSV)
        print(f"Restored main TEP CSV from {MAIN_BAK}")

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out_dir / "tep_sample_size_ablation.csv", index=False)
    summary = {
        "recall_floor": RECALL_FLOOR,
        "rows": summary_rows,
        "per_size_summary": {},
    }
    for size in SIZES:
        sub = sdf[sdf["sample_size"] == size]
        feas = sub[sub["feasible_at_95"]]
        summary["per_size_summary"][str(size)] = {
            "n_seeds_total": int(len(sub)),
            "n_seeds_feasible_at_95": int(len(feas)),
            "best_savings_mean": float(feas["best_savings_pct"].mean()) if len(feas) > 0 else None,
            "best_savings_std":  float(feas["best_savings_pct"].std(ddof=1)) if len(feas) > 1 else None,
        }
    with open(out_dir / "tep_sample_size_ablation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nHeadline: best feasible savings at >=95% recall, by sample size:")
    for size, blk in summary["per_size_summary"].items():
        sm = blk.get("best_savings_mean")
        ss = blk.get("best_savings_std")
        feas = f"{blk['n_seeds_feasible_at_95']}/{blk['n_seeds_total']}"
        if sm is not None and ss is not None:
            print(f"  S={size}: feasible {feas}, savings = {sm:.2f} +/- {ss:.2f}%")
        else:
            print(f"  S={size}: feasible {feas}, savings = {sm}")


if __name__ == "__main__":
    main()
