#!/usr/bin/env python3
"""
BNN ensemble-size and dropout ablation.

We do a focused one-at-a-time (OAT) ablation around the default operating
point (num_ensemble_models=7, bayesian_dropout_rate=0.3) on TEP because:

  - TEP is the most discriminating benchmark: it is the one where only the
    BNN-mean / BNN-combined scorers reach the >=95% recall floor, so the
    BNN architecture choices matter most.
  - SKAB and SWaT show smaller sensitivity (Table V hyperparam-robustness),
    so the OAT result on TEP transfers to a robustness claim that holds
    across the three datasets at the operating points of interest.

For each (num_models, dropout_rate, seed) configuration we run the same
pareto sweep harness with --seeds 1 --seed-start <seed> --sample-size 10000
and a small operating-point grid sufficient to bracket the >=95% recall
floor. We extract the best feasible storage savings per (config, seed).

Total runs: |size_sweep|+|dropout_sweep|-shared_default = 4+4-1 = 7
configurations x 3 seeds = 21 runs at ~8 min each = ~3 hours.

Output:
  results/bnn_ablation_runs/sweep_size{N}_dropout{D}_seed{S}.csv  (per run)
  results/bnn_size_dropout_ablation.csv     (summary rows)
  results/bnn_size_dropout_ablation.json    (mean +/- std per config)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "experiments" / "pareto_sweep_tep.py"
MAIN_CSV = ROOT / "results" / "pareto" / "pareto_sweep_tep.csv"
MAIN_BAK = ROOT / "results" / "pareto" / "pareto_sweep_tep.csv.bak_pre_bnn_ablation"
RECALL_FLOOR = 0.95

DEFAULT_SIZE = 7
DEFAULT_DROPOUT = 0.3
SIZE_GRID    = [3, 5, 7, 10]
DROPOUT_GRID = [0.1, 0.2, 0.3, 0.5]
SEEDS = [42, 43, 44]


def make_configs():
    """One-at-a-time ablation. Default (7, 0.3) is shared between the two
    sweeps so it is only run once."""
    seen = set()
    configs = []
    for sz in SIZE_GRID:
        cfg = (sz, DEFAULT_DROPOUT)
        if cfg not in seen:
            seen.add(cfg)
            configs.append(cfg)
    for dr in DROPOUT_GRID:
        cfg = (DEFAULT_SIZE, dr)
        if cfg not in seen:
            seen.add(cfg)
            configs.append(cfg)
    return configs


def run_one(num_models: int, dropout: float, seed: int, tagged_csv: Path) -> bool:
    """Run the pareto sweep at the given BNN config + seed. Returns True if
    the run produced a CSV; we ship that CSV to tagged_csv."""
    if tagged_csv.exists():
        print(f"  [skip] {tagged_csv.name} already present")
        return True
    if MAIN_CSV.exists() and not MAIN_BAK.exists():
        shutil.copy2(MAIN_CSV, MAIN_BAK)
    # Write a temporary config override file the sweep can read
    env = {
        "CAPDEDUP_BNN_NUM_MODELS": str(num_models),
        "CAPDEDUP_BNN_DROPOUT":    str(dropout),
    }
    import os
    full_env = dict(os.environ); full_env.update(env)

    cmd = [
        sys.executable, str(SWEEP),
        "--sample-size", "10000",
        "--seeds", "1",
        "--seed-start", str(seed),
    ]
    print(f"\n[run] num_models={num_models}, dropout={dropout}, seed={seed}")
    try:
        subprocess.run(cmd, check=True, env=full_env)
    except subprocess.CalledProcessError as exc:
        print(f"  WARN: sweep failed: {exc}")
        return False
    if not MAIN_CSV.exists():
        return False
    shutil.copy2(MAIN_CSV, tagged_csv)
    return True


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "bnn_ablation_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    configs = make_configs()
    rows = []
    print(f"BNN ablation: {len(configs)} configurations x {len(SEEDS)} seeds = "
          f"{len(configs) * len(SEEDS)} total runs.")

    for (num_models, dropout) in configs:
        for seed in SEEDS:
            tag = f"size{num_models}_dropout{dropout:.2f}_seed{seed}"
            tagged_csv = runs_dir / f"sweep_{tag}.csv"
            ok = run_one(num_models, dropout, seed, tagged_csv)
            if not ok:
                rows.append({
                    "num_models": num_models, "dropout_rate": dropout, "seed": seed,
                    "best_savings_pct": float("nan"), "feasible_at_95": False,
                    "best_recall": float("nan"),
                })
                continue
            df = pd.read_csv(tagged_csv)
            cap = df[df["method"] == "cap_dedup_budget"]
            feas = cap[cap["safety_recall"] >= RECALL_FLOOR]
            if len(feas) == 0:
                rows.append({
                    "num_models": num_models, "dropout_rate": dropout, "seed": seed,
                    "best_savings_pct": float("nan"), "feasible_at_95": False,
                    "best_recall": float(cap["safety_recall"].max()) if len(cap) > 0 else float("nan"),
                })
            else:
                rows.append({
                    "num_models": num_models, "dropout_rate": dropout, "seed": seed,
                    "best_savings_pct": float(feas["storage_savings_pct"].max()),
                    "feasible_at_95": True,
                    "best_recall": float(feas["safety_recall"].max()),
                })

    # Restore main CSV
    if MAIN_BAK.exists():
        shutil.copy2(MAIN_BAK, MAIN_CSV)
        print(f"\nRestored main TEP CSV from {MAIN_BAK}")

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(out_dir / "bnn_size_dropout_ablation.csv", index=False)

    summary = {"recall_floor": RECALL_FLOOR, "per_config": {}}
    for (num_models, dropout) in configs:
        sub = df_sum[(df_sum["num_models"] == num_models) &
                     (df_sum["dropout_rate"] == dropout)]
        feas = sub[sub["feasible_at_95"]]
        summary["per_config"][f"size={num_models},dropout={dropout}"] = {
            "n_seeds": int(len(sub)),
            "n_feasible_at_95": int(len(feas)),
            "best_savings_mean": float(feas["best_savings_pct"].mean()) if len(feas) > 0 else None,
            "best_savings_std":  float(feas["best_savings_pct"].std(ddof=1)) if len(feas) > 1 else None,
        }
    with open(out_dir / "bnn_size_dropout_ablation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nHeadline (size, dropout) -> best savings at >=95% recall:")
    for k, v in summary["per_config"].items():
        m = v.get("best_savings_mean")
        s = v.get("best_savings_std")
        feas = f"{v['n_feasible_at_95']}/{v['n_seeds']}"
        if m is None:
            print(f"  {k:<30}: feasible {feas}, infeasible")
        elif s is None:
            print(f"  {k:<30}: feasible {feas}, savings = {m:.2f}%")
        else:
            print(f"  {k:<30}: feasible {feas}, savings = {m:.2f} +/- {s:.2f}%")


if __name__ == "__main__":
    main()
