#!/usr/bin/env python3
"""
Per-fault (and per-attack) recall analysis for CAP-Dedup.

For TEP (which has 20 distinct fault types 1..20), the headline "fault recall"
hides whether all faults are equally well-preserved or whether the framework
systematically misses certain fault types. In safety-critical
domains will want to see this broken out.

This script:
  1. Re-loads test set + per-sample fault_type / attack_id labels.
  2. Re-runs the BEST CAP-Dedup operating point identified in the main sweep.
  3. Computes recall PER FAULT TYPE (TEP) or PER ATTACK EPISODE (SKAB/SWaT).
  4. Saves a CSV + bar plot showing per-fault recall vs aggregate recall.

Output: results/per_fault/{tep,skab,swat}/per_fault_recall.csv + .png

Usage:
  python experiments/per_fault_recall.py --dataset tep --seed 42
  python experiments/per_fault_recall.py --dataset skab --seed 42 --budget 0.70 --priority 0.30
  python experiments/per_fault_recall.py --dataset swat --seed 42 --budget 0.70 --priority 0.30

The defaults reproduce the recommended (30% savings, ~85% recall) operating
point identified in the main sweep.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch
import yaml

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
for p in (ROOT, SRC_DIR, THIS_DIR):
    sys.path.insert(0, str(p))

from core.framework import UncertaintyAwareFramework  # noqa: E402
from anomaly_scorers import build_default_scorers  # noqa: E402
from submodular_coreset import CoverageCoreset  # noqa: E402
from data_splitter import stratified_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("per_fault_recall")


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(name):
    """Return (df_features, labels, raw_df_with_fault_id) for the given dataset."""
    if name == "tep":
        from tep_data_loader import TEPDataLoader
        loader = TEPDataLoader()
        df, labels, _dup_pairs, raw_df = loader.load_data(
            sample_size=10000, random_state=42, label_type="gt_faults"
        )
        # TEP raw_df has 'faultNumber' column (0..20) - 0 means normal
        fault_id_col = "faultNumber"
        episode_col = "simulationRun"
        input_dim = 52
    elif name == "skab":
        from skab_loader import load_skab
        df, labels, raw_df = load_skab(sample_size=None, random_state=42)
        # SKAB: per-anomaly granularity = file_id when label=1
        # We'll group by 'file_id' for anomaly episodes (each file is one mini-episode)
        fault_id_col = "file_id"
        episode_col = "file_id"
        input_dim = 8
    elif name == "swat":
        from swat_loader import load_swat
        df, labels, raw_df = load_swat(sample_size=None, random_state=42)
        # SWaT: attack_id column distinguishes the 6 attacks vs "normal"
        fault_id_col = "attack_id"
        episode_col = "file_id"
        input_dim = df.shape[1]
    else:
        raise ValueError(f"unknown dataset: {name}")
    return df, labels.astype(int), raw_df, fault_id_col, episode_col, input_dim


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["tep", "skab", "swat"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scorer", default="ecod",
                   help="anomaly scorer to use (default ecod - cross-dataset robust)")
    p.add_argument("--priority-frac", type=float, default=0.30,
                   help="Mode B priority fraction (default 0.30)")
    p.add_argument("--budget-frac", type=float, default=0.70,
                   help="Mode B budget fraction (default 0.70 ~= 30% savings)")
    args = p.parse_args()

    set_seeds(args.seed)

    # Load dataset
    df, labels, raw_df, fault_id_col, episode_col, input_dim = load_dataset(args.dataset)
    logger.info(f"[{args.dataset}] loaded {len(df)} rows, {labels.sum()} anomalies "
                f"({labels.mean()*100:.1f}%)")

    # Load config
    with open(ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    cfg = config.copy()
    cfg["model"] = config["model"].copy()
    cfg["model"]["input_dim"] = input_dim

    framework = UncertaintyAwareFramework(cfg)

    # Stratified split
    masks = stratified_split(
        labels=labels, seed=args.seed,
        episode_ids=raw_df[episode_col].to_numpy(),
        mode="stratified",
        ratios=(0.70, 0.10, 0.10, 0.10),
    )

    df_train = df.iloc[masks["train"]].reset_index(drop=True)
    X_train = framework.scaler.fit_transform(df_train.values)
    X_val = framework.scaler.transform(df.iloc[masks["val"]].values)
    X_cal = framework.scaler.transform(df.iloc[masks["cal"]].values)
    X_test = framework.scaler.transform(df.iloc[masks["test"]].values)
    y_train = labels[masks["train"]]
    y_val = labels[masks["val"]]
    y_cal = labels[masks["cal"]]
    y_test = labels[masks["test"]]
    fault_ids_test = raw_df[fault_id_col].iloc[masks["test"]].reset_index(drop=True)

    logger.info(f"train={len(X_train)} val={len(X_val)} cal={len(X_cal)} test={len(X_test)}")

    # Train framework
    framework.train(X_train, y_train, X_val, y_val)

    # Build scorer, then run Mode B selection
    scorers = build_default_scorers()
    scorer = scorers[args.scorer]
    scorer.fit(X_train, y_train, framework=framework, seed=int(args.seed))
    s_test = np.asarray(scorer.score(X_test)).flatten()

    # Build Siamese embeddings + run coreset (Mode B)
    siamese_emb = framework.get_embeddings(X_test, use_siamese=True)
    n_test = len(X_test)
    k_priority = int(round(args.priority_frac * n_test))
    budget = max(k_priority, int(round(args.budget_frac * n_test)))

    priority_mask = np.zeros(n_test, dtype=bool)
    priority_mask[np.argsort(-s_test)[:k_priority]] = True

    cs = CoverageCoreset(seed=int(args.seed))
    keep_mask = cs.select(siamese_emb, s_test, priority_mask, budget)

    # Aggregate metrics
    n_anom = int((y_test == 1).sum())
    aggregate_recall = float((keep_mask & (y_test == 1)).sum() / n_anom) if n_anom else 1.0
    aggregate_savings = (1 - keep_mask.sum() / n_test) * 100
    logger.info(f"AGGREGATE: recall={aggregate_recall:.3f}, savings={aggregate_savings:.1f}%")

    # Per-fault breakdown
    fault_ids_anom = fault_ids_test[y_test == 1].values
    keep_anom = keep_mask[y_test == 1]
    unique_faults = np.unique(fault_ids_anom)
    rows = []
    for fid in unique_faults:
        idx = (fault_ids_anom == fid)
        n_total = int(idx.sum())
        n_kept = int(keep_anom[idx].sum())
        rows.append({
            "fault_id": str(fid),
            "n_test_anom": n_total,
            "n_kept": n_kept,
            "per_fault_recall": n_kept / n_total if n_total else 0.0,
        })
    pf = pd.DataFrame(rows).sort_values("per_fault_recall", ascending=False).reset_index(drop=True)

    out_dir = ROOT / "results" / "per_fault" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"per_fault_seed{args.seed}.csv"
    pf.to_csv(csv_path, index=False)
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "scorer": args.scorer,
        "priority_frac": args.priority_frac,
        "budget_frac": args.budget_frac,
        "aggregate_recall": aggregate_recall,
        "aggregate_savings_pct": aggregate_savings,
        "per_fault_min_recall": float(pf["per_fault_recall"].min()),
        "per_fault_max_recall": float(pf["per_fault_recall"].max()),
        "per_fault_mean_recall": float(pf["per_fault_recall"].mean()),
        "per_fault_std_recall": float(pf["per_fault_recall"].std()),
        "n_faults": len(unique_faults),
    }
    with open(out_dir / f"summary_seed{args.seed}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(10, len(pf) * 0.5), 5))
        bars = ax.bar(range(len(pf)), pf["per_fault_recall"] * 100,
                      color=["#2ca02c" if r >= aggregate_recall else "#d62728"
                             for r in pf["per_fault_recall"]])
        ax.axhline(aggregate_recall * 100, color="black", linestyle="--",
                   label=f"aggregate recall = {aggregate_recall*100:.1f}%")
        ax.set_xticks(range(len(pf)))
        ax.set_xticklabels(pf["fault_id"], rotation=45, ha="right")
        ax.set_xlabel("Fault / Attack ID")
        ax.set_ylabel("Per-fault recall (%)")
        ax.set_title(f"Per-fault recall - {args.dataset.upper()} seed={args.seed} "
                     f"(savings={aggregate_savings:.1f}%)")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / f"per_fault_seed{args.seed}.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"plot failed: {e}")

    print("\n" + "=" * 80)
    print(f"PER-FAULT RECALL SUMMARY ({args.dataset.upper()} seed={args.seed})")
    print("=" * 80)
    print(f"Aggregate: recall={aggregate_recall*100:.2f}%, savings={aggregate_savings:.2f}%")
    print(f"Per-fault: min={summary['per_fault_min_recall']*100:.1f}%, "
          f"max={summary['per_fault_max_recall']*100:.1f}%, "
          f"mean={summary['per_fault_mean_recall']*100:.1f}%, "
          f"std={summary['per_fault_std_recall']*100:.1f}%")
    print()
    print(pf.to_string(index=False))
    print()
    print(f"Saved: {csv_path}, summary JSON, and bar plot.")


if __name__ == "__main__":
    main()
