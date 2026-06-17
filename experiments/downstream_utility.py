#!/usr/bin/env python3
"""
Downstream fault-detection utility of anomaly-aware deduplication.

Question this answers
---------------------
The headline experiments measure *recall of preserved anomalies* (a property of
the kept set). They do not yet show that preserving those anomalies *matters*
for a downstream task. This script closes that gap:

    Treat the training partition as the storage pool that deduplication compresses.
    At a fixed budget B, build the kept subset two ways:
        (a) CAP-Dedup     : conformal must-preserve  +  k-center coverage coreset
        (b) anomaly-blind  : k-center coverage / random / reservoir (no preservation)
    Train an INDEPENDENT fault detector (RandomForest -- deliberately a different
    model family than the BNN/ECOD scorers that drive selection, so there is no
    circularity) on each kept subset, then evaluate it on the held-out TEST
    partition (never deduplicated).

Hypothesis
----------
At aggressive budgets, anomaly-blind dedup discards rare fault precursors from
storage, so a detector trained on the kept set generalises poorly (low fault
recall / AUPRC on the held-out test set). CAP-Dedup protects those precursors,
so its downstream detector should hold up as compression increases. The
expected signature is a GAP that WIDENS with storage savings.

Ethics
------
No result is engineered. Every kept subset, training run, and evaluation is
recorded as-is. If the gap does not appear (e.g. the pool is anomaly-rich enough
that blind dedup still retains faults at moderate budgets), that is reported.

Outputs
-------
    results/downstream/downstream_utility_<dataset>.csv   (one row per seed x budget x method)
    results/downstream/downstream_utility_<dataset>.json  (per-budget aggregates)
"""

import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from anomaly_scorers import build_default_scorers  # noqa: E402
from conformal_layer0 import ConformalAnomalyGate  # noqa: E402
from submodular_coreset import CoverageCoreset  # noqa: E402
from literature_baselines import (  # noqa: E402
    kcenter_no_conformal,
    random_uniform,
    reservoir_sample,
    stratified_by_score,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("downstream_utility")


# ----------------------------------------------------------------------------
# Downstream detector: an INDEPENDENT model family (RandomForest). Kept
# deliberately distinct from the BNN / ECOD scorers used for SELECTION so the
# experiment cannot be accused of circularity.
# ----------------------------------------------------------------------------

def evaluate_downstream(X_keep, y_keep, X_eval, y_eval, seed):
    """Train a RandomForest on the kept subset, evaluate on the held-out set.

    Returns a metrics dict. Handles the degenerate case where the kept subset
    contains a single class (e.g. anomaly-blind dedup dropped every fault):
    a one-class training set cannot detect the held-out faults, so fault
    recall is 0 and AUPRC collapses to the held-out positive rate -- recorded,
    not patched away.
    """
    pos_rate = float((y_eval == 1).mean())
    n_classes = len(np.unique(y_keep))
    if n_classes < 2:
        # Detector can only ever predict the single class it saw.
        only = int(np.unique(y_keep)[0])
        pred = np.full(len(y_eval), only, dtype=int)
        proba = np.full(len(y_eval), float(only))  # all-0 or all-1 confidence
        return dict(
            n_keep=int(len(y_keep)),
            n_faults_keep=int((y_keep == 1).sum()),
            single_class=True,
            recall=float(recall_score(y_eval, pred, zero_division=0)),
            precision=float(precision_score(y_eval, pred, zero_division=0)),
            f1=float(f1_score(y_eval, pred, zero_division=0)),
            auprc=float(pos_rate if only == 0 else max(pos_rate, 0.0)),
            roc_auc=0.5,
        )

    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=seed, n_jobs=2,
    )
    clf.fit(X_keep, y_keep)
    proba = clf.predict_proba(X_eval)[:, list(clf.classes_).index(1)]
    pred = (proba >= 0.5).astype(int)
    return dict(
        n_keep=int(len(y_keep)),
        n_faults_keep=int((y_keep == 1).sum()),
        single_class=False,
        recall=float(recall_score(y_eval, pred, zero_division=0)),
        precision=float(precision_score(y_eval, pred, zero_division=0)),
        f1=float(f1_score(y_eval, pred, zero_division=0)),
        auprc=float(average_precision_score(y_eval, proba)),
        roc_auc=float(roc_auc_score(y_eval, proba)) if len(np.unique(y_eval)) > 1 else 0.5,
    )


# ----------------------------------------------------------------------------
# Per-seed: train framework, build CAP + baseline keep-masks on the storage
# pool (train partition), evaluate each downstream on the held-out test set.
# ----------------------------------------------------------------------------

def run_seed(dataset, seed, sample_size, budget_grid, target_recall, config):
    if dataset == "tep":
        from pareto_sweep_tep import train_for_seed
        art = train_for_seed(seed, sample_size, config, label_type="gt_faults",
                             split_mode="stratified")
    elif dataset == "swat":
        from pareto_sweep_swat import train_for_seed
        art = train_for_seed(seed, sample_size, config, label_type="gt_attack",
                             split_mode="stratified")
    else:
        raise ValueError(f"dataset {dataset} not wired yet")

    framework = art["framework"]
    X_train, y_train = art["X_train"], art["y_train"]
    X_cal, y_cal = art["X_cal"], art["y_cal"]
    X_test, y_test = art["X_test"], art["y_test"]
    n_pool = len(X_train)

    logger.info(f"[seed={seed}] pool(train)={n_pool} faults={int((y_train==1).sum())} | "
                f"cal={len(X_cal)} | eval(test)={len(X_test)} faults={int((y_test==1).sum())}")

    # Siamese embeddings of the storage pool (drives coverage selection).
    emb_pool = np.asarray(framework.get_embeddings(X_train, use_siamese=True))

    # Anomaly scores on pool + cal. bnn_mean = paper default scorer for
    # conformal preservation; ecod feeds the score-based baselines.
    scorers = build_default_scorers()
    s_pool, s_cal, ecod_pool = None, None, None
    for name, scorer in scorers.items():
        try:
            scorer.fit(X_train, y_train, framework=framework, seed=int(seed))
            if name == "bnn_mean":
                s_pool = np.asarray(scorer.score(X_train)).flatten().astype(float)
                s_cal = np.asarray(scorer.score(X_cal)).flatten().astype(float)
            if name == "ecod":
                ecod_pool = np.asarray(scorer.score(X_train)).flatten().astype(float)
        except Exception as e:
            logger.warning(f"[seed={seed}] scorer {name} failed: {e}")
    if s_pool is None:
        raise RuntimeError("bnn_mean scorer unavailable")
    if ecod_pool is None:
        ecod_pool = s_pool

    # Conformal must-preserve set on the storage pool (Stage 1). The gate's
    # threshold is calibrated on the held-out cal partition for target_recall;
    # here it is used as the Stage-1 SELECTOR over the pool (the finite-sample
    # guarantee itself is the separate headline result on the test partition).
    def _cal_scorer(_X, _s=s_cal):
        return _s

    def _pool_scorer(_X, _s=s_pool):
        return _s

    gate = ConformalAnomalyGate(target_recall=target_recall).fit(X_cal, y_cal, _cal_scorer)
    must_preserve = gate.preserve_mask(X_train, _pool_scorer)
    logger.info(f"[seed={seed}] conformal must-preserve on pool: "
                f"{int(must_preserve.sum())}/{n_pool} "
                f"(captures {int((must_preserve & (y_train==1)).sum())}/{int((y_train==1).sum())} pool faults)")

    rows = []
    for bf in budget_grid:
        budget = int(round(bf * n_pool))
        savings = (1.0 - budget / n_pool) * 100.0

        # ---- method keep-masks on the storage pool ----
        masks = {}
        # CAP-Dedup: conformal preserve seed + k-center coverage coreset.
        cs = CoverageCoreset(seed=int(seed))
        masks["cap_dedup"] = cs.select(emb_pool, s_pool, must_preserve, budget)
        # Anomaly-blind coverage (k-center, no preservation).
        masks["kcenter_blind"] = kcenter_no_conformal(emb_pool, budget, int(seed))
        # Naive sampling.
        masks["random"] = random_uniform(n_pool, budget, int(seed))
        masks["reservoir"] = reservoir_sample(n_pool, budget, int(seed))
        # Anomaly-aware sampling without coverage (top-K by score).
        masks["topk_score"] = stratified_by_score(ecod_pool, budget)

        for method, keep in masks.items():
            keep = np.asarray(keep, dtype=bool)
            m = evaluate_downstream(X_train[keep], y_train[keep], X_test, y_test, int(seed))
            m.update(dict(
                dataset=dataset, seed=int(seed), method=method,
                budget_frac=float(bf), storage_savings_pct=float(savings),
                n_pool=int(n_pool),
            ))
            rows.append(m)
            logger.info(f"[seed={seed}] bf={bf:.2f} sav={savings:4.1f}% {method:<14} "
                        f"keep={m['n_keep']} faults_kept={m['n_faults_keep']} "
                        f"-> recall={m['recall']:.3f} f1={m['f1']:.3f} auprc={m['auprc']:.3f}")

    del framework, art
    gc.collect()
    return rows


def aggregate(df):
    """Mean +/- std across seeds, per (method, budget)."""
    agg = {}
    for (method, bf), g in df.groupby(["method", "budget_frac"]):
        agg.setdefault(method, {})[f"{bf:.2f}"] = {
            "storage_savings_pct": float(g["storage_savings_pct"].mean()),
            "n_seeds": int(g["seed"].nunique()),
            "recall_mean": float(g["recall"].mean()), "recall_std": float(g["recall"].std(ddof=0)),
            "f1_mean": float(g["f1"].mean()), "f1_std": float(g["f1"].std(ddof=0)),
            "auprc_mean": float(g["auprc"].mean()), "auprc_std": float(g["auprc"].std(ddof=0)),
            "roc_auc_mean": float(g["roc_auc"].mean()),
            "faults_kept_mean": float(g["n_faults_keep"].mean()),
        }
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="tep", choices=["tep", "swat"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=10000)
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--budgets", type=str, default="0.10,0.25,0.50,0.75,0.90")
    p.add_argument("--quick", action="store_true", help="1 seed, 2 budgets (smoke test)")
    args = p.parse_args()

    import yaml
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.quick:
        seeds = [args.seed_start]
        budget_grid = [0.25, 0.75]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
        budget_grid = [float(x) for x in args.budgets.split(",")]

    out_dir = ROOT / "results" / "downstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"downstream_utility_{args.dataset}.csv"

    all_rows = []
    for seed in seeds:
        try:
            all_rows += run_seed(args.dataset, seed, args.sample_size,
                                 budget_grid, args.target_recall, config)
        except Exception as e:
            logger.error(f"[seed={seed}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        logger.info(f"[seed={seed}] checkpoint -> {csv_path} ({len(all_rows)} rows)")

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    report = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset": args.dataset,
            "sample_size": args.sample_size,
            "target_recall": args.target_recall,
            "seeds": seeds,
            "budget_grid": budget_grid,
            "downstream_model": "RandomForest(n_estimators=200, class_weight=balanced)",
            "design": "dedup the train pool; train independent detector on kept subset; "
                      "evaluate on held-out test partition",
        },
        "per_method": aggregate(df),
    }
    json_path = out_dir / f"downstream_utility_{args.dataset}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Console summary: AUPRC vs savings, CAP vs anomaly-blind.
    print("\n" + "=" * 78)
    print(f"DOWNSTREAM UTILITY ({args.dataset.upper()}) - held-out fault detection AUPRC")
    print("=" * 78)
    methods = ["cap_dedup", "kcenter_blind", "random", "reservoir", "topk_score"]
    bsorted = sorted({f"{b:.2f}" for b in budget_grid})
    header = "method".ljust(16) + "".join(f"  sav~{(1-float(b))*100:4.0f}%" for b in bsorted)
    print(header)
    for mth in methods:
        if mth not in report["per_method"]:
            continue
        line = mth.ljust(16)
        for b in bsorted:
            cell = report["per_method"][mth].get(b)
            line += f"  {cell['auprc_mean']:.3f}    " if cell else "    n/a    "
        print(line)
    print(f"\nCSV:  {csv_path}\nJSON: {json_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
