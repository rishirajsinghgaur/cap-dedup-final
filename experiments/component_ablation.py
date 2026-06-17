#!/usr/bin/env python3
"""Component ablation: gate-only vs gate+random-fill vs gate+coverage-coreset at matched budget and shared recall floor, scored by held-out fault-detection AUPRC (supervised RandomForest or one-class IsolationForest)."""
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
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(THIS))

from anomaly_scorers import build_default_scorers          # noqa: E402
from conformal_layer0 import ConformalAnomalyGate          # noqa: E402
from submodular_coreset import CoverageCoreset             # noqa: E402
from downstream_utility import evaluate_downstream         # noqa: E402  (reuse identical protocol)

from sklearn.ensemble import IsolationForest               # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


def evaluate_oneclass(X_keep, y_keep, X_eval, y_eval, seed):
    """UNSUPERVISED downstream: fit a one-class detector (IsolationForest) on the
    KEPT NORMAL samples only, score the held-out set, measure fault-detection AUPRC.
    This is the task the coverage coreset is DESIGNED for: if it preserves a diverse
    span of the normal manifold (vs random fill that over-samples common modes), the
    normal model should generalise better and flag faults more cleanly. Eval labels
    are NEVER used for fitting."""
    Xn = X_keep[y_keep == 0]
    pos = float((y_eval == 1).mean())
    if len(Xn) < 20 or len(np.unique(y_eval)) < 2:
        return dict(n_keep=int(len(y_keep)), n_normal_keep=int(len(Xn)),
                    auprc=float(pos), roc_auc=0.5, single_class=True,
                    recall=0.0, precision=0.0, f1=0.0, n_faults_keep=int((y_keep == 1).sum()))
    clf = IsolationForest(n_estimators=200, random_state=seed, n_jobs=2).fit(Xn)
    score = -clf.score_samples(X_eval)   # higher = more anomalous
    return dict(n_keep=int(len(y_keep)), n_normal_keep=int(len(Xn)),
                n_faults_keep=int((y_keep == 1).sum()), single_class=False,
                auprc=float(average_precision_score(y_eval, score)),
                roc_auc=float(roc_auc_score(y_eval, score)),
                recall=0.0, precision=0.0, f1=0.0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("component_ablation")


def gate_random_fill(must_preserve, budget, seed):
    """keep = must_preserve U (random sample of the rest up to budget)."""
    keep = must_preserve.copy()
    n_must = int(keep.sum())
    if budget > n_must:
        cand = np.where(~must_preserve)[0]
        rng = np.random.default_rng(seed)
        k = min(budget - n_must, len(cand))
        if k > 0:
            keep[rng.choice(cand, size=k, replace=False)] = True
    return keep


def run_seed(dataset, seed, sample_size, budget_grid, target_recall, config, evaluate):
    if dataset == "tep":
        from pareto_sweep_tep import train_for_seed
        art = train_for_seed(seed, sample_size, config, label_type="gt_faults", split_mode="stratified")
    elif dataset == "swat":
        from pareto_sweep_swat import train_for_seed
        art = train_for_seed(seed, sample_size, config, label_type="gt_attack", split_mode="stratified")
    else:
        raise ValueError(dataset)

    fw = art["framework"]
    X_train, y_train = art["X_train"], art["y_train"]
    X_cal, y_cal = art["X_cal"], art["y_cal"]
    X_test, y_test = art["X_test"], art["y_test"]
    n_pool = len(X_train)

    emb_pool = np.asarray(fw.get_embeddings(X_train, use_siamese=True))
    scorers = build_default_scorers()
    s_pool = s_cal = None
    for name, sc in scorers.items():
        if name != "bnn_mean":
            continue
        sc.fit(X_train, y_train, framework=fw, seed=int(seed))
        s_pool = np.asarray(sc.score(X_train)).flatten().astype(float)
        s_cal = np.asarray(sc.score(X_cal)).flatten().astype(float)
    if s_pool is None:
        raise RuntimeError("bnn_mean unavailable")

    gate = ConformalAnomalyGate(target_recall=target_recall).fit(
        X_cal, y_cal, lambda _X, _s=s_cal: _s)
    must = gate.preserve_mask(X_train, lambda _X, _s=s_pool: _s)
    n_must = int(must.sum())
    logger.info(f"[seed={seed}] pool={n_pool} must-preserve={n_must} "
                f"({n_must/n_pool*100:.1f}%) faults_in_pool={int((y_train==1).sum())}")

    rows = []
    # gate_only is a single operating point (keep == M).
    for label, keep in [("gate_only", must.copy())]:
        m = evaluate(X_train[keep], y_train[keep], X_test, y_test, int(seed))
        m.update(dataset=dataset, seed=int(seed), method=label,
                 budget_frac=float(n_must / n_pool),
                 storage_savings_pct=float((1 - n_must / n_pool) * 100), n_pool=n_pool)
        rows.append(m)

    for bf in budget_grid:
        budget = int(round(bf * n_pool))
        if budget <= n_must:
            continue  # below the must-preserve floor; gate_only already covers it
        masks = {
            "gate_random": gate_random_fill(must, budget, int(seed)),
            "gate_coreset": CoverageCoreset(seed=int(seed)).select(emb_pool, s_pool, must, budget),
        }
        for method, keep in masks.items():
            keep = np.asarray(keep, dtype=bool)
            m = evaluate(X_train[keep], y_train[keep], X_test, y_test, int(seed))
            m.update(dataset=dataset, seed=int(seed), method=method,
                     budget_frac=float(bf), storage_savings_pct=float((1 - budget / n_pool) * 100),
                     n_pool=n_pool)
            rows.append(m)
            logger.info(f"[seed={seed}] sav={m['storage_savings_pct']:.0f}% {method:<13} "
                        f"keep={m['n_keep']} -> auprc={m['auprc']:.3f} recall={m['recall']:.3f}")
    del fw, art; gc.collect()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tep", "swat"], default="tep")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--sample-size", type=int, default=10000)
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--budgets", type=str, default="0.25,0.50,0.75,0.90")
    p.add_argument("--mode", choices=["supervised", "oneclass"], default="oneclass",
                   help="downstream evaluator: supervised RF, or one-class IF on kept normals "
                        "(the task the coverage coreset is designed for)")
    args = p.parse_args()

    evaluate = evaluate_oneclass if args.mode == "oneclass" else evaluate_downstream
    import yaml
    config = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    seeds = list(range(42, 42 + args.seeds))
    budget_grid = [float(x) for x in args.budgets.split(",")]
    out_dir = ROOT / "results" / "ablation"; out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"component_ablation_{args.dataset}_{args.mode}.csv"

    rows = []
    for sd in seeds:
        try:
            rows += run_seed(args.dataset, sd, args.sample_size, budget_grid, args.target_recall, config, evaluate)
        except Exception as e:
            logger.error(f"[seed={sd}] FAILED: {e}"); import traceback; traceback.print_exc(); continue
        pd.DataFrame(rows).to_csv(csv, index=False)

    df = pd.DataFrame(rows)
    # Aggregate AUPRC mean per (method, budget).
    print("\n" + "=" * 76)
    print(f"COMPONENT ABLATION ({args.dataset.upper()}, {args.mode}) - held-out AUPRC (mean over {len(seeds)} seeds)")
    print("does gate+coreset beat gate+random at matched budget? (both share the gate)")
    print("=" * 76)
    for bf in sorted(df[df.method != "gate_only"].budget_frac.unique()):
        sav = (1 - bf) * 100
        gr = df[(df.method == "gate_random") & (df.budget_frac == bf)].auprc
        gc_ = df[(df.method == "gate_coreset") & (df.budget_frac == bf)].auprc
        print(f"  savings~{sav:4.0f}%:  gate_random={gr.mean():.3f}   gate_coreset={gc_.mean():.3f}   "
              f"delta={gc_.mean()-gr.mean():+.3f}")
    go = df[df.method == "gate_only"]
    if len(go):
        print(f"  gate_only: savings~{go.storage_savings_pct.mean():.0f}%  auprc={go.auprc.mean():.3f}  "
              f"recall={go.recall.mean():.3f}")
    report = {"dataset": args.dataset, "seeds": seeds, "budget_grid": budget_grid,
              "timestamp": datetime.utcnow().isoformat() + "Z",
              "rows": df.to_dict(orient="records")}
    (out_dir / f"component_ablation_{args.dataset}_{args.mode}.json").write_text(json.dumps(report, indent=2))
    print(f"\nCSV: {csv}")


if __name__ == "__main__":
    main()
