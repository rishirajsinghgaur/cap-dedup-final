#!/usr/bin/env python3
"""
Stage 11: Empirical drift / rolling-window re-calibration ablation.

Addresses a robustness request
exchangeability assumption may not hold in production because calibration
and test windows are collected at different times. We empirically simulate
this drift on TEP by splitting the 20 fault classes into an "old" subset
(faults 1-10, used for calibration) and a "new" subset (faults 11-20, used
for test). We then compare three conformal-calibration policies:

  (a) static cold-calibration:
        tau is computed from the 'old' calibration draw and never updated.
        Recall on the 'new' (different fault types) test stream is the
        worst-case drift result.

  (b) static warm-calibration:
        same as (a), but the calibration set is a *fresh* draw from the
        post-drift distribution (the 'new' faults). This is the upper
        bound: it shows what would be achievable if the operator had the
        opportunity to recalibrate after every drift event.

  (c) rolling-window re-calibration (every K_recal samples):
        every K_recal test samples we re-draw the calibration set from the
        most recent rolling window of preserved positives, recompute tau,
        and continue. K_recal in {500, 1000} is reported.

Output:
    results/drift_recalibration.csv
    results/drift_recalibration.json

The result:
  - how much recall drops under cold (a),
  - how much recall is recovered by rolling re-cal (c) vs the warm baseline (b),
  - the cost per recalibration in latency.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # for tep_data_loader at repo root
sys.path.insert(0, str(ROOT / "experiments"))

from tep_data_loader import TEPDataLoader  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402


def conformal_tau(scores_pos: np.ndarray, target_recall: float) -> float:
    """Standard split-conformal threshold."""
    alpha = 1.0 - target_recall
    n = len(scores_pos)
    k = int(np.floor(alpha * (n + 1)))
    k = max(1, min(k, n))
    return float(np.sort(scores_pos)[k - 1])


def load_tep_subset(sample_size: int = 20_000, random_state: int = 42):
    """Load a TEP sample with fault labels attached."""
    loader = TEPDataLoader()
    df, safety_labels, _dup_pairs, raw_df = loader.load_data(
        sample_size=sample_size, random_state=random_state, label_type="gt_faults"
    )
    safety_labels = safety_labels.astype(int)
    # Per-sample fault id is in raw_df["faultNumber"] (TEP convention)
    if "faultNumber" in raw_df.columns:
        fault_id = raw_df["faultNumber"].to_numpy().astype(int)
    else:
        # Synthesize a single-fault label = 1 for every anomalous row
        fault_id = safety_labels.copy()
    X = df.to_numpy(dtype=np.float32)
    return X, safety_labels, fault_id


def split_by_fault(X, y, fault_id, old_fault_max: int = 10):
    """Build calibration (old faults) and test (new faults) draws."""
    old_mask = (fault_id <= old_fault_max) & (fault_id > 0)
    new_mask = (fault_id > old_fault_max)
    cal_pos_mask = old_mask & (y == 1)
    test_mask    = new_mask | (fault_id == 0)
    return cal_pos_mask, test_mask


def fit_score_iforest(X_train, X_score, seed):
    iforest = IsolationForest(n_estimators=200, contamination="auto",
                               random_state=seed, n_jobs=1)
    iforest.fit(X_train)
    return -iforest.score_samples(X_score)


def policy_cold(scores_test, y_test, scores_cal_pos, target_recall):
    tau = conformal_tau(scores_cal_pos, target_recall)
    preserved = scores_test >= tau
    recall = preserved[y_test == 1].sum() / max(1, (y_test == 1).sum())
    return float(tau), float(recall)


def policy_warm(scores_test, y_test, scores_cal_pos_post, target_recall):
    tau = conformal_tau(scores_cal_pos_post, target_recall)
    preserved = scores_test >= tau
    recall = preserved[y_test == 1].sum() / max(1, (y_test == 1).sum())
    return float(tau), float(recall)


def policy_rolling(
    scores_test, y_test, scores_cal_pos_init, target_recall,
    K_recal: int, window_size: int = 1000
):
    """Rolling-window recalibration: every K_recal preserved positives we
    re-draw cal_pos from the most recent window_size preserved positives.

    Returns:
        recall : overall recall on test stream
        recal_count : number of re-calibrations triggered
        ms_per_recal : time per recalibration event
    """
    n = len(scores_test)
    cal_pos = list(scores_cal_pos_init)
    tau = conformal_tau(np.asarray(cal_pos), target_recall)
    preserved = np.zeros(n, dtype=bool)
    recal_count = 0
    recal_times = []
    pos_seen_since_recal = 0
    seen_pos_scores = []

    for i in range(n):
        if scores_test[i] >= tau:
            preserved[i] = True
            if y_test[i] == 1:
                seen_pos_scores.append(scores_test[i])
                pos_seen_since_recal += 1
        if pos_seen_since_recal >= K_recal:
            t0 = time.perf_counter()
            # rolling window: most recent window_size preserved positives
            window = seen_pos_scores[-window_size:]
            if len(window) >= 30:
                tau = conformal_tau(np.asarray(window), target_recall)
            recal_times.append(time.perf_counter() - t0)
            pos_seen_since_recal = 0
            recal_count += 1

    n_pos = max(1, (y_test == 1).sum())
    recall = preserved[y_test == 1].sum() / n_pos
    return {
        "recall": float(recall), "recal_count": recal_count,
        "ms_per_recal": float(np.mean(recal_times) * 1000) if recal_times else 0.0,
    }


def main():
    target_recall = 0.95
    seeds = [42, 43, 44, 45, 46]
    rows = []

    for seed in seeds:
        print(f"\n[seed={seed}] loading TEP ...")
        X, y, fault_id = load_tep_subset(sample_size=20_000, random_state=seed)

        # Train Isolation Forest on the FIRST 50% of rows (chronological proxy)
        # to keep it independent of both calibration and test
        n_total = len(X)
        train_end = n_total // 2
        train_idx = np.arange(0, train_end)
        rest_idx = np.arange(train_end, n_total)

        # Fault split among the held-out rest
        X_rest, y_rest, fid_rest = X[rest_idx], y[rest_idx], fault_id[rest_idx]
        cal_pos_mask, test_mask = split_by_fault(X_rest, y_rest, fid_rest)
        n_cal_pos = int(cal_pos_mask.sum())
        n_test    = int(test_mask.sum())
        n_test_pos = int((y_rest[test_mask] == 1).sum())
        print(f"   n_train={train_end}, n_cal_pos={n_cal_pos} "
              f"(faults 1-10 only), n_test={n_test} (faults 11-20 + normal), "
              f"test_anomalies={n_test_pos}")
        if n_cal_pos < 30 or n_test_pos < 30:
            print(f"   insufficient positives, skipping seed {seed}")
            continue

        scores_all = fit_score_iforest(X[train_idx], X_rest, seed)
        scores_cal_pos = scores_all[cal_pos_mask]
        scores_test    = scores_all[test_mask]
        y_test         = y_rest[test_mask]

        # Warm draw (the unrealistic upper bound): post-drift positives
        post_drift_pos_mask = (test_mask) & (y_rest == 1)
        scores_post_drift_pos = scores_all[post_drift_pos_mask]
        # Use a held-out fraction of post-drift positives for the warm policy
        rng = np.random.default_rng(seed)
        if len(scores_post_drift_pos) >= 60:
            picked = rng.choice(len(scores_post_drift_pos), size=len(scores_post_drift_pos)//2, replace=False)
            scores_cal_pos_post = scores_post_drift_pos[picked]
        else:
            scores_cal_pos_post = scores_post_drift_pos

        # (a) cold
        tau_cold, rec_cold = policy_cold(scores_test, y_test, scores_cal_pos, target_recall)
        # (b) warm
        tau_warm, rec_warm = policy_warm(scores_test, y_test, scores_cal_pos_post, target_recall)
        # (c) rolling K=100 and K=300 (preserved-positive events)
        rec_roll_100 = policy_rolling(scores_test, y_test, scores_cal_pos,
                                       target_recall, K_recal=100, window_size=300)
        rec_roll_300 = policy_rolling(scores_test, y_test, scores_cal_pos,
                                       target_recall, K_recal=300, window_size=500)

        rows.append({
            "seed":           seed,
            "n_cal_pos":      n_cal_pos,
            "n_test":         n_test,
            "n_test_pos":     n_test_pos,
            "target_recall":  target_recall,
            "tau_cold":       tau_cold,
            "recall_cold":    rec_cold,
            "tau_warm":       tau_warm,
            "recall_warm":    rec_warm,
            "recall_rolling_K100":     rec_roll_100["recall"],
            "n_recal_K100":            rec_roll_100["recal_count"],
            "ms_per_recal_K100":       rec_roll_100["ms_per_recal"],
            "recall_rolling_K300":     rec_roll_300["recall"],
            "n_recal_K300":            rec_roll_300["recal_count"],
            "ms_per_recal_K300":       rec_roll_300["ms_per_recal"],
            "drift_recovery_pct_K100": (rec_roll_100["recall"] - rec_cold) /
                                        max(rec_warm - rec_cold, 1e-9) * 100.0,
            "drift_recovery_pct_K300": (rec_roll_300["recall"] - rec_cold) /
                                        max(rec_warm - rec_cold, 1e-9) * 100.0,
        })

        print(f"   target {target_recall:.0%}:"
              f"  cold={rec_cold:.3f}"
              f"  warm={rec_warm:.3f}"
              f"  roll(K=100)={rec_roll_100['recall']:.3f}"
              f"  roll(K=300)={rec_roll_300['recall']:.3f}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "drift_recalibration.csv", index=False)

    summary = {"target_recall": target_recall, "n_seeds": len(rows),
                "per_policy": {}}
    if len(rows) > 0:
        for col, label in [
            ("recall_cold",          "cold (no re-cal)"),
            ("recall_warm",          "warm (idealised re-cal)"),
            ("recall_rolling_K100",  "rolling K=100 preserved positives"),
            ("recall_rolling_K300",  "rolling K=300 preserved positives"),
        ]:
            vals = df[col].dropna().to_numpy(dtype=float)
            if len(vals) >= 3:
                summary["per_policy"][label] = {
                    "recall_mean": float(vals.mean()),
                    "recall_std":  float(vals.std(ddof=1)),
                    "n_seeds":     int(len(vals)),
                }
        # Drift recovery (rolling vs warm)
        for col, label in [
            ("drift_recovery_pct_K100", "drift recovery K=100"),
            ("drift_recovery_pct_K300", "drift recovery K=300"),
        ]:
            vals = df[col].dropna().to_numpy(dtype=float)
            if len(vals) >= 3:
                summary[label] = {
                    "recovery_pct_mean": float(vals.mean()),
                    "recovery_pct_std":  float(vals.std(ddof=1)),
                }
    with open(out_dir / "drift_recalibration.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary (mean +/- std across seeds):")
    for label, blk in summary["per_policy"].items():
        print(f"  {label:<45}: recall = {blk['recall_mean']:.3f} +/- {blk['recall_std']:.3f}")
    print(f"\nSaved: {out_dir/'drift_recalibration.csv'}")
    print(f"Saved: {out_dir/'drift_recalibration.json'}")


if __name__ == "__main__":
    main()
