#!/usr/bin/env python3
"""
Empirical exchangeability check.

Theorem 1 assumes that the fault scores in the calibration set and the
fault scores in the test set are exchangeable. Under our stratified
random splitting protocol this is satisfied by construction, but we still
report empirical evidence: a two-sample Kolmogorov-Smirnov test on the
calibration-positive vs test-positive score distributions across the 10
seeds. Under exchangeability, the two distributions should be
statistically indistinguishable; large p-values support the assumption.

Output:
    results/exchangeability_ks_test.csv
    results/exchangeability_ks_test.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from skab_loader import load_skab          # noqa: E402
from swat_loader import load_swat          # noqa: E402
from data_splitter import stratified_split  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402


def per_seed_ks(X: np.ndarray, y: np.ndarray, seeds: list) -> list:
    rows = []
    for seed in seeds:
        masks = stratified_split(y, seed, mode="stratified",
                                  ratios=(0.7, 0.1, 0.1, 0.1))
        cal_idx = np.where(masks["cal"])[0]
        test_idx = np.where(masks["test"])[0]
        train_idx = np.where(masks["train"] | masks["val"])[0]
        # Isolation Forest scorer
        iforest = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1)
        iforest.fit(X[train_idx])
        cal_pos = -iforest.score_samples(X[cal_idx][y[cal_idx] == 1])
        test_pos = -iforest.score_samples(X[test_idx][y[test_idx] == 1])
        if len(cal_pos) < 5 or len(test_pos) < 5:
            continue
        ks_stat, ks_p = stats.ks_2samp(cal_pos, test_pos, alternative="two-sided")
        rows.append({
            "seed": int(seed),
            "n_cal_pos":  int(len(cal_pos)),
            "n_test_pos": int(len(test_pos)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue":    float(ks_p),
            "exchangeable_at_alpha_005": bool(ks_p > 0.05),
        })
    return rows


def main():
    out_dir = ROOT / "results"
    seeds = list(range(42, 52))
    summary = {"per_dataset": {}}
    all_rows = []

    print("=" * 80)
    print(f"{'Empirical exchangeability check (KS two-sample, cal-pos vs test-pos)':^80}")
    print("=" * 80)

    for ds_name, loader in [("SKAB", load_skab), ("SWaT", load_swat)]:
        print(f"\nLoading {ds_name}...")
        feats_df, y, _raw = loader()
        X = feats_df.to_numpy(dtype=np.float32)
        rows = per_seed_ks(X, y, seeds)
        for r in rows:
            r["dataset"] = ds_name
        all_rows.extend(rows)
        p_vals = [r["ks_pvalue"] for r in rows]
        ks_stats = [r["ks_statistic"] for r in rows]
        n_pass = sum(r["exchangeable_at_alpha_005"] for r in rows)
        block = {
            "n_seeds": len(rows),
            "p_value_mean": float(np.mean(p_vals)),
            "p_value_min":  float(np.min(p_vals)),
            "ks_statistic_mean": float(np.mean(ks_stats)),
            "n_seeds_failing_to_reject_exchangeability_at_005": n_pass,
        }
        summary["per_dataset"][ds_name] = block
        print(f"  {ds_name}: across {len(rows)} seeds, "
              f"KS p-value mean={block['p_value_mean']:.3f}, "
              f"min={block['p_value_min']:.3f}, "
              f"{n_pass}/{len(rows)} seeds DO NOT REJECT exchangeability at alpha=0.05")

    pd.DataFrame(all_rows).to_csv(out_dir / "exchangeability_ks_test.csv", index=False)
    with open(out_dir / "exchangeability_ks_test.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir/'exchangeability_ks_test.csv'} and .json")


if __name__ == "__main__":
    main()
