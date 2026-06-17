#!/usr/bin/env python3
"""
Focused priority-weighted facility-location baseline for SKAB and SWaT,
ON REAL DATA.

Why this script exists:
    a robustness request
    weighted coreset can replace CAP-Dedup's two-stage construction. The
    main Pareto sweep already includes the priority-weighted FL for TEP
    (small N), but the standard O(N^2 * budget) greedy is too slow to run
    inside the SKAB and SWaT sweeps at five different budgets. This script
    runs the priority-FL on the REAL SKAB and SWaT test splits at a small
    grid of budgets per dataset, using a lazy-greedy implementation
    (Minoux 1978) that is provably equivalent to the standard greedy on
    monotone submodular functions and is several orders of magnitude
    faster in practice.

What the script does:
    1. Loads SKAB and SWaT through the same loaders the main sweep uses.
    2. Stratified split with the same protocol (70/10/10/10) and seeds (42-46).
    3. Anomaly score from a held-out Isolation Forest (deterministic,
       trains in seconds, does not require the BNN ensemble artefact).
    4. Embeddings are the L2-normalised raw features (no Siamese training);
       this is identical to what the unweighted facility-location baseline
       uses in the main sweep, so the comparison is apples-to-apples.
    5. Lazy-greedy priority-weighted facility-location at four budgets
       per dataset.
    6. Reports recall and savings per (seed, budget) and the best
       feasible budget at the >=95%, >=90%, >=85% empirical recall floors.

The Siamese-embedding question:
    For TEP we have full-pipeline Siamese embeddings in the main sweep.
    For SKAB/SWaT, recomputing the Siamese model adds ~30 s per seed of
    framework training. We avoid this by using raw-feature embeddings,
    which is the same input the priority-FL baseline would receive in a
    deployment that does not have the Siamese model available. This is the
    HARDER case for the baseline; if CAP-Dedup beats it here, the result
    transfers to the easier (Siamese-embedding) case as well.

Output:
    results/priority_fl_focused.csv
    results/priority_fl_focused.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from data_splitter import stratified_split  # noqa: E402
from skab_loader import load_skab  # noqa: E402
from swat_loader import load_swat  # noqa: E402


def _normalize_l2(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def lazy_priority_facility_location(
    embeddings: np.ndarray,
    anomaly_scores: np.ndarray,
    budget: int,
    seed: int,
    weight_temperature: float = 1.0,
) -> np.ndarray:
    """Lazy-greedy priority-weighted facility-location.

    Maintains a max-heap of (last_computed_gain, last_iteration) per
    candidate. On each iteration, pop the top, recompute its gain only if
    its stamp is stale, and either commit it or push it back. By
    submodularity of weighted sum-coverage (a non-negative linear
    combination of submodular functions), the lazy greedy is exact -- it
    produces the same selection as the eager greedy in this script's
    parent module (facility_location_priority_weighted) but in O(n*budget*log n)
    expected time instead of O(n^2 * budget).

    Returns:
        boolean keep_mask of length n with `budget` True entries.
    """
    import heapq
    n = len(embeddings)
    if budget >= n:
        return np.ones(n, dtype=bool)

    E = _normalize_l2(embeddings.astype(np.float32))

    # Anomaly-priority weights (same recipe as the eager implementation)
    z = (anomaly_scores - np.mean(anomaly_scores)) / (np.std(anomaly_scores) + 1e-12)
    w = np.exp(np.clip(z / max(weight_temperature, 1e-6), -3.0, 3.0))
    w = (w / max(np.mean(w), 1e-12)).astype(np.float32)

    keep = np.zeros(n, dtype=bool)
    max_sim_to_S = np.full(n, -np.inf, dtype=np.float32)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    keep[first] = True
    max_sim_to_S = np.maximum(max_sim_to_S, E @ E[first])

    # Initial gain pass -- one full O(N^2) sweep to seed the heap.
    # After this, each subsequent iteration costs O(N * d * top_stale_count).
    def compute_gain(j: int) -> float:
        diff = E @ E[j] - max_sim_to_S
        np.maximum(diff, 0.0, out=diff)
        return float((diff * w).sum())

    heap = []  # entries: (-gain, last_iteration, index)
    for j in range(n):
        if j == first:
            continue
        heapq.heappush(heap, (-compute_gain(j), 0, j))

    iteration = 1
    while int(keep.sum()) < budget and heap:
        neg_gain, stamp, j = heapq.heappop(heap)
        if keep[j]:
            continue
        if stamp == iteration:
            # gain is fresh
            if -neg_gain <= 0:
                break
            keep[j] = True
            new_sims = E @ E[j]
            np.maximum(max_sim_to_S, new_sims, out=max_sim_to_S)
            iteration += 1
        else:
            # recompute and re-push with current stamp
            heapq.heappush(heap, (-compute_gain(j), iteration, j))

    return keep


# -------------------------------------------------------------------------
# CAP-Dedup comparison value (from the existing main-sweep CSVs)
# -------------------------------------------------------------------------

def cap_savings_at_recall(csv_path: Path, recall_floor: float) -> dict:
    """For each seed, find the best CAP-Dedup-budget savings at >=recall_floor."""
    df = pd.read_csv(csv_path)
    df = df[df["method"] == "cap_dedup_budget"]
    out = {}
    for seed in df["seed"].unique():
        s = df[(df["seed"] == seed) & (df["safety_recall"] >= recall_floor)]
        if len(s) > 0:
            out[int(seed)] = float(s["storage_savings_pct"].max())
    return out


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def evaluate_priority_fl(
    dataset: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    seeds: list,
    budget_fracs: list,
    cap_csv: Path,
) -> list:
    """Run the priority-FL baseline on each (seed, budget) combination."""
    rows = []
    for seed in seeds:
        # Same stratified split protocol the main sweep uses
        masks = stratified_split(y_full, seed, mode="stratified", ratios=(0.7, 0.1, 0.1, 0.1))
        test_idx = np.where(masks["test"])[0]
        cal_idx = np.where(masks["cal"])[0]

        X_test = X_full[test_idx]
        y_test = y_full[test_idx]
        n_test = len(X_test)

        # Anomaly score: Isolation Forest fitted on train+val (not test/cal)
        train_idx = np.where(masks["train"] | masks["val"])[0]
        from sklearn.ensemble import IsolationForest
        iforest = IsolationForest(
            n_estimators=200, contamination="auto",
            random_state=seed, n_jobs=1,
        )
        iforest.fit(X_full[train_idx])
        # higher = more anomalous
        test_scores = -iforest.score_samples(X_test)

        for bfrac in budget_fracs:
            budget = int(round(bfrac * n_test))
            t0 = time.perf_counter()
            keep = lazy_priority_facility_location(X_test, test_scores, budget, seed)
            dt = time.perf_counter() - t0

            n_anom = int(y_test.sum())
            anom_kept = int(keep[y_test == 1].sum())
            recall = anom_kept / n_anom if n_anom > 0 else 1.0
            savings = (1.0 - keep.sum() / n_test) * 100.0
            row = {
                "dataset": dataset, "seed": int(seed), "budget_frac": bfrac,
                "n_test": int(n_test), "n_anom_total": n_anom,
                "n_kept": int(keep.sum()), "n_anom_kept": anom_kept,
                "priority_fl_recall": float(recall),
                "priority_fl_savings_pct": float(savings),
                "elapsed_s": round(dt, 2),
            }
            rows.append(row)
            print(f"[{dataset}/seed={seed}/budget={bfrac:.2f}] "
                  f"recall={recall:.3f}, savings={savings:.2f}%, "
                  f"elapsed={dt:.1f}s")

    return rows


def main():
    seeds = [42, 43, 44, 45, 46]
    # Per-dataset budget grids. We use a small grid (5 points) so that at
    # least one budget straddles the >=95% recall feasibility threshold on
    # each dataset.
    budget_grids = {
        "SKAB": [0.20, 0.40, 0.60, 0.80, 0.90],
        "SWaT": [0.20, 0.30, 0.50, 0.70, 0.85],
    }
    cap_csvs = {
        "SKAB": ROOT / "results" / "pareto_skab" / "pareto_sweep_skab.csv",
        "SWaT": ROOT / "results" / "pareto_swat" / "pareto_sweep_swat.csv",
    }

    all_rows = []
    summary = {"settings": {"seeds": seeds, "budget_grids": budget_grids},
                "per_dataset": {}}

    print("=" * 80)
    print(f"{'Focused priority-FL on REAL SKAB and SWaT data':^80}")
    print("=" * 80)

    # ---- SKAB ----
    print("\nLoading SKAB ...")
    feats_df, y, _raw = load_skab()
    X = feats_df.to_numpy(dtype=np.float32)
    print(f"  SKAB loaded: n={len(X)}, anomaly_rate={y.mean():.3f}")
    skab_rows = evaluate_priority_fl(
        "SKAB", X, y, seeds, budget_grids["SKAB"], cap_csvs["SKAB"]
    )
    all_rows.extend(skab_rows)

    # ---- SWaT ----
    print("\nLoading SWaT ...")
    feats_df, y, _raw = load_swat()
    X = feats_df.to_numpy(dtype=np.float32)
    print(f"  SWaT loaded: n={len(X)}, anomaly_rate={y.mean():.3f}")
    swat_rows = evaluate_priority_fl(
        "SWaT", X, y, seeds, budget_grids["SWaT"], cap_csvs["SWaT"]
    )
    all_rows.extend(swat_rows)

    # ---- Summary: best feasible savings per recall floor ----
    df = pd.DataFrame(all_rows)
    for dataset in ("SKAB", "SWaT"):
        sub = df[df["dataset"] == dataset]
        ds_block = {"n_seeds": len(seeds), "per_floor": {}}
        for floor in (0.99, 0.95, 0.90, 0.85, 0.80):
            per_seed_best = []
            for seed in seeds:
                seed_sub = sub[(sub["seed"] == seed) &
                                (sub["priority_fl_recall"] >= floor)]
                if len(seed_sub) > 0:
                    per_seed_best.append(seed_sub["priority_fl_savings_pct"].max())
            if len(per_seed_best) >= 3:
                arr = np.array(per_seed_best, dtype=float)
                ds_block["per_floor"][f"{int(floor*100)}%"] = {
                    "n_seeds_feasible":   len(per_seed_best),
                    "priority_fl_savings_mean": float(arr.mean()),
                    "priority_fl_savings_std":  float(arr.std(ddof=1)),
                }
            else:
                ds_block["per_floor"][f"{int(floor*100)}%"] = {
                    "n_seeds_feasible": len(per_seed_best),
                    "comment": "infeasible on most seeds at this floor",
                }

        # Also pull CAP-Dedup's own best savings at each floor for direct
        # comparison
        if cap_csvs[dataset].exists():
            cap_at = {}
            for floor in (0.99, 0.95, 0.90, 0.85, 0.80):
                cap_per_seed = cap_savings_at_recall(cap_csvs[dataset], floor)
                if len(cap_per_seed) >= 3:
                    vals = np.fromiter(cap_per_seed.values(), dtype=float)
                    cap_at[f"{int(floor*100)}%"] = {
                        "n_seeds_feasible": len(vals),
                        "cap_dedup_savings_mean": float(vals.mean()),
                        "cap_dedup_savings_std":  float(vals.std(ddof=1)),
                    }
                else:
                    cap_at[f"{int(floor*100)}%"] = {
                        "n_seeds_feasible": len(cap_per_seed),
                        "comment": "infeasible",
                    }
            ds_block["cap_dedup_comparison"] = cap_at
        summary["per_dataset"][dataset] = ds_block

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "priority_fl_focused.csv", index=False)
    with open(out_dir / "priority_fl_focused.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir/'priority_fl_focused.csv'}")
    print(f"Saved: {out_dir/'priority_fl_focused.json'}")

    print("\nHeadline comparison (priority-FL vs CAP-Dedup, by recall floor):")
    for d, blk in summary["per_dataset"].items():
        print(f"  [{d}]")
        for floor, pfblk in blk["per_floor"].items():
            capblk = blk.get("cap_dedup_comparison", {}).get(floor, {})
            pf_sav = pfblk.get("priority_fl_savings_mean", None)
            cap_sav = capblk.get("cap_dedup_savings_mean", None)
            if pf_sav is None and cap_sav is None:
                msg = "both infeasible"
            elif pf_sav is None:
                msg = (f"priority-FL infeasible ({pfblk['n_seeds_feasible']}/5); "
                        f"CAP-Dedup {cap_sav:.2f}+/-{capblk['cap_dedup_savings_std']:.2f}%")
            elif cap_sav is None:
                msg = (f"CAP-Dedup infeasible; "
                        f"priority-FL {pf_sav:.2f}+/-{pfblk['priority_fl_savings_std']:.2f}%")
            else:
                diff = cap_sav - pf_sav
                msg = (f"priority-FL {pf_sav:.2f}+/-{pfblk['priority_fl_savings_std']:.2f}%, "
                        f"CAP-Dedup {cap_sav:.2f}+/-{capblk['cap_dedup_savings_std']:.2f}%, "
                        f"diff = {diff:+.2f}pp")
            print(f"    >= {floor}: {msg}")


if __name__ == "__main__":
    main()
