#!/usr/bin/env python3
"""
Pareto Frontier Sweep on SWaT (Phase 2 multi-dataset validation).

Same CAP-Dedup framework as pareto_sweep_tep.py applied to the SWaT
(Secure Water Treatment) A4 & A5 Jul-2019 release:
  - 43 process sensor features (after dropping null/constant columns)
  - Binary attack labels (ground truth, derived from PDF attack timestamps)
  - ~15,000 rows (1 Hz, 4 hours)
  - 13% attack rate (6 distinct attack episodes)

Splits by file_id (each attack episode + each normal stretch between them
is one episode -> 13 episodes total) so train/test respects attack boundaries.

USAGE:
  python experiments/pareto_sweep_swat.py                # full 10-seed sweep
  python experiments/pareto_sweep_swat.py --quick        # 2-seed smoke test

Output: results/pareto_swat/{pareto_sweep_swat.csv, knee_report.json, *.png}
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# CPU-only mode for stability (matches tep_experiment.py)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(2)

# Path setup — this script lives in experiments/, so ROOT is one level up
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(THIS_DIR))  # so sibling module conformal_layer0 imports cleanly

from swat_loader import load_swat  # noqa: E402
from core.framework import UncertaintyAwareFramework  # noqa: E402
from conformal_layer0 import ConformalAnomalyGate  # noqa: E402
from anomaly_scorers import build_default_scorers  # noqa: E402
from submodular_coreset import CoverageCoreset  # noqa: E402
from data_splitter import stratified_split  # noqa: E402
from literature_baselines import build_baseline_keep_masks  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pareto_sweep")


# ----------------------------------------------------------------------------
# Helpers (copied/adapted from tep_experiment.py so this script is self-contained)
# ----------------------------------------------------------------------------

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_uncertainty(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(float).flatten()
    rng = scores.max() - scores.min()
    if rng < 1e-10:
        return np.zeros_like(scores)
    return (scores - scores.min()) / (rng + 1e-10)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def compute_removed_indices(duplicates):
    """Apply union-find on duplicate pairs, return set of indices to remove
    (keeping the minimum-indexed representative of each equivalence class)."""
    if not duplicates:
        return set()
    pair_indices = {idx for pair in duplicates for idx in pair if idx is not None}
    if not pair_indices:
        return set()
    uf = UnionFind(max(pair_indices) + 1)
    for i, j in duplicates:
        if i is not None and j is not None:
            uf.union(i, j)
    groups = {}
    for idx in pair_indices:
        root = uf.find(idx)
        groups.setdefault(root, set()).add(idx)
    removed = set()
    for indices in groups.values():
        rep = min(indices)
        removed.update(indices - {rep})
    return removed


def compute_metrics(details, duplicates, y_test, n_test, conformal_mask=None):
    """Compute (safety_recall, storage_savings_pct) for one threshold combo.

    If conformal_mask is provided, samples flagged True by it are forcibly
    preserved (Layer 0). They are also excluded from the set of indices that
    can be removed by L1 deduplication — even if FAISS pairs them with a
    duplicate, the conformal-preserved one stays.
    """
    removed = compute_removed_indices(duplicates)

    # Conformal Layer 0 protection: never remove a conformal-preserved sample
    if conformal_mask is not None:
        conformal_preserved_indices = set(np.where(conformal_mask)[0])
        removed = removed - conformal_preserved_indices

    preserved_by_levels = details["level_2_mask"] | details["level_3_mask"]
    if conformal_mask is not None:
        preserved_by_levels = preserved_by_levels | conformal_mask

    non_removed = ~np.isin(np.arange(n_test), list(removed))
    all_preserved = preserved_by_levels | non_removed

    y_bool = (y_test == 1)
    n_critical = int(y_bool.sum())
    preserved_critical = int((all_preserved & y_bool).sum())
    recall = preserved_critical / n_critical if n_critical > 0 else 1.0

    n_removed = int(len(removed))
    savings = max(0.0, min(100.0, (n_removed / n_test * 100.0))) if n_test > 0 else 0.0
    return float(recall), float(savings), int(len(duplicates)), n_removed


# ----------------------------------------------------------------------------
# Train framework once per seed and return the artefacts needed for sweep
# ----------------------------------------------------------------------------

def train_for_seed(seed, sample_size, config, label_type="gt_attack",
                   split_mode: str = "stratified"):
    """Load SWaT, split 4-way (train/val/cal/test) using stratified random
    (default) or episode-level mode. 43 features + binary attack labels."""
    logger.info(f"[seed={seed}] loading SWaT data (split_mode={split_mode})")
    set_seeds(seed)

    df, safety_labels, raw_df = load_swat(
        sample_size=sample_size if sample_size and sample_size < 15000 else None,
        random_state=seed,
    )
    safety_labels = safety_labels.astype(int)

    cfg = config.copy()
    cfg["model"] = config["model"].copy()
    cfg["model"]["input_dim"] = df.shape[1]

    framework = UncertaintyAwareFramework(cfg)

    masks = stratified_split(
        labels=safety_labels,
        seed=seed,
        episode_ids=raw_df["file_id"].to_numpy(),
        mode=split_mode,
        ratios=(0.70, 0.10, 0.10, 0.10),
    )
    train_mask, val_mask, cal_mask, test_mask = (
        masks["train"], masks["val"], masks["cal"], masks["test"]
    )

    df_train = df.iloc[train_mask].reset_index(drop=True)
    df_test = df.iloc[test_mask].reset_index(drop=True)
    X_train = framework.scaler.fit_transform(df_train.values)
    X_val = framework.scaler.transform(df.iloc[val_mask].values)
    X_cal = framework.scaler.transform(df.iloc[cal_mask].values)
    X_test = framework.scaler.transform(df_test.values)
    y_train = safety_labels[train_mask]
    y_val = safety_labels[val_mask]
    y_cal = safety_labels[cal_mask]
    y_test = safety_labels[test_mask]

    logger.info(f"[seed={seed}] split sizes: train={len(X_train)} val={len(X_val)} "
                f"cal={len(X_cal)} test={len(X_test)} critical_in_test={int((y_test == 1).sum())}")

    logger.info(f"[seed={seed}] training framework")
    framework.train(X_train, y_train, X_val, y_val)

    # Causal discovery on training set only (no leakage)
    logger.info(f"[seed={seed}] causal discovery on training set")
    framework.causal_discovery.discover_causal_structure(df_train, y_train)

    # Uncertainty on test set
    logger.info(f"[seed={seed}] computing test uncertainty")
    u_test = framework.predict_with_uncertainty(X_test)
    if hasattr(u_test, "cpu"):
        u_test = u_test.cpu().numpy()
    u_test = normalize_uncertainty(np.array(u_test))

    return dict(
        framework=framework,
        df_test=df_test,
        X_test=X_test,
        y_test=y_test,
        u_test=u_test,
        X_cal=X_cal,
        y_cal=y_cal,
        X_val=X_val,
        y_val=y_val,
        X_train=X_train,
        y_train=y_train,
    )


# ----------------------------------------------------------------------------
# Pareto frontier + knee analysis
# ----------------------------------------------------------------------------

def pareto_front(points):
    """Return the upper-right Pareto frontier of (recall, savings) points
    (we want to MAXIMIZE both)."""
    pts = sorted(set(points), key=lambda p: (-p[0], -p[1]))  # descending recall
    front = []
    best_savings = -np.inf
    for r, s in pts:
        if s > best_savings:
            front.append((r, s))
            best_savings = s
    front.sort()  # ascending recall for plotting
    return front


def savings_at_recall(front, target_recall):
    """Linearly interpolate the Pareto frontier to find max savings achievable
    at recall >= target. front is sorted by recall ascending."""
    # Filter to points meeting recall target
    feasible = [(r, s) for r, s in front if r >= target_recall]
    if not feasible:
        return None
    # The Pareto frontier is monotone — at higher recall the savings drop.
    # Max feasible savings = savings at the lowest-recall feasible point.
    return max(s for r, s in feasible)


# ----------------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------------

def run_sweep(args):
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Sweep grids
    # NOTE: preliminary runs showed tau_high and theta have minimal
    # effect on the recall/savings trade-off in the current architecture; the
    # real lever is tau_low (which controls L1 set size). Path B introduces
    # the conformal Layer 0 (target_recall sweep) as a *true* knob with
    # finite-sample guarantees on fault preservation.
    if args.quick:
        tau_low_grid = [0.05, 0.15, 0.30]
        tau_high_grid = [0.15, 0.50]
        theta_grid = [0.85]
        conformal_recall_grid = [None, 0.90, 0.95, 0.99]
        coreset_budget_grid = [None, 0.7, 0.5, 0.3]
        # Mode B (budget-first): top-K by score (no strict recall guarantee).
        # priority_fraction = how much of test to unconditionally keep as anomaly priority
        priority_fraction_grid = [0.20, 0.30, 0.40, 0.50]
        # budget_fraction = total coreset size (must be >= priority_fraction)
        budget_fraction_grid = [0.30, 0.50, 0.70]
        seeds = list(range(42, 42 + 2))
    else:
        tau_low_grid = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        tau_high_grid = [0.15, 0.50]
        theta_grid = [0.85]
        conformal_recall_grid = [None, 0.90, 0.95, 0.99]
        coreset_budget_grid = [None, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.10, 0.05]
        priority_fraction_grid = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
        budget_fraction_grid = [0.25, 0.40, 0.55, 0.70, 0.85, 0.90, 0.95]
        seeds = list(range(42, 42 + args.seeds))

    # Filter: tau_high must be >= tau_low (otherwise L2 mask collapses)
    base_combos = [
        (tl, th, theta)
        for tl, th, theta in product(tau_low_grid, tau_high_grid, theta_grid)
        if th >= tl
    ]
    # Each base combo is evaluated once per conformal recall target. Conformal
    # gate fitting is cheap (just a quantile); only the dedup call is the
    # expensive part, so we share the FAISS pass across conformal alphas.
    n_evals = len(base_combos) * len(conformal_recall_grid)
    logger.info(f"sweep grid: {len(base_combos)} threshold combos x "
                f"{len(conformal_recall_grid)} conformal targets x "
                f"{len(seeds)} seeds = {n_evals * len(seeds)} evaluations "
                f"({n_evals} per seed)")

    out_dir = ROOT / "results" / "pareto_swat"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pareto_sweep_swat.csv"

    rows = []
    for seed in seeds:
        try:
            artefacts = train_for_seed(seed, args.sample_size, config, args.label_type,
                                       split_mode=args.split_mode)
        except Exception as e:
            logger.error(f"[seed={seed}] FAILED to train: {e}")
            import traceback; traceback.print_exc()
            continue

        framework = artefacts["framework"]
        df_test = artefacts["df_test"]
        X_test = artefacts["X_test"]
        y_test = artefacts["y_test"]
        u_test = artefacts["u_test"]
        X_train = artefacts.get("X_train")  # may be None on legacy path; recomputed if needed
        y_train = artefacts.get("y_train")
        X_val = artefacts["X_val"]
        y_val = artefacts["y_val"]
        n_test = len(df_test)

        X_cal = artefacts["X_cal"]
        y_cal = artefacts["y_cal"]

        # ---- Conformal Layer 0 (Path B+) — multi-scorer setup ----
        # Score on the HELD-OUT calibration set + test set.
        scorers = build_default_scorers()
        scored_cache = {}  # name -> (scores_cal, scores_test)
        for name, scorer in scorers.items():
            try:
                if X_train is None:
                    scorer.fit(framework=framework, seed=int(seed))
                else:
                    scorer.fit(X_train, y_train, framework=framework, seed=int(seed))
                s_cal = np.asarray(scorer.score(X_cal)).flatten().astype(float)
                s_test = np.asarray(scorer.score(X_test)).flatten().astype(float)
                scored_cache[name] = (s_cal, s_test)
                logger.info(f"[seed={seed}] scorer {name}: fitted; "
                            f"cal_mean={s_cal.mean():.4f} test_mean={s_test.mean():.4f}")
            except Exception as e:
                logger.warning(f"[seed={seed}] scorer {name} FAILED: {e}")
                scored_cache[name] = None

        # Conformal gates calibrated on the HELD-OUT cal set
        conformal_masks = {}
        for name, scored in scored_cache.items():
            if scored is None:
                continue
            s_cal, s_test = scored
            def _cal_scorer(_X, _s=s_cal): return _s
            def _test_scorer(_X, _s=s_test): return _s
            for target_recall in conformal_recall_grid:
                if target_recall is None:
                    conformal_masks[(name, None)] = None
                    continue
                try:
                    gate = ConformalAnomalyGate(target_recall=target_recall).fit(
                        X_cal, y_cal, _cal_scorer
                    )
                    mask = gate.preserve_mask(X_test, _test_scorer)
                    conformal_masks[(name, target_recall)] = mask
                    layer0_recall = float(mask[y_test == 1].mean()) if (y_test == 1).any() else 0.0
                    logger.info(f"[seed={seed}] gate ({name}, target={target_recall:.2f}): "
                                f"tau={gate.tau:.4f} preserves {mask.mean()*100:.1f}% of test, "
                                f"Layer-0-only test-recall={layer0_recall:.3f}")
                except Exception as e:
                    logger.warning(f"[seed={seed}] gate ({name},{target_recall}) FAILED: {e}")
                    conformal_masks[(name, target_recall)] = None

        # Also include the (no-conformal) baseline by treating "off" as a scorer
        # name with mask=None. This keeps all rows in one CSV.
        for tr in conformal_recall_grid:
            if tr is None:
                conformal_masks[("off", None)] = None

        n_evals_per_combo = sum(
            1 for (_n, _t) in conformal_masks
            if (_n == "off" and _t is None) or (_n != "off" and _t is not None)
        )
        logger.info(f"[seed={seed}] sweeping {len(base_combos)} threshold combos x "
                    f"{n_evals_per_combo} (scorer, conformal_target) pairs "
                    f"= {len(base_combos) * n_evals_per_combo} evaluations")

        for k, (tau_low, tau_high, theta) in enumerate(base_combos, 1):
            # One expensive FAISS call per threshold combo; reused across all gates.
            duplicates, _preserved_count, details = framework.find_duplicates_multi_level(
                df_test, X_test, u_test,
                low_threshold=tau_low,
                high_threshold=tau_high,
                similarity_threshold=theta,
                return_details=True,
            )
            # No-conformal row (one per combo)
            recall, savings, n_dup_pairs, n_removed = compute_metrics(
                details, duplicates, y_test, n_test, conformal_mask=None
            )
            rows.append(dict(
                seed=seed, method="conformal_only",
                tau_low=tau_low, tau_high=tau_high, theta=theta,
                scorer="off", conformal_target_recall="off", coreset_budget="off",
                safety_recall=recall, storage_savings_pct=savings,
                duplicate_pairs=n_dup_pairs, samples_removed=n_removed,
                n_test=n_test, level0_count=0,
                level1_count=int(details["level_1_mask"].sum()),
                level2_count=int(details["level_2_mask"].sum()),
                level3_count=int(details["level_3_mask"].sum()),
            ))
            # Per-scorer x per-conformal-target rows (still Path B)
            for (sname, tr), cmask in conformal_masks.items():
                if sname == "off" or tr is None:
                    continue
                if cmask is None:
                    continue
                recall, savings, n_dup_pairs, n_removed = compute_metrics(
                    details, duplicates, y_test, n_test, conformal_mask=cmask
                )
                rows.append(dict(
                    seed=seed, method="conformal_only",
                    tau_low=tau_low, tau_high=tau_high, theta=theta,
                    scorer=sname, conformal_target_recall=tr, coreset_budget="off",
                    safety_recall=recall, storage_savings_pct=savings,
                    duplicate_pairs=n_dup_pairs, samples_removed=n_removed,
                    n_test=n_test, level0_count=int(cmask.sum()),
                    level1_count=int(details["level_1_mask"].sum()),
                    level2_count=int(details["level_2_mask"].sum()),
                    level3_count=int(details["level_3_mask"].sum()),
                ))
            if k % 5 == 0 or k == len(base_combos):
                logger.info(f"[seed={seed}]   combo {k}/{len(base_combos)}: "
                            f"tl={tau_low} th={tau_high} theta={theta} done "
                            f"({len(rows)} total rows so far)")

        # ====================================================================
        # CAP-Dedup (Path A+B): two-stage Conformal + Submodular Coreset
        # Independent of (tau_low, tau_high, theta) - operates directly on
        # conformal masks + Siamese embeddings. Computed once per seed.
        # ====================================================================
        logger.info(f"[seed={seed}] CAP-Dedup: building Siamese embeddings for coreset")
        siamese_emb = framework.get_embeddings(X_test, use_siamese=True)
        # Compute anomaly scores per scorer for coreset's anomaly-priority logic
        n_capdedup_rows = 0
        for (sname, tr), cmask in conformal_masks.items():
            if sname == "off" or tr is None or cmask is None:
                continue
            if scored_cache.get(sname) is None:
                continue
            _s_val, s_test_arr = scored_cache[sname]
            must_preserve = cmask.copy()
            n_must = int(must_preserve.sum())
            for budget_frac in coreset_budget_grid:
                if budget_frac is None:
                    continue  # the "no coreset" baseline is already in Path B rows above
                budget = max(n_must, int(round(budget_frac * n_test)))
                cs = CoverageCoreset(seed=int(seed))
                keep_mask = cs.select(siamese_emb, s_test_arr, must_preserve, budget)
                y_bool = (y_test == 1)
                n_critical = int(y_bool.sum())
                preserved_critical = int((keep_mask & y_bool).sum())
                cap_recall = preserved_critical / n_critical if n_critical > 0 else 1.0
                cap_savings = (1.0 - keep_mask.sum() / n_test) * 100.0
                rows.append(dict(
                    seed=seed, method="cap_dedup",
                    tau_low="n/a", tau_high="n/a", theta="n/a",
                    scorer=sname, conformal_target_recall=tr,
                    coreset_budget=budget_frac,
                    safety_recall=float(cap_recall),
                    storage_savings_pct=float(cap_savings),
                    duplicate_pairs=0, samples_removed=int(n_test - keep_mask.sum()),
                    n_test=n_test, level0_count=int(n_must),
                    level1_count=0, level2_count=0, level3_count=0,
                ))
                n_capdedup_rows += 1
        logger.info(f"[seed={seed}] CAP-Dedup-strict: {n_capdedup_rows} rows generated "
                    f"({len(conformal_recall_grid)-1} targets x "
                    f"{len([b for b in coreset_budget_grid if b is not None])} budgets x "
                    f"{sum(1 for n in scored_cache if scored_cache[n] is not None)} scorers)")

        # ====================================================================
        # CAP-Dedup-Budget (Mode B): drop strict conformal guarantee; use
        # direct priority-fraction + budget control. Empirical recall reported.
        # Two-knob trade-off curve:
        #   priority_fraction (P): top-(P * n_test) by score, unconditionally kept
        #   budget_fraction   (B): total coreset size, with budget >= P * n_test
        # Coreset selects (B - P) additional samples via facility-location.
        # ====================================================================
        n_capdedup_budget_rows = 0
        for sname, scored in scored_cache.items():
            if scored is None:
                continue
            _s_val, s_test_arr = scored
            order_by_score_desc = np.argsort(-s_test_arr)
            for priority_frac in priority_fraction_grid:
                # Build priority mask: top-(priority_frac * n_test) by anomaly score
                k_priority = int(round(priority_frac * n_test))
                priority_mask = np.zeros(n_test, dtype=bool)
                priority_mask[order_by_score_desc[:k_priority]] = True
                for budget_frac in budget_fraction_grid:
                    if budget_frac < priority_frac:
                        continue  # infeasible
                    budget = max(k_priority, int(round(budget_frac * n_test)))
                    cs = CoverageCoreset(seed=int(seed))
                    keep_mask = cs.select(siamese_emb, s_test_arr, priority_mask, budget)
                    y_bool = (y_test == 1)
                    n_critical = int(y_bool.sum())
                    preserved_critical = int((keep_mask & y_bool).sum())
                    bd_recall = preserved_critical / n_critical if n_critical > 0 else 1.0
                    bd_savings = (1.0 - keep_mask.sum() / n_test) * 100.0
                    rows.append(dict(
                        seed=seed, method="cap_dedup_budget",
                        tau_low="n/a", tau_high="n/a", theta="n/a",
                        scorer=sname,
                        conformal_target_recall=f"priority={priority_frac}",
                        coreset_budget=budget_frac,
                        safety_recall=float(bd_recall),
                        storage_savings_pct=float(bd_savings),
                        duplicate_pairs=0,
                        samples_removed=int(n_test - keep_mask.sum()),
                        n_test=n_test, level0_count=int(k_priority),
                        level1_count=0, level2_count=0, level3_count=0,
                    ))
                    n_capdedup_budget_rows += 1
        logger.info(f"[seed={seed}] CAP-Dedup-budget: {n_capdedup_budget_rows} rows generated")

        # ====================================================================
        # Literature baselines at matched budgets
        # ====================================================================
        n_baseline_rows = 0
        ecod_test_scores = (scored_cache["ecod"][1]
                            if scored_cache.get("ecod") is not None else u_test)
        for budget_frac in budget_fraction_grid:
            budget = int(round(budget_frac * n_test))
            try:
                baseline_masks = build_baseline_keep_masks(
                    embeddings=siamese_emb,
                    anomaly_scores=ecod_test_scores,
                    budget=budget,
                    seed=int(seed),
                )
            except Exception as e:
                logger.warning(f"[seed={seed}] baseline build failed at budget={budget}: {e}")
                continue
            for bname, keep_mask in baseline_masks.items():
                y_bool = (y_test == 1)
                n_critical = int(y_bool.sum())
                preserved_critical = int((keep_mask & y_bool).sum())
                rec = preserved_critical / n_critical if n_critical > 0 else 1.0
                sav = (1.0 - keep_mask.sum() / n_test) * 100.0
                rows.append(dict(
                    seed=seed, method=f"baseline_{bname}",
                    tau_low="n/a", tau_high="n/a", theta="n/a",
                    scorer="n/a", conformal_target_recall="n/a",
                    coreset_budget=budget_frac,
                    safety_recall=float(rec), storage_savings_pct=float(sav),
                    duplicate_pairs=0,
                    samples_removed=int(n_test - keep_mask.sum()),
                    n_test=n_test, level0_count=0,
                    level1_count=0, level2_count=0, level3_count=0,
                ))
                n_baseline_rows += 1
        logger.info(f"[seed={seed}] literature baselines: {n_baseline_rows} rows generated")

        # Incremental save after each seed
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"[seed={seed}] checkpoint: {len(rows)} rows -> {csv_path}")

        del artefacts, framework
        gc.collect()

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(csv_path, index=False)
    logger.info(f"FULL sweep saved: {csv_path} ({len(df_rows)} rows)")

    # ------------------------------------------------------------------------
    # Knee analysis
    # ------------------------------------------------------------------------
    # Compute the knee TWO ways:
    #   (a) "overall" : best savings at >=R% recall across ALL operating points
    #                   (including conformal alphas) - what you'd actually deploy
    #   (b) "by_conformal_target" : per conformal alpha (and "off"), what's the
    #                   knee on that subset alone. Lets us see what conformal adds.
    knee_targets = [0.90, 0.95, 0.99]

    def knee_block(subset_df, label):
        """Compute per-seed knee for the given subset of rows."""
        out = {"label": label, "n_seeds_total": int(subset_df["seed"].nunique())}
        for t in knee_targets:
            per_seed = []
            for seed in subset_df["seed"].unique():
                seed_df = subset_df[subset_df["seed"] == seed]
                pts = list(zip(seed_df["safety_recall"].tolist(),
                               seed_df["storage_savings_pct"].tolist()))
                front = pareto_front(pts)
                s = savings_at_recall(front, t)
                if s is not None:
                    per_seed.append(s)
            key = f"savings_at_recall_{int(t*100)}pct"
            if per_seed:
                out[key] = {
                    "mean": float(np.mean(per_seed)),
                    "std": float(np.std(per_seed)),
                    "min": float(np.min(per_seed)),
                    "max": float(np.max(per_seed)),
                    "n_seeds_feasible": len(per_seed),
                }
            else:
                out[key] = {"mean": None, "note": "no feasible point on any seed's frontier"}
        return out

    knee_report = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sample_size": args.sample_size,
            "label_type": args.label_type,
            "n_seeds": int(df_rows["seed"].nunique()),
            "n_base_combos": len(base_combos),
            "n_conformal_targets": len(conformal_recall_grid),
            "n_points_total": int(len(df_rows)),
            "scorers": sorted(df_rows["scorer"].dropna().unique().tolist()),
        },
        "overall": knee_block(df_rows, "overall (all operating points)"),
        "by_scorer": {},
        "by_scorer_x_conformal": {},
    }
    for sname in df_rows["scorer"].unique():
        sub = df_rows[df_rows["scorer"] == sname]
        knee_report["by_scorer"][str(sname)] = knee_block(sub, f"scorer={sname}")
    for sname in df_rows["scorer"].unique():
        for ct in df_rows["conformal_target_recall"].unique():
            sub = df_rows[(df_rows["scorer"] == sname) & (df_rows["conformal_target_recall"] == ct)]
            if len(sub) == 0:
                continue
            key = f"{sname}__ct={ct}"
            knee_report["by_scorer_x_conformal"][key] = knee_block(sub, key)

    knee_path = out_dir / "knee_report.json"
    with open(knee_path, "w", encoding="utf-8") as f:
        json.dump(knee_report, f, indent=2)
    logger.info(f"knee report saved: {knee_path}")

    # ------------------------------------------------------------------------
    # Pareto plot (matplotlib)
    # ------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Color-code by scorer (more informative than conformal target in multi-scorer mode)
        scorer_colors = {
            "off":             "lightgray",
            "bnn_mean":        "#1f77b4",
            "bnn_variance":    "#17becf",
            "bnn_combined":    "#9467bd",
            "isolation_forest":"#d62728",
            "autoencoder":     "#2ca02c",
        }
        fig, ax = plt.subplots(figsize=(12, 7))

        for sname in df_rows["scorer"].unique():
            sub = df_rows[df_rows["scorer"] == sname]
            label = f"{sname} (n={len(sub)})"
            color = scorer_colors.get(str(sname), None)
            ax.scatter(sub["safety_recall"] * 100, sub["storage_savings_pct"],
                       s=18, c=color, alpha=0.55, label=label, edgecolors="none")

        # Aggregate Pareto frontier (best of all operating points)
        all_pts = list(zip(df_rows["safety_recall"].tolist(),
                           df_rows["storage_savings_pct"].tolist()))
        agg_front = pareto_front(all_pts)
        if agg_front:
            xs = [r * 100 for r, _ in agg_front]
            ys = [s for _, s in agg_front]
            ax.plot(xs, ys, "k-", linewidth=2.5, label="overall Pareto frontier")

        # Knee vertical lines + annotations
        for t in knee_targets:
            ax.axvline(t * 100, color="red", linestyle=":", alpha=0.4)
            s_info = knee_report["overall"][f"savings_at_recall_{int(t*100)}pct"]
            if s_info.get("mean") is not None:
                ax.annotate(
                    f">={int(t*100)}% recall:\n{s_info['mean']:.1f}+/-{s_info['std']:.1f}%",
                    xy=(t * 100, s_info["mean"]),
                    xytext=(t * 100 + 0.8, s_info["mean"] + 5),
                    fontsize=9, color="red",
                )

        ax.set_xlabel("Fault Recall (%)", fontsize=12)
        ax.set_ylabel("Storage Savings (%)", fontsize=12)
        ax.set_title(f"CAP-Dedup: Pareto Frontier on SWaT "
                     f"({df_rows['seed'].nunique()} seeds, "
                     f"label={args.label_type})", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_xlim(0, 105)
        max_sav = float(df_rows["storage_savings_pct"].max()) if len(df_rows) else 50
        ax.set_ylim(-2, max(max_sav * 1.1, 50))

        plot_path = out_dir / "pareto_frontier_swat.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        logger.info(f"plot saved: {plot_path}")
    except Exception as e:
        logger.warning(f"plot failed (matplotlib not available?): {e}")

    # ------------------------------------------------------------------------
    # Console summary (ASCII only - Windows cp1252 cannot encode unicode >=, etc.)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PARETO SWEEP - KNEE REPORT (SWaT, label_type=" + args.label_type + ")")
    print("=" * 78)
    print(f"Seeds: {df_rows['seed'].nunique()}  |  Base threshold combos: {len(base_combos)}  |  "
          f"Conformal targets: {len(conformal_recall_grid)}  |  "
          f"Total points: {len(df_rows)}")
    print()
    print("[overall - best across ALL operating points incl. conformal]")
    for t in knee_targets:
        s_info = knee_report["overall"][f"savings_at_recall_{int(t*100)}pct"]
        if s_info.get("mean") is not None:
            print(f"  Storage savings @ >={int(t*100)}% recall : "
                  f"{s_info['mean']:6.2f}% +/- {s_info['std']:.2f}%   "
                  f"(range {s_info['min']:.1f}-{s_info['max']:.1f}, "
                  f"feasible on {s_info['n_seeds_feasible']}/{df_rows['seed'].nunique()} seeds)")
        else:
            print(f"  Storage savings @ >={int(t*100)}% recall : "
                  f"NOT FEASIBLE on any seed (frontier never reaches this recall)")
    print()
    print("[per-scorer breakdown — best savings at each recall floor]")
    for sname in sorted(df_rows["scorer"].unique()):
        block = knee_report["by_scorer"][str(sname)]
        line_parts = [f"  scorer={sname:<18}"]
        for t in knee_targets:
            s_info = block[f"savings_at_recall_{int(t*100)}pct"]
            if s_info.get("mean") is not None:
                line_parts.append(f"@{int(t*100)}%R: {s_info['mean']:5.1f}%")
            else:
                line_parts.append(f"@{int(t*100)}%R:   n/a")
        print(" | ".join(line_parts))
    print()
    print("[scorer x conformal_target - savings @ 95% recall]")
    for sname in sorted(df_rows["scorer"].unique()):
        # Sort conformal_target_recall as strings to avoid mixed-type comparison
        ct_vals = df_rows[df_rows["scorer"] == sname]["conformal_target_recall"].unique()
        ct_vals = sorted(ct_vals, key=lambda x: (x == "off", str(x)))
        for ct in ct_vals:
            key = f"{sname}__ct={ct}"
            if key not in knee_report["by_scorer_x_conformal"]:
                continue
            s_info = knee_report["by_scorer_x_conformal"][key].get("savings_at_recall_95pct", {})
            tag = f"  {sname:<18} ct={str(ct):<5}"
            if s_info.get("mean") is not None:
                print(f"{tag} -> savings@95%R: {s_info['mean']:5.1f}% +/- {s_info['std']:.1f}%")
            else:
                print(f"{tag} -> savings@95%R:   n/a")
    print()
    print(f"CSV:   {csv_path}")
    print(f"JSON:  {knee_path}")
    print(f"Plot:  {out_dir / 'pareto_frontier_swat.png'}")
    print("=" * 78)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="2 seeds + 3x3x3 grid (~5 min smoke test)")
    p.add_argument("--seeds", type=int, default=10,
                   help="number of seeds (default 10; ignored when --quick)")
    p.add_argument("--sample-size", type=int, default=None,
                   help="SWaT sample size (default: use full ~15k rows)")
    p.add_argument("--label-type", default="gt_attack",
                   choices=["gt_attack"],
                   help="label scheme (only gt_attack available for SWaT)")
    p.add_argument("--split-mode", default="stratified",
                   choices=["stratified", "episode"],
                   help="Data split mode (default stratified, recommended)")
    args = p.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
