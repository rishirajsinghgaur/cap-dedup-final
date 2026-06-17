#!/usr/bin/env python3
"""
Rigorous data splitting for CAP-Dedup sweeps.

The original CAP-Dedup paper splits by `simulationRun` (TEP) or `file_id`
(SKAB/SWaT) to avoid temporal information leakage. This works for TEP (where
runs have similar sizes ~100 samples each) but fails for SKAB and SWaT (where
episode sizes range from ~100 to ~13,000 rows). Result: per-seed test set
composition varies wildly, inflating result variance.

This module provides a unified splitter with two modes:

  - STRATIFIED-RANDOM (`mode="stratified"`):
        Random sample-level split stratified by anomaly label. Class
        distributions are balanced across splits. Anomaly rate in train/val/
        cal/test is approximately constant per-seed. This is the recommended
        DEFAULT for SKAB and SWaT.
        Caveat: same-episode samples may appear in both train and test.
        For our application (per-sample classification + per-sample
        preservation), this is acceptable because the BNN/Siamese models
        have no temporal memory.

  - EPISODE-LEVEL (`mode="episode"`):
        Original split-by-episode behaviour. Preserves strict temporal
        independence between splits. Useful as an ablation/sanity check.

Both modes also produce a SEPARATE calibration set (split off from validation)
so the conformal gate's calibration data is never used to make a training
decision (currently, val is used for both BNN early-stopping AND conformal
calibration — a mild exchangeability violation acknowledged in the audit).

USAGE:
    from data_splitter import stratified_split
    masks = stratified_split(
        labels=safety_labels,
        seed=seed,
        episode_ids=raw_df["file_id"].to_numpy(),
        mode="stratified",
        ratios=(0.70, 0.10, 0.10, 0.10),
    )
    train_mask, val_mask, cal_mask, test_mask = (
        masks["train"], masks["val"], masks["cal"], masks["test"]
    )
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("data_splitter")


def stratified_split(
    labels: np.ndarray,
    seed: int,
    episode_ids: Optional[np.ndarray] = None,
    mode: str = "stratified",
    ratios: Tuple[float, float, float, float] = (0.70, 0.10, 0.10, 0.10),
) -> Dict[str, np.ndarray]:
    """Split n samples into train/val/cal/test sets.

    Parameters
    ----------
    labels : (n,) int
        Anomaly labels (0/1). Used for stratification in 'stratified' mode.
    seed : int
        Per-experiment seed.
    episode_ids : (n,) optional
        Episode/file identifier per sample (e.g. simulationRun for TEP).
        Required for mode='episode'.
    mode : "stratified" | "episode"
        Splitting strategy. Default 'stratified' (recommended for SKAB/SWaT).
    ratios : 4-tuple (train, val, cal, test)
        Fractions, must sum to 1.0.

    Returns
    -------
    dict with boolean masks: {"train": (n,), "val": (n,), "cal": (n,), "test": (n,)}
    """
    if not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError(f"ratios must sum to 1.0, got {ratios} -> {sum(ratios)}")
    n = len(labels)
    labels = np.asarray(labels).astype(int)

    if mode == "stratified":
        return _stratified_random(labels, seed, ratios, n)
    elif mode == "episode":
        if episode_ids is None:
            raise ValueError("mode='episode' requires episode_ids")
        return _episode_split(labels, seed, episode_ids, ratios, n)
    else:
        raise ValueError(f"unknown mode: {mode}")


def _stratified_random(labels, seed, ratios, n):
    """Stratified random split by label."""
    rng = np.random.default_rng(seed)
    masks = {k: np.zeros(n, dtype=bool) for k in ("train", "val", "cal", "test")}

    # For each class, shuffle its indices and assign to splits in proportion
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0].copy()
        rng.shuffle(idx)
        n_cls = len(idx)
        # Cut points: cumulative ratios * n_cls
        n_train = int(round(ratios[0] * n_cls))
        n_val = int(round(ratios[1] * n_cls))
        n_cal = int(round(ratios[2] * n_cls))
        n_test = n_cls - n_train - n_val - n_cal  # absorbs rounding
        if n_test < 0:
            # Rare: rounding overshoots. Shave from train.
            n_train += n_test
            n_test = 0
        ends = np.cumsum([n_train, n_val, n_cal])
        masks["train"][idx[: ends[0]]] = True
        masks["val"][idx[ends[0]: ends[1]]] = True
        masks["cal"][idx[ends[1]: ends[2]]] = True
        masks["test"][idx[ends[2]:]] = True

    # Sanity check: no overlap, total = n
    overlap = sum(masks[k] for k in masks)
    assert (overlap == 1).all(), \
        f"Split overlap or gap detected; coverage: {overlap.min()}-{overlap.max()}"

    _log_split_diagnostics("stratified", seed, masks, labels, n)
    return masks


def _episode_split(labels, seed, episode_ids, ratios, n):
    """Episode-level split: assign whole episodes to one split each.

    Episodes are shuffled then assigned by cumulative row-fraction. This handles
    uneven episode sizes better than naive count-based assignment by stopping
    each split when its target fraction is reached.
    """
    rng = np.random.default_rng(seed)
    episode_ids = np.asarray(episode_ids)
    masks = {k: np.zeros(n, dtype=bool) for k in ("train", "val", "cal", "test")}

    unique_eps = np.unique(episode_ids).copy()
    rng.shuffle(unique_eps)

    cumulative = 0
    targets = [int(round(r * n)) for r in ratios]
    # Ensure they sum to n
    diff = n - sum(targets)
    targets[0] += diff

    split_order = ["train", "val", "cal", "test"]
    split_idx = 0

    for ep in unique_eps:
        ep_mask = (episode_ids == ep)
        ep_size = int(ep_mask.sum())
        if split_idx < len(split_order) - 1 and cumulative + ep_size > sum(targets[: split_idx + 1]):
            # Move to next split if adding this episode would significantly overshoot
            overshoot = (cumulative + ep_size) - sum(targets[: split_idx + 1])
            undershoot = sum(targets[: split_idx + 1]) - cumulative
            if overshoot > undershoot:
                split_idx += 1
        masks[split_order[split_idx]][ep_mask] = True
        cumulative += ep_size

    _log_split_diagnostics("episode", seed, masks, labels, n)
    return masks


def _log_split_diagnostics(mode_name, seed, masks, labels, n):
    """Log per-split size + anomaly RATE WITHIN the split (not 'fraction of anomalies')."""
    parts = []
    for k in ("train", "val", "cal", "test"):
        m = masks[k]
        n_k = int(m.sum())
        rate = float(labels[m].mean() * 100) if n_k > 0 else 0.0
        parts.append(f"{k}={n_k}(anom_rate={rate:.1f}%)")
    logger.info(f"[{mode_name}] seed={seed} n={n}: " + " | ".join(parts))


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(0)
    n = 1000
    labels = rng.binomial(1, 0.3, size=n)  # 30% anomaly rate
    episodes = rng.integers(0, 20, size=n)  # 20 episodes

    print(f"Total: {n} samples, {labels.sum()} anomalies ({100*labels.mean():.1f}%)")
    print()
    print("--- Stratified random split ---")
    for s in [42, 43, 44]:
        m = stratified_split(labels, s, episode_ids=episodes, mode="stratified")
    print()
    print("--- Episode-level split ---")
    for s in [42, 43, 44]:
        m = stratified_split(labels, s, episode_ids=episodes, mode="episode")
