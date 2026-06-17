#!/usr/bin/env python3
"""Coreset and sampling baselines: uniform random, reservoir, stratified-by-score, k-center, and facility-location (eager and memory-bounded lazy variants)."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("literature_baselines")


def _normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def random_uniform(n_total: int, budget: int, seed: int) -> np.ndarray:
    """Random uniform subset of `budget` indices. Returns a (n_total,) boolean mask."""
    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(n_total, size=min(budget, n_total), replace=False)
    mask = np.zeros(n_total, dtype=bool)
    mask[keep_idx] = True
    return mask


def reservoir_sample(n_total: int, budget: int, seed: int) -> np.ndarray:
    """Vitter's algorithm-R reservoir sampling (single-pass, streaming-friendly).
    Statistically equivalent to random_uniform but represents the natural
    streaming-IIoT baseline."""
    rng = np.random.default_rng(seed)
    if budget >= n_total:
        return np.ones(n_total, dtype=bool)
    reservoir = list(range(budget))
    for i in range(budget, n_total):
        j = rng.integers(0, i + 1)
        if j < budget:
            reservoir[j] = i
    mask = np.zeros(n_total, dtype=bool)
    mask[reservoir] = True
    return mask


def stratified_by_score(scores: np.ndarray, budget: int) -> np.ndarray:
    """Top-K by anomaly score. Deterministic given scores."""
    n = len(scores)
    if budget >= n:
        return np.ones(n, dtype=bool)
    top_idx = np.argsort(-scores)[:budget]
    mask = np.zeros(n, dtype=bool)
    mask[top_idx] = True
    return mask


def kcenter_no_conformal(embeddings: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """k-center greedy on the FULL test set without any must-preserve set.

    Isolates the contribution of conformal+priority: this is what you'd get
    by applying our submodular selector without the recall guarantee.
    """
    n = len(embeddings)
    if budget >= n:
        return np.ones(n, dtype=bool)
    E = _normalize(embeddings.astype(np.float32))
    rng = np.random.default_rng(seed)
    # Seed: start with a random sample
    first = int(rng.integers(0, n))
    keep = np.zeros(n, dtype=bool)
    keep[first] = True
    max_sim = (E @ E[first]).astype(np.float64)
    max_sim[first] = np.inf  # exclude
    for _ in range(budget - 1):
        best = int(np.argmin(max_sim))  # farthest from current
        if max_sim[best] == np.inf:
            break
        keep[best] = True
        new_sims = E @ E[best]
        max_sim = np.maximum(max_sim, new_sims)
        max_sim[best] = np.inf
    return keep


def facility_location_priority_weighted(
    embeddings: np.ndarray,
    anomaly_scores: np.ndarray,
    budget: int,
    seed: int,
    weight_temperature: float = 1.0,
    max_iterations: Optional[int] = None,
) -> np.ndarray:
    """Anomaly-priority-weighted facility-location greedy.

    Rationale:
        The natural single-stage alternative to our two-stage construction is
        to fold anomaly priority into the facility-location objective directly,
        so that a single submodular greedy maximises
            sum_j w_j * max_{s in S} sim(s, j)
        with w_j a monotone increasing function of the anomaly score. This is
        the strongest single-stage priority-coreset baseline; if CAP-Dedup
        still dominates it on the recall floor, the two-stage construction is
        defended; if not, the contribution claim must be re-framed.

    Weights:
        w_j = softmax_temperature(z_j) where z_j is the z-scored anomaly score
        and the temperature defaults to 1.0 (so that very anomalous points
        contribute roughly exp(3) ~ 20x more to the coverage objective than
        the median).

    Returns:
        boolean mask of size n with `budget` True entries.

    Complexity:
        O(N^2) memory + O(N^2 * budget) time. Same memory bound as the
        unweighted facility-location baseline.
    """
    n = len(embeddings)
    if budget >= n:
        return np.ones(n, dtype=bool)
    if max_iterations is None:
        max_iterations = budget
    max_iterations = min(max_iterations, budget)

    E = _normalize(embeddings.astype(np.float32))
    S = (E @ E.T).astype(np.float64)

    # Convert anomaly scores to non-negative weights via z-score + softmax-style
    # exponential weighting. Floor the minimum weight at exp(-3) so that normal
    # samples still receive non-zero coverage benefit (otherwise the algorithm
    # degenerates to top-K by score).
    z = (anomaly_scores - np.mean(anomaly_scores)) / (np.std(anomaly_scores) + 1e-12)
    w = np.exp(np.clip(z / max(weight_temperature, 1e-6), -3.0, 3.0))
    # Normalise so the average weight is 1 (numerical stability)
    w = w / max(np.mean(w), 1e-12)

    keep = np.zeros(n, dtype=bool)
    max_sim_to_S = np.full(n, -np.inf, dtype=np.float64)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    keep[first] = True
    max_sim_to_S = np.maximum(max_sim_to_S, S[first])

    for _ in range(max_iterations - 1):
        # Weighted marginal gain
        improvements = np.maximum(0.0, S - max_sim_to_S[None, :])
        weighted_improvements = improvements * w[None, :]
        gains = weighted_improvements.sum(axis=1)
        gains[keep] = -np.inf
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        keep[best] = True
        max_sim_to_S = np.maximum(max_sim_to_S, S[best])

    return keep


def facility_location_greedy(
    embeddings: np.ndarray,
    budget: int,
    seed: int,
    max_iterations: Optional[int] = None,
) -> np.ndarray:
    """Greedy facility-location with sum-coverage objective:
        coverage(S) = sum_j max_{s in S} sim(s, j)
    Gives (1 - 1/e) approximation to the optimal sum-coverage subset of size
    `budget`. SLOWER than k-center (O(n^2) per iter), so we cap iterations.

    Returns boolean mask of size n with `budget` True entries.
    """
    n = len(embeddings)
    if budget >= n:
        return np.ones(n, dtype=bool)
    if max_iterations is None:
        max_iterations = budget
    max_iterations = min(max_iterations, budget)

    E = _normalize(embeddings.astype(np.float32))
    # Precompute full pairwise sim matrix once (memory: n*n*8 bytes; OK up to ~10k)
    S = (E @ E.T).astype(np.float64)
    keep = np.zeros(n, dtype=bool)
    max_sim_to_S = np.full(n, -np.inf, dtype=np.float64)

    rng = np.random.default_rng(seed)
    # Seed with a random sample
    first = int(rng.integers(0, n))
    keep[first] = True
    max_sim_to_S = np.maximum(max_sim_to_S, S[first])

    for _ in range(max_iterations - 1):
        # Marginal gain: sum_j max(0, S[c,j] - max_sim_to_S[j])
        improvements = np.maximum(0.0, S - max_sim_to_S[None, :])
        gains = improvements.sum(axis=1)
        gains[keep] = -np.inf
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        keep[best] = True
        max_sim_to_S = np.maximum(max_sim_to_S, S[best])

    return keep


def lazy_facility_location_greedy(
    embeddings: np.ndarray,
    budget: int,
    seed: int,
    max_iterations: Optional[int] = None,
) -> np.ndarray:
    """Memory-bounded lazy-greedy (CELF) facility-location.

    Identical objective and seeding as facility_location_greedy (random first
    point from the same RNG, sum-coverage greedy), but never materialises the
    full N x N similarity matrix nor an N x N per-iteration temporary: each
    marginal-gain evaluation touches only an N-vector (row of similarities
    computed on the fly as E @ E[j]). By submodularity of sum-coverage, the
    lazy greedy selects the same set as the eager greedy, in O(N*d) memory
    instead of O(N^2). This lets the baseline run on the larger SKAB/SWaT test
    sets (n~4-5k) for large test sets, where the eager O(N^2) variant is memory-bound.

    Tie-break matches the eager np.argmax (lowest index among equal gains):
    heap entries are (-gain, index) so equal gains pop the lowest index first;
    a per-candidate freshness stamp makes the lazy re-evaluation exact.
    """
    import heapq
    n = len(embeddings)
    if budget >= n:
        return np.ones(n, dtype=bool)
    if max_iterations is None:
        max_iterations = budget
    max_iterations = min(max_iterations, budget)

    E = _normalize(embeddings.astype(np.float32))
    keep = np.zeros(n, dtype=bool)
    max_sim_to_S = np.full(n, -np.inf, dtype=np.float64)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    keep[first] = True
    max_sim_to_S = np.maximum(max_sim_to_S, (E @ E[first]).astype(np.float64))

    def compute_gain(j: int) -> float:
        diff = (E @ E[j]).astype(np.float64) - max_sim_to_S
        np.maximum(diff, 0.0, out=diff)
        return float(diff.sum())

    # Seed the heap with every candidate's gain (one O(N*d) pass per candidate;
    # no N x N matrix is ever allocated).
    heap = []  # entries: (-gain, last_iteration, index); index breaks ties low-first
    for j in range(n):
        if keep[j]:
            continue
        heapq.heappush(heap, (-compute_gain(j), 0, j))

    iteration = 1
    while int(keep.sum()) < max_iterations and heap:
        neg_gain, stamp, j = heapq.heappop(heap)
        if keep[j]:
            continue
        if stamp == iteration:
            if -neg_gain <= 0:
                break
            keep[j] = True
            np.maximum(max_sim_to_S, (E @ E[j]).astype(np.float64), out=max_sim_to_S)
            iteration += 1
        else:
            heapq.heappush(heap, (-compute_gain(j), iteration, j))

    return keep


# -----------------------------------------------------------------------------
# Unified entry point for the sweep
# -----------------------------------------------------------------------------

def build_baseline_keep_masks(
    embeddings: np.ndarray,
    anomaly_scores: np.ndarray,
    budget: int,
    seed: int,
) -> dict:
    """Run all baselines at the given budget. Returns dict: name -> keep_mask."""
    n = len(embeddings)
    out = {
        "random_uniform":   random_uniform(n, budget, seed),
        "reservoir":        reservoir_sample(n, budget, seed),
        "stratified_score": stratified_by_score(anomaly_scores, budget),
        "kcenter":          kcenter_no_conformal(embeddings, budget, seed),
    }
    # The unweighted facility-location is O(N^2 * budget) but in practice
    # finishes in ~60-90s per call on SKAB/SWaT, which is tractable inside
    # the Pareto sweep (5 budgets x ~80s = ~7 min per seed of overhead).
    # The priority-weighted variant has the same asymptotic complexity but
    # the eager implementation runs ~5x slower because of the per-iteration
    # weighted-sum reduction. We therefore allow facility_location at
    # n<=6000 (covers all three datasets) and gate facility_location_priority
    # to n<=1500 (TEP only). For SKAB and SWaT, the priority comparison is
    # generated by priority_fl_focused.py using the lazy-greedy variant in
    # O(N * budget * log N), at a single representative budget per dataset.
    # Memory-bounded lazy-greedy (CELF): identical selection objective and
    # identical recall/savings as the eager O(N^2) variant (validated to 0.00pp),
    # but O(N*d) memory so it runs on the larger SKAB/SWaT test sets (n~4-5k)
    # without OOM. No n gate needed.
    out["facility_location"] = lazy_facility_location_greedy(embeddings, budget, seed)
    if n <= 1500:
        out["facility_location_priority"] = facility_location_priority_weighted(
            embeddings, anomaly_scores, budget, seed
        )
    return out


# -----------------------------------------------------------------------------
# Self-test on synthetic data
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(0)
    n_norm, n_anom = 400, 100
    X = np.vstack([
        rng.normal(0, 1, size=(n_norm, 8)),
        rng.normal(4, 1, size=(n_anom, 8)),
    ]).astype(np.float32)
    y = np.array([0] * n_norm + [1] * n_anom)
    # Score: simple feature mean (high for anomaly cluster)
    scores = X.mean(axis=1)

    print(f"n={len(X)}, anomalies={n_anom} ({100*n_anom/len(X):.1f}%)")
    print(f"{'baseline':<22} {'budget':>8} {'kept':>6} {'anom_recall':>12} {'savings%':>10}")
    print("-" * 65)
    for budget in [150, 300]:
        masks = build_baseline_keep_masks(X, scores, budget, seed=42)
        for name, mask in masks.items():
            n_kept = int(mask.sum())
            anom_kept = int(mask[y == 1].sum())
            recall = anom_kept / n_anom
            savings = 1 - n_kept / len(X)
            print(f"{name:<22} {budget:>8} {n_kept:>6} {recall:>11.2%} {savings*100:>9.1f}%")
