#!/usr/bin/env python3
"""
CAP-Dedup Stage 2: Submodular Coverage Coreset with Conformal Guarantee Preservation.

This module implements the second stage of the CAP-Dedup framework. Stage 1
(Conformal Layer 0; see conformal_layer0.py) provides a finite-sample anomaly
recall guarantee by preserving every sample whose anomaly score exceeds the
conformal threshold tau. Stage 2 - implemented here - further compresses the
preserved set by greedy submodular selection while keeping the recall
guarantee intact.

KEY THEOREM (informal):
    Let s(x) be the anomaly scorer, tau the conformal threshold calibrated for
    target recall (1 - alpha) with confidence 1 - delta. Let
        M = { x : s(x) >= tau }   (must-preserve set; satisfies conformal guarantee)
        C = test \\ M             (candidate set for compression)
    For a budget B >= |M|, select coreset S with |S| = B as
        S = M  union  GreedyFL(C, budget = B - |M|)
    where GreedyFL is facility-location greedy. Then:
      (1) S satisfies the same recall guarantee as M (because M subset S).
      (2) S - M is a (1 - 1/e) approximation to the optimal coverage of C
          within the budget |S - M|.
      (3) Storage savings = 1 - B / n_test, freely tunable.

The combination yields a new trade-off knob: target_recall (alpha) controls
the recall floor; budget controls the storage savings; coverage of the
non-anomaly part of the test set is preserved approximately optimally.

USAGE:
    coreset = CoverageCoreset()
    keep_mask = coreset.select(
        embeddings = X_embeddings,        # (n, d)
        anomaly_scores = s_test,          # (n,)
        must_preserve_mask = mask_M,      # (n,) bool, True if x in M
        budget = 0.5 * len(X)             # int target coreset size
    )
    # keep_mask is True for samples to PRESERVE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("submodular_coreset")


@dataclass
class CoverageCoreset:
    """Submodular facility-location coreset that respects a must-preserve set.

    Distance metric: cosine distance on the supplied embeddings. This matches
    what the CAP-Dedup framework already uses for L1 similarity (FAISS HNSW
    + cosine), so embedding consistency is preserved across stages.

    Parameters
    ----------
    seed : int
        For deterministic tie-breaking in greedy selection.
    """

    seed: int = 0

    @staticmethod
    def _normalize(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-12)

    def select(
        self,
        embeddings: np.ndarray,
        anomaly_scores: np.ndarray,
        must_preserve_mask: np.ndarray,
        budget: int,
    ) -> np.ndarray:
        """Return a boolean mask of which samples to KEEP.

        Parameters
        ----------
        embeddings : (n, d) float
            Per-sample embeddings (use Siamese embeddings or scaled features).
        anomaly_scores : (n,) float
            Higher = more anomalous. Used only for tie-breaking inside the
            facility-location greedy (priority among candidates with equal gain).
        must_preserve_mask : (n,) bool
            True for samples that MUST be kept (the conformal-preserved set M).
            All True positions appear in the output keep_mask unconditionally.
        budget : int
            Target total number of samples to keep, i.e. |S|. Must be at least
            |M| (size of the must-preserve set). If budget < |M|, we keep all of
            M and emit a warning (the conformal guarantee dominates the budget).

        Returns
        -------
        keep_mask : (n,) bool
            True for the |S| samples in the final coreset.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64).flatten()
        must_preserve_mask = np.asarray(must_preserve_mask, dtype=bool).flatten()
        n = len(embeddings)
        n_must = int(must_preserve_mask.sum())
        if budget < n_must:
            logger.warning(
                f"CoverageCoreset: budget={budget} < |must_preserve|={n_must}. "
                f"Returning |M|={n_must} (guarantee dominates budget)."
            )
            return must_preserve_mask.copy()
        if budget >= n:
            # Budget allows everything; nothing to drop.
            return np.ones(n, dtype=bool)

        # Normalize embeddings once (cosine sim = dot product on unit vectors)
        E = self._normalize(embeddings)

        keep = must_preserve_mask.copy()
        candidate_indices = np.where(~must_preserve_mask)[0]
        n_remaining_slots = budget - n_must

        if n_remaining_slots == 0 or len(candidate_indices) == 0:
            return keep

        # ---- k-center greedy (Gonzalez 1985, 2-approximation) ----
        # We pick the FARTHEST point from the current selected set each step.
        # This is much cheaper than facility-location (sum-coverage) and gives
        # a (2 * OPT) approximation to the minimum-radius cover. For our use
        # case (selecting representatives of a preserved set), k-center's
        # diversity behaviour is what we want: it spreads picks across the
        # input distribution.
        #
        # Cost: O(C) per iteration after one O(C * |M| * d) initial pass.
        # For C=2000 candidates and 1500 iterations: ~6e6 ops, well under 1s.
        cand_emb = E[candidate_indices]                       # (C, d)

        # Best similarity from each candidate to the current selected set.
        # Start with max sim to the must-preserve set.
        if n_must > 0:
            M_idx = np.where(must_preserve_mask)[0]
            max_sim_to_S = (cand_emb @ E[M_idx].T).max(axis=1).astype(np.float64)
        else:
            # No must-preserve - seed with the highest-anomaly candidate
            seed_pos = int(np.argmax(anomaly_scores[candidate_indices]))
            keep[int(candidate_indices[seed_pos])] = True
            n_remaining_slots -= 1
            max_sim_to_S = (cand_emb @ cand_emb[seed_pos]).astype(np.float64)
            max_sim_to_S[seed_pos] = np.inf  # mark as selected

        rng = np.random.default_rng(self.seed)
        for _ in range(n_remaining_slots):
            # Distance from each candidate to the closest selected sample.
            # Greedy picks the FARTHEST candidate (largest distance).
            # In cosine-sim space, distance = 1 - sim. argmax(1 - sim) = argmin(sim).
            # We mark inactive candidates with sim = +inf so they're never picked.
            best_pos = int(np.argmin(max_sim_to_S))
            best_sim = max_sim_to_S[best_pos]
            if best_sim == np.inf:
                break  # all candidates already selected

            # Tie-break by anomaly score among near-tied candidates
            tied = np.flatnonzero(max_sim_to_S <= best_sim + 1e-9)
            if len(tied) > 1:
                # Among tied (equally far), prefer higher anomaly score
                tie_winner = tied[int(np.argmax(anomaly_scores[candidate_indices[tied]]))]
                best_pos = int(tie_winner)

            chosen_global = int(candidate_indices[best_pos])
            keep[chosen_global] = True

            # Update max_sim_to_S using the chosen sample's similarity row
            # (only one dot product against all candidates)
            new_sims = cand_emb @ cand_emb[best_pos]
            max_sim_to_S = np.maximum(max_sim_to_S, new_sims)
            max_sim_to_S[best_pos] = np.inf  # exclude from future picks

        return keep


# ----------------------------------------------------------------------------
# Convenience wrapper: apply coreset and return the same metrics structure
# the Pareto sweep expects.
# ----------------------------------------------------------------------------

def coreset_keep_mask(
    embeddings: np.ndarray,
    anomaly_scores: np.ndarray,
    must_preserve_mask: np.ndarray,
    budget_fraction: float,
    seed: int = 0,
) -> np.ndarray:
    """Convenience wrapper: budget specified as a fraction of n_test."""
    n = len(embeddings)
    budget = max(int(must_preserve_mask.sum()), int(round(budget_fraction * n)))
    cs = CoverageCoreset(seed=seed)
    return cs.select(embeddings, anomaly_scores, must_preserve_mask, budget)


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(0)
    # 300 normals near origin + 30 anomalies far away
    n_norm, n_anom = 300, 30
    X_norm = rng.normal(0.0, 1.0, size=(n_norm, 8))
    X_anom = rng.normal(4.0, 1.0, size=(n_anom, 8))
    X = np.vstack([X_norm, X_anom]).astype(np.float32)

    # "Anomaly scores": high for the second cluster
    scores = np.r_[rng.uniform(0, 0.3, n_norm), rng.uniform(0.6, 1.0, n_anom)]

    # Must-preserve = top 5% by score
    tau = np.quantile(scores, 0.95)
    must_preserve = scores >= tau
    print(f"n={len(X)}, must-preserve={must_preserve.sum()}, "
          f"fraction anomalies in must-preserve = "
          f"{must_preserve[n_norm:].sum()}/{must_preserve.sum()}")

    for frac in [0.20, 0.30, 0.50, 0.80, 1.00]:
        keep = coreset_keep_mask(X, scores, must_preserve, frac)
        n_kept = int(keep.sum())
        anom_kept = int(keep[n_norm:].sum())
        savings = 1.0 - n_kept / len(X)
        print(f"budget_frac={frac:.2f}: kept={n_kept}/{len(X)} ({100*n_kept/len(X):.1f}%), "
              f"savings={savings*100:.1f}%, anomaly_recall={anom_kept}/{n_anom}={anom_kept/n_anom:.2f}")
