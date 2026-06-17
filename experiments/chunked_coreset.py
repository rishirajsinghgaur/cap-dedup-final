#!/usr/bin/env python3
"""
Chunked / streaming variant of the CAP-Dedup coreset stage (Appendix A
prototype).

Handles the large-N case: the
unchunked Stage-2 caps N at ~50k per call because of the O(N^2)
similarity-matrix construction. The chunked variant processes the test
stream in fixed-size blocks of B samples and applies CAP-Dedup within each
block, preserving Theorem 1's marginal recall guarantee on a per-block
basis and maintaining the 2-approximation cover guarantee within each
block. The chunk-to-chunk relationship is documented honestly: cross-chunk
diversity is not guaranteed in this prototype; a sliding-window variant
that maintains a shared coreset is sketched in the docstring below.

Usage:
  python experiments/chunked_coreset.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Imports from the main framework
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(ROOT / "experiments"))
from conformal_layer0 import ConformalAnomalyGate  # noqa: E402
from submodular_coreset import CoverageCoreset  # noqa: E402


def chunked_cap_dedup(
    X_test: np.ndarray,
    embeddings: np.ndarray,
    test_scores: np.ndarray,
    cal_scores_pos: np.ndarray,
    cal_target_recall: float,
    coreset_budget_per_chunk: int,
    chunk_size: int,
) -> dict:
    """Run CAP-Dedup over the test stream in fixed-size chunks.

    Each chunk:
        1. Take chunk_size consecutive test samples.
        2. Apply conformal gate with the (already calibrated) threshold tau
           computed from cal_scores_pos.
        3. Apply k-center greedy within the chunk, seeded by the chunk's
           preserve set M_chunk. Budget = coreset_budget_per_chunk (capped
           at chunk_size).

    Returns:
        {
          "n_chunks": int,
          "total_kept": int,
          "total_in":  int,
          "total_savings_pct": float,
          "per_chunk": [ {"chunk_idx", "n_in", "n_M", "n_kept", "elapsed_s"} ],
          "elapsed_total_s": float,
        }

    Cross-chunk diversity note:
        Because each chunk runs its own k-center greedy, the global coreset
        S = union_chunks(S_chunk) is *not* guaranteed to be a 2-approximation
        on the full stream. The marginal recall guarantee (Theorem 1) is
        preserved per chunk because the calibration set is drawn once and is
        independent of the test chunks; if the test chunks are exchangeable
        with the calibration set (the i.i.d.-stratified-random regime we use
        in this study) the per-chunk preserve sets are sufficient.
        A sliding-window variant that maintains a shared coreset of fixed
        size K (the "fixed-budget streaming greedy" of Krause et al. 2014)
        recovers a global (1 - 1/e) approximation; we leave that to follow-up
        because it changes the algorithmic structure beyond what this
        prototype claims.
    """
    n_total = len(X_test)
    n_chunks = int(np.ceil(n_total / chunk_size))

    # We bypass the gate's full fit() because our cal_scores_pos is already
    # the per-positive scorer output (the synthetic benchmark does not have
    # raw features). We compute tau directly using the same finite-sample
    # rule the gate uses internally.
    alpha = 1.0 - cal_target_recall
    n_cal = len(cal_scores_pos)
    k = int(np.floor(alpha * (n_cal + 1)))
    k = max(1, min(k, n_cal))
    tau = float(np.sort(cal_scores_pos)[k - 1])

    out = {
        "n_chunks": n_chunks, "tau": tau,
        "chunk_size": chunk_size, "budget_per_chunk": coreset_budget_per_chunk,
        "per_chunk": [], "total_kept": 0, "total_in": 0,
        "elapsed_total_s": 0.0,
    }
    t_total = time.perf_counter()
    for ci in range(n_chunks):
        a, b = ci * chunk_size, min((ci + 1) * chunk_size, n_total)
        emb_chunk = embeddings[a:b]
        scores_chunk = test_scores[a:b]
        n_in = b - a

        # Stage 1: conformal preserve mask
        preserve_mask = scores_chunk >= tau
        n_M = int(preserve_mask.sum())

        # Stage 2: k-center greedy with chunk's must-preserve as seed
        budget = min(coreset_budget_per_chunk, n_in)
        if budget <= n_M:
            keep_chunk = preserve_mask.copy()
        else:
            coreset = CoverageCoreset(seed=42)
            try:
                keep_chunk = coreset.select(
                    embeddings=emb_chunk,
                    anomaly_scores=scores_chunk,
                    must_preserve_mask=preserve_mask,
                    budget=budget,
                )
            except Exception as exc:
                # Fall back to keeping just M_chunk if greedy fails
                print(f"  [chunk {ci}] coreset select failed: {exc}; falling back")
                keep_chunk = preserve_mask.copy()

        n_kept = int(keep_chunk.sum())
        out["per_chunk"].append({
            "chunk_idx": ci, "n_in": int(n_in),
            "n_M":  n_M,    "n_kept": n_kept,
        })
        out["total_kept"] += n_kept
        out["total_in"]   += n_in
    out["elapsed_total_s"]   = float(time.perf_counter() - t_total)
    out["total_savings_pct"] = (1.0 - out["total_kept"] / out["total_in"]) * 100.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", default="20000,50000,100000,250000",
                          help="comma-separated N values to benchmark")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--budget-frac", type=float, default=0.30,
                          help="per-chunk coreset budget as a fraction of chunk_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--cal-size", type=int, default=500)
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()

    ns = [int(s) for s in args.ns.split(",")]
    out_dir = ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic data: 70% normal, 30% anomalies with separated mean.
    # The Siamese-projected embeddings are simulated directly so we
    # can scale N without re-training the encoder.
    rng = np.random.default_rng(args.seed)
    cal_scores_pos = rng.standard_normal(args.cal_size).clip(0.0) + 1.5

    results = []
    for N in ns:
        n_anom = int(0.30 * N)
        n_norm = N - n_anom
        # Synthetic test scores: normals ~ N(0, 1) clipped >=0; anomalies ~ N(2, 0.7)
        test_scores = np.concatenate([
            rng.standard_normal(n_norm).clip(0.0),
            rng.normal(2.0, 0.7, size=n_anom),
        ])
        idx = rng.permutation(N)
        test_scores = test_scores[idx]
        # Synthetic embeddings: 32-d gaussian, L2-normalised
        emb = rng.standard_normal((N, args.feature_dim)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12

        budget = int(args.budget_frac * args.chunk_size)
        res = chunked_cap_dedup(
            X_test=np.zeros((N, 1)),  # not used downstream of embeddings
            embeddings=emb,
            test_scores=test_scores,
            cal_scores_pos=cal_scores_pos,
            cal_target_recall=args.target_recall,
            coreset_budget_per_chunk=budget,
            chunk_size=args.chunk_size,
        )
        row = {
            "N":                       N,
            "n_chunks":                res["n_chunks"],
            "chunk_size":              args.chunk_size,
            "budget_per_chunk":        budget,
            "total_kept":              res["total_kept"],
            "total_savings_pct":       round(res["total_savings_pct"], 2),
            "elapsed_total_s":         round(res["elapsed_total_s"], 2),
            "throughput_samples_per_s":round(N / max(res["elapsed_total_s"], 1e-9), 1),
            "ms_per_sample":           round(1000 * res["elapsed_total_s"] / max(N, 1), 4),
        }
        results.append(row)
        print(f"N={N:>7}, chunks={row['n_chunks']:>3}, "
              f"kept={row['total_kept']:>7}, savings={row['total_savings_pct']:>5.1f}%, "
              f"elapsed={row['elapsed_total_s']:>7.2f}s, "
              f"throughput={row['throughput_samples_per_s']:>8.0f} samp/s")

    pd.DataFrame(results).to_csv(out_dir / "chunked_latency.csv", index=False)
    with open(out_dir / "chunked_latency.json", "w", encoding="utf-8") as f:
        json.dump({"settings": vars(args), "rows": results}, f, indent=2)
    print(f"\nSaved: {out_dir / 'chunked_latency.csv'}")
    print(f"Saved: {out_dir / 'chunked_latency.json'}")


if __name__ == "__main__":
    main()
