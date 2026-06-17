#!/usr/bin/env python3
"""
Latency and scalability benchmark for CAP-Dedup.

Q1 IIoT venues (IEEE IoT-J, IEEE TII) require explicit per-sample latency
+ throughput numbers + scaling analysis. This script measures:

  1. Median + p95 latency per sample for each stage:
       - Anomaly scoring (ECOD)
       - Conformal calibration + gate
       - Submodular coreset selection
       - Total end-to-end
  2. Peak memory usage (resident set size)
  3. Throughput (samples / second)
  4. Scaling: N in {1k, 5k, 10k, 50k, 100k} (synthetic replicates of TEP)

Methodology: uses TEP (10k samples) as the base dataset, with synthetic
sample replication for N > 10k to mimic large industrial deployments.

Output: results/benchmarks/latency_scalability.json + .png

Usage:
    python experiments/latency_benchmark.py
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
import torch
import yaml

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
for p in (ROOT, SRC_DIR, THIS_DIR):
    sys.path.insert(0, str(p))

from core.framework import UncertaintyAwareFramework  # noqa: E402
from anomaly_scorers import build_default_scorers  # noqa: E402
from submodular_coreset import CoverageCoreset  # noqa: E402
from conformal_layer0 import ConformalAnomalyGate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("latency_bench")


def measure_peak_memory_mb():
    """Best-effort peak memory measurement (cross-platform)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB -> MB
    except (ImportError, AttributeError):
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 ** 2)  # bytes -> MB
        except ImportError:
            return float("nan")


def percentile_latency(latencies_per_call, n_samples_per_call):
    """latencies_per_call: list of seconds per call.
    n_samples_per_call: number of samples processed per call.
    Returns dict with median/p95 in MILLISECONDS-per-sample."""
    if len(latencies_per_call) == 0:
        return {"median_ms_per_sample": float("nan"), "p95_ms_per_sample": float("nan")}
    arr = np.array(latencies_per_call) / max(n_samples_per_call, 1)  # sec per sample
    return {
        "median_ms_per_sample": float(np.median(arr) * 1000.0),
        "p95_ms_per_sample": float(np.percentile(arr, 95) * 1000.0),
        "mean_ms_per_sample": float(np.mean(arr) * 1000.0),
        "n_calls": len(latencies_per_call),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[1000, 5000, 10000, 50000, 100000],
                        help="Test set sizes to benchmark")
    parser.add_argument("--n-warmup", type=int, default=3, help="warmup runs (excluded)")
    parser.add_argument("--n-repeats", type=int, default=5, help="measured repeats per size")
    args = parser.parse_args()

    # Load config + base TEP data once
    with open(ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    from tep_data_loader import TEPDataLoader
    loader = TEPDataLoader()
    df, labels, _, raw_df = loader.load_data(sample_size=10000, random_state=42,
                                              label_type="gt_faults")
    labels = labels.astype(int)
    cfg = config.copy()
    cfg["model"] = config["model"].copy()
    cfg["model"]["input_dim"] = 52

    framework = UncertaintyAwareFramework(cfg)

    # Small train portion to get a working framework (we'll reuse it for all sizes)
    rng = np.random.default_rng(42)
    n_train = 5000
    idx = rng.permutation(len(df))
    train_idx = idx[:n_train]
    cal_idx = idx[n_train:n_train + 1000]
    X_train = framework.scaler.fit_transform(df.iloc[train_idx].values)
    X_cal = framework.scaler.transform(df.iloc[cal_idx].values)
    y_train = labels[train_idx]
    y_cal = labels[cal_idx]

    framework.train(X_train, y_train, X_cal[:500], y_cal[:500])  # quick val for early stop

    # Set up the chosen scorer (ECOD - cross-dataset robust per the paper)
    scorers = build_default_scorers()
    scorer = scorers["ecod"]
    scorer.fit(X_train, y_train, framework=framework, seed=42)

    # Fit conformal gate ONCE on cal set
    cal_scores = scorer.score(X_cal)
    gate = ConformalAnomalyGate(target_recall=0.95).fit(
        X_cal, y_cal, lambda _X, _s=cal_scores: _s
    )

    # Benchmark loop
    bench = {"args": vars(args), "sizes": []}
    for N in args.sizes:
        logger.info(f"=== Benchmark size N={N} ===")
        # Build a synthetic test set of size N by replicating TEP rows
        if N <= len(df):
            X_test = framework.scaler.transform(df.iloc[:N].values)
            y_test = labels[:N]
        else:
            reps = (N // len(df)) + 1
            X_test_full = np.tile(framework.scaler.transform(df.values), (reps, 1))[:N]
            y_test_full = np.tile(labels, reps)[:N]
            X_test = X_test_full
            y_test = y_test_full
            del X_test_full, y_test_full

        # 1) Scorer.score(X_test)
        scorer_times = []
        for _ in range(args.n_warmup):
            _ = scorer.score(X_test)
        for _ in range(args.n_repeats):
            t0 = time.time(); _ = scorer.score(X_test); scorer_times.append(time.time() - t0)

        # 2) Conformal gate preserve_mask (just a scalar threshold compare; trivial)
        cmask_times = []
        test_scores = scorer.score(X_test)
        for _ in range(args.n_warmup):
            _ = gate.preserve_mask(X_test, lambda _X, _s=test_scores: _s)
        for _ in range(args.n_repeats):
            t0 = time.time()
            _ = gate.preserve_mask(X_test, lambda _X, _s=test_scores: _s)
            cmask_times.append(time.time() - t0)

        # 3) Siamese embeddings + Coreset
        # Siamese: framework.get_embeddings is the costly part for large N
        emb_times = []
        for _ in range(args.n_warmup):
            _ = framework.get_embeddings(X_test, use_siamese=True)
        for _ in range(args.n_repeats):
            t0 = time.time()
            siamese_emb = framework.get_embeddings(X_test, use_siamese=True)
            emb_times.append(time.time() - t0)

        # Coreset: warmup once (cost scales O(N^2) memory, may run out for very large N)
        cs = CoverageCoreset(seed=42)
        priority_mask = np.zeros(len(X_test), dtype=bool)
        priority_mask[np.argsort(-test_scores)[: int(0.30 * len(X_test))]] = True
        budget = int(0.70 * len(X_test))
        coreset_times = []
        for _ in range(args.n_warmup):
            try:
                _ = cs.select(siamese_emb, test_scores, priority_mask, budget)
            except MemoryError:
                logger.warning(f"Coreset OOM at N={N}; skipping")
                coreset_times = []
                break
        if coreset_times is not None and len(coreset_times) >= 0:
            for _ in range(args.n_repeats):
                t0 = time.time()
                try:
                    _ = cs.select(siamese_emb, test_scores, priority_mask, budget)
                    coreset_times.append(time.time() - t0)
                except MemoryError:
                    logger.warning(f"Coreset OOM at N={N}")
                    coreset_times = None
                    break

        mem_mb = measure_peak_memory_mb()

        entry = {
            "N": N,
            "memory_peak_mb": mem_mb,
            "scorer_ecod":   percentile_latency(scorer_times, N),
            "conformal_gate": percentile_latency(cmask_times, N),
            "siamese_embed":  percentile_latency(emb_times, N),
            "submodular_coreset": (percentile_latency(coreset_times, N)
                                    if coreset_times else {"note": "OOM"}),
        }
        # End-to-end median (sum of components)
        e2e_median = sum(
            entry[k].get("median_ms_per_sample", 0.0) or 0.0
            for k in ["scorer_ecod", "conformal_gate", "siamese_embed", "submodular_coreset"]
        )
        entry["end_to_end_median_ms_per_sample"] = e2e_median
        entry["throughput_samples_per_sec"] = 1000.0 / e2e_median if e2e_median > 0 else float("inf")
        bench["sizes"].append(entry)
        logger.info(f"N={N}: e2e_median={e2e_median:.3f} ms/sample, "
                    f"throughput={entry['throughput_samples_per_sec']:.0f} samples/sec, "
                    f"peak_mem={mem_mb:.0f} MB")
        gc.collect()

    out_dir = ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency_scalability.json"
    with open(out_path, "w") as f:
        json.dump(bench, f, indent=2)
    logger.info(f"Saved: {out_path}")

    # Plot scaling
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        Ns = [e["N"] for e in bench["sizes"]]
        e2e = [e["end_to_end_median_ms_per_sample"] for e in bench["sizes"]]
        tput = [e["throughput_samples_per_sec"] for e in bench["sizes"]]
        ax1.loglog(Ns, e2e, "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("N (test set size)"); ax1.set_ylabel("End-to-end latency (ms/sample)")
        ax1.set_title("CAP-Dedup Per-sample Latency"); ax1.grid(True, alpha=0.3, which="both")
        ax2.loglog(Ns, tput, "s-", linewidth=2, color="darkgreen", markersize=8)
        ax2.set_xlabel("N (test set size)"); ax2.set_ylabel("Throughput (samples/sec)")
        ax2.set_title("CAP-Dedup Throughput"); ax2.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(out_dir / "scalability.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"plot failed: {e}")

    print("\n" + "=" * 80)
    print(f"{'LATENCY + SCALABILITY BENCHMARK':^80}")
    print("=" * 80)
    print(f"{'N':>8} {'scorer ms':>12} {'conformal ms':>14} {'siamese ms':>12} "
          f"{'coreset ms':>12} {'e2e ms':>10} {'tput/sec':>12} {'mem MB':>10}")
    for e in bench["sizes"]:
        sc = e["scorer_ecod"]["median_ms_per_sample"]
        cn = e["conformal_gate"]["median_ms_per_sample"]
        si = e["siamese_embed"]["median_ms_per_sample"]
        co = e.get("submodular_coreset", {}).get("median_ms_per_sample", float("nan"))
        e2 = e["end_to_end_median_ms_per_sample"]
        tp = e["throughput_samples_per_sec"]
        mm = e["memory_peak_mb"]
        print(f"{e['N']:>8} {sc:>12.4f} {cn:>14.4f} {si:>12.4f} {co:>12.4f} "
              f"{e2:>10.4f} {tp:>12.0f} {mm:>10.0f}")


if __name__ == "__main__":
    main()
