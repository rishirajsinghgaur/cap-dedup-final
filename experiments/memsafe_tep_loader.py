#!/usr/bin/env python3
"""Memory-bounded TEP loader: streams parquet row-group batches and gathers a seeded sample, so peak memory is independent of file size. Time-aware labels (active fault iff faultNumber>0 and sample>injection). load_tep_stream() returns a time-ordered episode stream."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
TEP_DIR = ROOT / "datasets" / "TEP"
FILES = [
    "TEP_FaultFree_Training.parquet",
    "TEP_FaultFree_Testing.parquet",
    "TEP_Faulty_Training.parquet",
    "TEP_Faulty_Testing.parquet",
]
NON_FEATURE = {"faultNumber", "simulationRun", "sample"}

# Verified empirically (verify_injection.py) against the data: a faulty-run row is a
# genuine (active) fault only when `sample` exceeds the fault-injection index.
#   testing files  -> injection at sample 160
#   training files -> injection at sample 20
# Fault-free files contain no faults (faultNumber==0), so injection is irrelevant.
INJECTION = {
    "TEP_FaultFree_Training.parquet": 0,     # all normal
    "TEP_FaultFree_Testing.parquet":  0,     # all normal
    "TEP_Faulty_Training.parquet":    20,
    "TEP_Faulty_Testing.parquet":     160,
}


def _feature_columns(parquet_path: Path) -> list[str]:
    names = pq.read_schema(parquet_path).names
    return [c for c in names if c not in NON_FEATURE]


def _sample_one_file(parquet_path: Path, n_keep: int, rng: np.random.Generator,
                     feat_cols: list[str], injection: int, batch_size: int = 65536):
    """Uniformly sample n_keep rows from one parquet file (streaming batches) and
    return (X, y) where y is the OPTION-C time-aware label:
    y = 1 iff faultNumber>0 AND sample > injection."""
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    n_keep = min(n_keep, total)
    chosen = np.sort(rng.choice(total, size=n_keep, replace=False))  # global row idxs

    feats_parts, y_parts, fid_parts = [], [], []
    offset = 0
    ci = 0  # pointer into `chosen`
    cols = feat_cols + ["faultNumber", "sample"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        blen = batch.num_rows
        lo = ci
        while ci < len(chosen) and chosen[ci] < offset + blen:
            ci += 1
        if ci > lo:
            local = chosen[lo:ci] - offset
            tbl = batch.to_pydict()
            fmat = np.column_stack([np.asarray(tbl[c], dtype=np.float32)[local] for c in feat_cols])
            fnum = np.asarray(tbl["faultNumber"])[local]
            samp = np.asarray(tbl["sample"])[local]
            y = ((fnum > 0) & (samp > injection)).astype(np.int8)   # OPTION C: time-aware
            fid = np.where(y == 1, fnum, 0).astype(np.int16)        # active fault class, else 0
            feats_parts.append(fmat)
            y_parts.append(y); fid_parts.append(fid)
        offset += blen
        if ci >= len(chosen):
            break
    X = np.vstack(feats_parts) if feats_parts else np.empty((0, len(feat_cols)), np.float32)
    y = np.concatenate(y_parts) if y_parts else np.empty((0,), np.int8)
    fault_id = np.concatenate(fid_parts) if fid_parts else np.empty((0,), np.int16)
    return X, y, fault_id


def load_tep_sample(sample_size: int, seed: int = 42, tep_dir: Path = TEP_DIR,
                    file_weights=None, return_fault_id: bool = False):
    """Return (X[float32, n x 52], y[int8, n], feat_cols) sampled across the 4 TEP
    files with OPTION-C time-aware fault labelling (no magnitude cap).
    If return_fault_id=True, returns (X, y, fault_id, feat_cols) where fault_id is the
    active fault class (IDV 1..20) for anomalies and 0 for normal — needed for
    class-conditional conformal calibration.

    file_weights: optional list of 4 fractions (sum=1) controlling how many rows come
    from each of [FaultFree_Train, FaultFree_Test, Faulty_Train, Faulty_Test]. Default
    is equal (0.25 each). Increasing the fault-free weights lowers the anomaly rate.
    """
    rng = np.random.default_rng(seed)
    paths = [tep_dir / f for f in FILES]
    feat_cols = _feature_columns(paths[0])
    if file_weights is None:
        file_weights = [1 / len(paths)] * len(paths)
    Xs, ys, fids = [], [], []
    for p, w in zip(paths, file_weights):
        n_p = max(1, int(round(sample_size * w)))
        sub_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        X, y, fid = _sample_one_file(p, n_p, sub_rng, feat_cols, INJECTION[p.name])
        Xs.append(X); ys.append(y); fids.append(fid)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    fault_id = np.concatenate(fids)
    perm = rng.permutation(len(y))                 # shuffle combined sample
    if return_fault_id:
        return X[perm], y[perm], fault_id[perm], feat_cols
    return X[perm], y[perm], feat_cols


def load_tep_stream(seed: int = 42, faults=range(1, 9), faulty_runs: int = 3,
                    normal_runs: int = 45, tep_dir: Path = TEP_DIR):
    """Load ORIGINAL TEP data as a faithful TIME-ORDERED stream (no shuffling of
    within-episode order). Uses the TESTING files (960-sample trajectories,
    fault injected at sample 160), so each simulationRun is a real normal->fault
    episode. We select a bounded set of episodes via pyarrow pushdown filters
    (so we never read the multi-GB file fully), keep each episode's samples in
    temporal order, and randomise only the EPISODE ARRIVAL order (seeded) so the
    stream is a realistic mix of normal and faulted episodes over time (warm-up
    therefore contains both classes). Returns (X, y, seg, feat_cols) where y is
    the Option-C time-aware label and seg is a per-episode id.
    """
    import pandas as pd  # local import; base loader stays pandas-free
    faulty = tep_dir / "TEP_Faulty_Testing.parquet"
    free = tep_dir / "TEP_FaultFree_Testing.parquet"
    feat_cols = _feature_columns(faulty)
    cols = feat_cols + ["faultNumber", "simulationRun", "sample"]
    fl = [int(f) for f in faults]
    rr = list(range(1, faulty_runs + 1))
    nr = list(range(1, normal_runs + 1))
    tf = pq.read_table(faulty, columns=cols,
                       filters=[("faultNumber", "in", fl), ("simulationRun", "in", rr)]).to_pandas()
    tn = pq.read_table(free, columns=cols,
                       filters=[("simulationRun", "in", nr)]).to_pandas()
    df = pd.concat([tf, tn], ignore_index=True)
    # Episode id = (faultNumber, simulationRun). Randomise episode ARRIVAL order
    # (seeded) but keep samples temporally ordered WITHIN each episode.
    df["_epi"] = df["faultNumber"].astype(int) * 100000 + df["simulationRun"].astype(int)
    rng = np.random.default_rng(seed)
    epis = df["_epi"].unique()
    arr = {int(e): i for i, e in enumerate(rng.permutation(epis))}
    df["_arr"] = df["_epi"].map(arr)
    df = df.sort_values(["_arr", "sample"], kind="mergesort").reset_index(drop=True)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    fnum = df["faultNumber"].to_numpy(); samp = df["sample"].to_numpy()
    y = ((fnum > 0) & (samp > INJECTION["TEP_Faulty_Testing.parquet"])).astype(int)
    seg = df["_epi"].to_numpy().astype(int)
    return X, y, seg, feat_cols


if __name__ == "__main__":
    import argparse, time, os
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss0 = proc.memory_info().rss
    except Exception:
        proc = None
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=str, default=None,
                    help="comma-separated 4 file weights, e.g. 0.33,0.33,0.17,0.17")
    args = ap.parse_args()

    fw = [float(x) for x in args.weights.split(",")] if args.weights else None
    t0 = time.time()
    X, y, cols = load_tep_sample(args.n, args.seed, file_weights=fw)
    dt = time.time() - t0
    print(f"requested N={args.n}  ->  loaded {len(y)} rows x {X.shape[1]} features")
    print(f"fault rate (y=1): {y.mean()*100:.2f}%")
    print(f"wall-clock: {dt:.1f}s")
    if proc is not None:
        peak = (proc.memory_info().rss - rss0) / 1e6
        print(f"approx RSS growth during load: {peak:.0f} MB "
              f"(current RSS {proc.memory_info().rss/1e9:.2f} GB)")
