#!/usr/bin/env python3
"""Online evaluation of the conformal gate on a time-ordered stream: per-sample keep/drop at a calibrated threshold, with static and rolling sliding-window recalibration. Reports recall, savings, and per-sample latency."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTDIR = ROOT / "results" / "streaming"
OUTDIR.mkdir(parents=True, exist_ok=True)

ALPHA_DEFAULT = 0.05  # target 95% recall floor


def conformal_tau(anomaly_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold = alpha-quantile of calibration ANOMALY scores
    (finite-sample 'lower' method), matching the production gate."""
    if len(anomaly_scores) == 0:
        return -np.inf
    return float(np.quantile(anomaly_scores, alpha, method="lower"))


# ---------------------------------------------------------------------------
# Data: load in TRUE TIME ORDER (no shuffle). Returns X, y, and a per-sample
# stream order index plus optional segment ids (e.g. TEP run id) so we never
# bleed across independent trajectories when forming the stream.
# ---------------------------------------------------------------------------

def load_stream(dataset: str, seed: int):
    if dataset == "swat":
        from swat_loader import load_swat
        df, y, raw = load_swat(random_state=seed)  # raw has the timestamp order
        X = df.values.astype(np.float32)
        y = np.asarray(y).astype(int)
        # SWaT rows are already in acquisition order; sort by timestamp to be safe.
        order = np.argsort(raw["timestamp"].values.astype("datetime64[ns]"))
        seg = np.zeros(len(y), dtype=int)  # single continuous timeline
        return X[order], y[order], seg[order]
    elif dataset == "tep":
        # NOTE (integrity): the production memsafe_tep_loader RANDOMLY PERMUTES rows,
        # so it cannot be used to form a stream — that would be a fake (shuffled)
        # "stream". A faithful TEP stream must read the parquet in (simulationRun,
        # sample) order. We deliberately refuse the shuffled path rather than report
        # a meaningless number. SWaT is a genuine single 1 Hz timeline and is the
        # honest real-stream testbed here.
        from memsafe_tep_loader import load_tep_stream
        # ORIGINAL TEP data, time-ordered: episodes (faultNumber, simulationRun) in
        # randomised arrival order, samples temporally ordered within each episode.
        X, y, seg, _ = load_tep_stream(seed=seed)
        return X.astype(np.float32), np.asarray(y).astype(int), np.asarray(seg).astype(int)
    else:
        raise ValueError(f"unknown dataset {dataset}")


def make_scorer(X_warm, y_warm, dataset, seed):
    """Lightweight, edge-realistic scorer (Isolation Forest) fit on the warm-up
    window only. We also report the BNN-mean variant elsewhere; IF keeps the
    streaming-latency story honest for a resource-constrained edge node."""
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=seed, n_jobs=1)
    clf.fit(X_warm)
    # higher = more anomalous
    return lambda Z: -clf.score_samples(np.atleast_2d(Z))


def run_one(dataset: str, seed: int, alpha: float, cal_frac: float = 0.3,
            windows=(("1min", 60), ("5min", 300), ("10min", 600)), recal_samples: int = 60):
    X, y, seg = load_stream(dataset, seed)
    n = len(y)
    # Warm-up = first cal_frac of the ordered stream.
    n_warm = int(round(cal_frac * n))
    Xw, yw = X[:n_warm], y[:n_warm]
    if yw.sum() < 5:
        # ensure warm-up has some anomalies; extend if needed
        first_anom = np.where(y == 1)[0]
        if len(first_anom) >= 5:
            n_warm = max(n_warm, int(first_anom[4]) + 1)
            Xw, yw = X[:n_warm], y[:n_warm]

    scorer = make_scorer(Xw, yw, dataset, seed)
    s_warm = scorer(Xw)
    tau_static = conformal_tau(s_warm[yw == 1], alpha)

    # Stream the remainder.
    Xs, ys = X[n_warm:], y[n_warm:]
    s_stream = scorer(Xs)  # precompute scores; latency measured separately below

    # --- (a) STATIC tau ---
    keep_static = s_stream >= tau_static

    def metrics(keep):
        n_anom = int(ys.sum())
        rec = float(keep[ys == 1].mean()) if n_anom > 0 else 1.0
        sav = float((1.0 - keep.mean()) * 100.0)
        return rec, sav, n_anom

    # --- (b) ROLLING tau over a sliding window of the last `roll_window` samples.
    #         Windows are given as (label, n_samples); the label encodes the physical
    #         time (SWaT is 1 Hz so 300 samples = 5 min; TEP is sampled every ~3 min so
    #         the same count spans far longer). Sensitivity over several windows
    #         (literature: too-short under-calibrates, too-long masks single-instance
    #         anomalies). Recalibrate every `recal_samples` on the recent anomaly scores. ---
    def rolling(roll_window):
        roll_window = max(10, int(roll_window))
        tau_roll = tau_static
        keep = np.empty(len(ys), dtype=bool)
        recent = list(s_warm[yw == 1][-roll_window:])
        for t in range(len(ys)):
            keep[t] = s_stream[t] >= tau_roll
            if ys[t] == 1:
                recent.append(float(s_stream[t]))
                if len(recent) > roll_window:
                    recent = recent[-roll_window:]
            if (t + 1) % max(1, recal_samples) == 0 and len(recent) >= 10:
                tau_roll = conformal_tau(np.asarray(recent), alpha)
        return metrics(keep)

    rec_s, sav_s, n_anom = metrics(keep_static)
    rolling_by_win = {label: rolling(nsamp) for (label, nsamp) in windows}

    # Per-sample gate latency (the actual streaming decision cost): time the
    # score+compare for a batch of single samples.
    probe = Xs[: min(1000, len(Xs))]
    t0 = time.perf_counter()
    for row in probe:
        _ = scorer(row)[0] >= tau_static
    lat_ms = (time.perf_counter() - t0) / max(1, len(probe)) * 1000.0

    out = dict(seed=seed, n=n, n_warm=int(n_warm), n_stream=int(len(ys)),
               n_anom_stream=n_anom, tau_static=tau_static,
               recall_static=rec_s, savings_static=sav_s,
               gate_latency_ms=lat_ms)
    for w, (rc, sv, _) in rolling_by_win.items():
        out[f"recall_rolling_{w}"] = rc
        out[f"savings_rolling_{w}"] = sv
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["swat", "tep"], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    args = ap.parse_args()

    # Per-dataset windows (sample counts) with physical-time labels.
    #   SWaT: 1 Hz  -> 60/300/600 samples = 1/5/10 min.
    #   TEP : ~3-min sampling -> 100/300/500 samples ~ 5/15/25 process-hours.
    if args.dataset == "swat":
        windows = (("1min", 60), ("5min", 300), ("10min", 600)); recal = 60
    else:  # tep
        windows = (("100smp~5h", 100), ("300smp~15h", 300), ("500smp~25h", 500)); recal = 100

    runs = []
    for sd in range(42, 42 + args.seeds):
        print(f">>> [{args.dataset}] streaming seed {sd}", flush=True)
        runs.append(run_one(args.dataset, sd, args.alpha, windows=windows, recal_samples=recal))

    def agg(key):
        v = np.array([r[key] for r in runs], dtype=float)
        return float(v.mean()), float(v.std())

    win_keys = [k.replace("recall_rolling_", "") for k in runs[0] if k.startswith("recall_rolling_")]
    summary = dict(
        dataset=args.dataset, alpha=args.alpha, target_recall=1 - args.alpha,
        seeds=list(range(42, 42 + args.seeds)),
        recall_static=agg("recall_static"), savings_static=agg("savings_static"),
        rolling={w: dict(recall=agg(f"recall_rolling_{w}"), savings=agg(f"savings_rolling_{w}"))
                 for w in win_keys},
        gate_latency_ms=agg("gate_latency_ms"), per_seed=runs,
    )
    out = OUTDIR / f"streaming_dedup_{args.dataset}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== STREAMING {args.dataset.upper()} (target {100*(1-args.alpha):.0f}% recall, "
          f"{args.seeds} seeds, TIME-ORDERED real stream) ===")
    print(f"  STATIC  tau: recall {summary['recall_static'][0]*100:.2f}% +/-{summary['recall_static'][1]*100:.2f}"
          f"  savings {summary['savings_static'][0]:.2f}% +/-{summary['savings_static'][1]:.2f}")
    for w in win_keys:
        r = summary['rolling'][w]
        print(f"  ROLLING tau ({w} window): recall {r['recall'][0]*100:.2f}% +/-{r['recall'][1]*100:.2f}"
              f"  savings {r['savings'][0]:.2f}% +/-{r['savings'][1]:.2f}")
    print(f"  gate latency: {summary['gate_latency_ms'][0]:.4f} ms/sample (edge feasibility)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
