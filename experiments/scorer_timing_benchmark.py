#!/usr/bin/env python3
"""
Scorer training and inference time benchmark (a robustness request.

a robustness request
inference cost, so practitioners can pick the right one for their compute
budget. We benchmark the six scorers used in CAP-Dedup on a synthetic
dataset shaped like TEP (52 features, ~10000 samples, 30% anomaly).

The Bayesian-ensemble scorers (mean, variance, combined) share a single
training pass, so we report the joint training cost once and inference
cost per variant.

Output:
    results/scorer_timing.csv
    results/scorer_timing.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

N_TRAIN, N_TEST = 7000, 3000
D = 52
ANOM_RATE_TRAIN = 0.30
ANOM_RATE_TEST = 0.30


def make_synth(seed: int):
    rng = np.random.default_rng(seed)
    def _gen(n, rate):
        n_anom = int(rate * n)
        n_norm = n - n_anom
        X = np.vstack([
            rng.normal(0, 1, size=(n_norm, D)),
            rng.normal(1.5, 1, size=(n_anom, D)),
        ]).astype(np.float32)
        y = np.concatenate([np.zeros(n_norm), np.ones(n_anom)]).astype(np.int8)
        idx = rng.permutation(n)
        return X[idx], y[idx]
    X_train, y_train = _gen(N_TRAIN, ANOM_RATE_TRAIN)
    X_test,  y_test  = _gen(N_TEST,  ANOM_RATE_TEST)
    return X_train, y_train, X_test, y_test


def bench_iforest(X_train, X_test):
    from sklearn.ensemble import IsolationForest
    t0 = time.perf_counter()
    m = IsolationForest(n_estimators=200, random_state=42, n_jobs=1)
    m.fit(X_train)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = -m.score_samples(X_test)
    t_inf = time.perf_counter() - t0
    return {"train_s": round(t_train, 3),
             "inference_ms_per_sample": round(1000 * t_inf / len(X_test), 4)}


def bench_autoencoder(X_train, X_test):
    """Small MLP autoencoder on normal samples only."""
    import torch
    import torch.nn as nn
    device = "cpu"
    torch.manual_seed(42)
    enc = nn.Sequential(
        nn.Linear(D, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, 64), nn.ReLU(),
        nn.Linear(64, D),
    ).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    X_normal = torch.from_numpy(X_train).to(device)
    t0 = time.perf_counter()
    for epoch in range(20):
        opt.zero_grad()
        out = enc(X_normal)
        loss = ((out - X_normal) ** 2).mean()
        loss.backward()
        opt.step()
    t_train = time.perf_counter() - t0
    with torch.no_grad():
        t0 = time.perf_counter()
        Xt = torch.from_numpy(X_test).to(device)
        recon = enc(Xt)
        scores = ((recon - Xt) ** 2).mean(dim=1).cpu().numpy()
        t_inf = time.perf_counter() - t0
    return {"train_s": round(t_train, 3),
             "inference_ms_per_sample": round(1000 * t_inf / len(X_test), 4)}


def bench_ecod(X_train, X_test):
    from pyod.models.ecod import ECOD
    t0 = time.perf_counter()
    m = ECOD()
    m.fit(X_train)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = m.decision_function(X_test)
    t_inf = time.perf_counter() - t0
    return {"train_s": round(t_train, 3),
             "inference_ms_per_sample": round(1000 * t_inf / len(X_test), 4)}


def bench_bnn_ensemble(X_train, y_train, X_test, num_models=7, dropout=0.3, epochs=20):
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    device = "cpu"
    models = nn.ModuleList()
    for i in range(num_models):
        h = 64 + (i * 8) - 24
        models.append(nn.Sequential(
            nn.Linear(D, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(dropout*2/3),
            nn.Linear(h//2, 1), nn.Sigmoid(),
        ).to(device))
    opt = torch.optim.Adam([p for m in models for p in m.parameters()], lr=1e-3)
    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train.astype(np.float32)).to(device).unsqueeze(1)
    crit = nn.BCELoss()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        for m in models:
            opt.zero_grad()
            p = m(Xtr)
            loss = crit(p, ytr)
            loss.backward()
            opt.step()
    t_train_total = time.perf_counter() - t0
    Xte = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        # turn dropout ON for MC-Dropout uncertainty (predict_with_uncertainty does this)
        for m in models: m.train()
        # mean inference
        t0 = time.perf_counter()
        preds = torch.stack([m(Xte) for m in models], dim=0).cpu().numpy().squeeze(-1)
        t_inf_mean = time.perf_counter() - t0
        # variance inference (10 MC passes per model)
        t0 = time.perf_counter()
        mc_preds = []
        for _ in range(10):
            mc_preds.append(torch.stack([m(Xte) for m in models], dim=0))
        mc_arr = torch.stack(mc_preds).cpu().numpy().squeeze(-1)
        t_inf_var = time.perf_counter() - t0
    return {
        "train_s_shared": round(t_train_total, 3),
        "bnn_mean_inference_ms_per_sample":     round(1000 * t_inf_mean / len(X_test), 4),
        "bnn_variance_inference_ms_per_sample": round(1000 * t_inf_var / len(X_test), 4),
        "bnn_combined_inference_ms_per_sample": round(1000 * (t_inf_mean + t_inf_var) / len(X_test), 4),
    }


def main():
    seed = 42
    X_train, y_train, X_test, _ = make_synth(seed)
    print(f"Synthetic TEP-shaped dataset: n_train={len(X_train)}, n_test={len(X_test)}, d={D}")

    rows = []

    print("Bayesian neural ensemble...")
    bnn = bench_bnn_ensemble(X_train, y_train, X_test)
    rows.append({"scorer": "BNN-mean",     "train_s": bnn["train_s_shared"],
                  "inference_ms_per_sample": bnn["bnn_mean_inference_ms_per_sample"],
                  "note": "training shared across all three BNN variants"})
    rows.append({"scorer": "BNN-variance", "train_s": bnn["train_s_shared"],
                  "inference_ms_per_sample": bnn["bnn_variance_inference_ms_per_sample"],
                  "note": "10 MC-Dropout passes per ensemble member"})
    rows.append({"scorer": "BNN-combined", "train_s": bnn["train_s_shared"],
                  "inference_ms_per_sample": bnn["bnn_combined_inference_ms_per_sample"],
                  "note": "mean + variance, runs both"})

    print("Isolation Forest...")
    r = bench_iforest(X_train, X_test)
    rows.append({"scorer": "Isolation Forest", **r, "note": "200 trees"})

    print("Autoencoder...")
    r = bench_autoencoder(X_train, X_test)
    rows.append({"scorer": "Autoencoder", **r, "note": "20 training epochs"})

    print("ECOD...")
    r = bench_ecod(X_train, X_test)
    rows.append({"scorer": "ECOD", **r, "note": "pyod default"})

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "scorer_timing.csv", index=False)
    with open(out_dir / "scorer_timing.json", "w", encoding="utf-8") as f:
        json.dump({"settings": {"n_train": N_TRAIN, "n_test": N_TEST, "d": D},
                    "rows": rows}, f, indent=2)

    print("\nScorer timing on synthetic TEP-shaped data:")
    print(f"{'Scorer':<20} {'Train (s)':>10} {'Inference (ms/sample)':>22} {'Notes':<40}")
    for r in rows:
        print(f"{r['scorer']:<20} {r['train_s']:>10.3f} {r['inference_ms_per_sample']:>22.4f} {r['note']:<40}")
    print(f"\nSaved: {out_dir/'scorer_timing.csv'} and .json")


if __name__ == "__main__":
    main()
