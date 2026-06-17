#!/usr/bin/env python3
"""Large-N conformal-gate recall on real TEP (Stage 1 only).

Recall is a Stage-1 (conformal gate) property; the gate is O(N). The Stage-2 coreset is
O(N^2) and caps at ~50k, so it is not run here -- and it cannot lower recall (it only adds
points; recall is monotone under set inclusion). We measure the gate's achieved fault recall
and gate-level savings at the 95% conformal operating point on real TEP draws of
N in {10000, 50000, 100000}, 3 seeds each, reusing the exact pipeline
(loader -> stratified 70/10/10/10 split -> BNN-mean scorer -> ConformalAnomalyGate).

N=10000 reproduces the Table 2 headline recall (~95.6% over 10 seeds; ~94.6% over these 3
seeds), validating the gate-only path. Writes results/largeN_gate_recall.json only.
"""
import os, sys, json, random
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402
torch.set_num_threads(2)
import yaml  # noqa: E402
from tep_data_loader import TEPDataLoader  # noqa: E402
from core.framework import UncertaintyAwareFramework  # noqa: E402
from conformal_layer0 import ConformalAnomalyGate  # noqa: E402
from anomaly_scorers import build_default_scorers  # noqa: E402
from data_splitter import stratified_split  # noqa: E402


def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def gate_recall_for(seed, sample_size, config, target_recall=0.95):
    set_seeds(seed)
    loader = TEPDataLoader()
    df, safety_labels, _dup, raw_df = loader.load_data(
        sample_size=sample_size, random_state=seed, label_type="gt_faults")
    safety_labels = safety_labels.astype(int)

    cfg = config.copy(); cfg["model"] = config["model"].copy(); cfg["model"]["input_dim"] = 52
    framework = UncertaintyAwareFramework(cfg)

    episode_ids = (raw_df["simulationRun"].to_numpy()
                   if "simulationRun" in raw_df.columns else np.arange(len(df)))
    masks = stratified_split(labels=safety_labels, seed=seed, episode_ids=episode_ids,
                             mode="stratified", ratios=(0.70, 0.10, 0.10, 0.10))
    df_train = df.iloc[masks["train"]].reset_index(drop=True)
    X_train = framework.scaler.fit_transform(df_train.values)
    X_val = framework.scaler.transform(df.iloc[masks["val"]].values)
    X_cal = framework.scaler.transform(df.iloc[masks["cal"]].values)
    X_test = framework.scaler.transform(df.iloc[masks["test"]].values)
    y_train = safety_labels[masks["train"]]; y_val = safety_labels[masks["val"]]
    y_cal = safety_labels[masks["cal"]]; y_test = safety_labels[masks["test"]]

    # Train BNN (+Siamese) on TRAIN, val for early stopping. Causal discovery and uncertainty
    # are skipped: neither feeds the gate, and they are the costly steps at large N.
    framework.train(X_train, y_train, X_val, y_val)

    scorers = build_default_scorers()
    name = "bnn_mean" if "bnn_mean" in scorers else [k for k in scorers if "bnn" in k and "mean" in k][0]
    scorer = scorers[name]
    scorer.fit(X_train, y_train, framework=framework, seed=int(seed))
    s_cal = np.asarray(scorer.score(X_cal)).flatten().astype(float)
    s_test = np.asarray(scorer.score(X_test)).flatten().astype(float)

    gate = ConformalAnomalyGate(target_recall=target_recall).fit(
        X_cal, y_cal, lambda _X, _s=s_cal: _s)
    mask = gate.preserve_mask(X_test, lambda _X, _s=s_test: _s)
    faults = (y_test == 1)
    recall = float(mask[faults].mean()) if faults.sum() else float("nan")
    preserved_frac = float(mask.mean())
    return dict(seed=int(seed), sample_size=int(sample_size), scorer=name,
                n_test=int(len(y_test)), n_test_faults=int(faults.sum()),
                n_cal_faults=int((y_cal == 1).sum()), tau=float(gate.tau),
                recall_pct=round(100 * recall, 3),
                gate_savings_pct=round(100.0 * (1.0 - preserved_frac), 3))


def main():
    import ctypes
    try:  # keep Windows awake for the run: ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_AWAYMODE_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)
    except Exception:
        pass
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    seeds = list(range(42, 52))  # 10 seeds, matching the headline protocol
    sizes = [10000, 50000, 100000]
    out_path = ROOT / "results" / "largeN_gate_recall.json"
    out = {"target_recall": 0.95, "runs": [], "summary": {}}
    if out_path.exists():  # resume per (size, seed): reuse any completed run
        try:
            prev = json.load(open(out_path))
            out["runs"] = [r for r in prev.get("runs", []) if "recall_pct" in r]
        except Exception:
            pass
    done = {(r["sample_size"], r["seed"]) for r in out["runs"]}

    def summarise(N):
        rows = [r for r in out["runs"] if r["sample_size"] == N]
        rec = np.array([r["recall_pct"] for r in rows]); sav = np.array([r["gate_savings_pct"] for r in rows])
        out["summary"][str(N)] = dict(
            n_seeds=len(rows),
            recall_mean=round(float(rec.mean()), 2), recall_std=round(float(rec.std(ddof=1)), 2),
            gate_savings_mean=round(float(sav.mean()), 2), gate_savings_std=round(float(sav.std(ddof=1)), 2),
            seeds_at_or_above_95=int((rec >= 95.0).sum()))

    for N in sizes:
        for sd in seeds:
            if (N, sd) in done:
                continue
            print(f"[N={N} seed={sd}] running...", flush=True)
            r = gate_recall_for(sd, N, config)
            out["runs"].append(r); done.add((N, sd))
            print(f"   -> recall={r['recall_pct']}% gate_savings={r['gate_savings_pct']}%", flush=True)
            summarise(N); json.dump(out, open(out_path, "w"), indent=2)
        summarise(N); json.dump(out, open(out_path, "w"), indent=2)
        s = out["summary"][str(N)]
        print(f"== N={N}: recall {s['recall_mean']}+-{s['recall_std']}% "
              f"({s['seeds_at_or_above_95']}/{s['n_seeds']} >=95) ==", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
