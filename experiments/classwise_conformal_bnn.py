#!/usr/bin/env python3
"""Class-conditional (Mondrian) conformal calibration on TEP with the BNN-mean scorer: per-fault-type threshold tau_c, gate uses tau = min_c tau_c. Compares pooled vs class-conditional per-fault recall and savings."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments")); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))

from memsafe_tep_loader import load_tep_sample            # noqa: E402
from anomaly_scorers import BNNMeanScorer                  # noqa: E402
from data_splitter import stratified_split                 # noqa: E402
from core.framework import UncertaintyAwareFramework       # noqa: E402

ALPHA = 0.05
W = [0.33, 0.33, 0.17, 0.17]


def conformal_tau(scores, alpha):
    return float(np.quantile(scores, alpha, method="lower"))


def run(seed=42, n=20000):
    X, y, fid, _ = load_tep_sample(n, seed=seed, file_weights=W, return_fault_id=True)
    m = stratified_split(labels=y, seed=seed, episode_ids=np.arange(len(y)),
                         mode="stratified", ratios=(0.5, 0.1, 0.2, 0.2))
    def as_idx(a):
        a = np.asarray(a)
        return np.where(a)[0] if a.dtype == bool else a   # handle boolean masks OR index arrays
    tr, va, ca, te = as_idx(m["train"]), as_idx(m["val"]), as_idx(m["cal"]), as_idx(m["test"])

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    cfg["model"] = dict(cfg["model"]); cfg["model"]["input_dim"] = 52
    fw = UncertaintyAwareFramework(cfg)
    Xtr = fw.scaler.fit_transform(X[tr]); Xva = fw.scaler.transform(X[va])
    Xall = fw.scaler.transform(X)
    fw.train(Xtr, y[tr], Xva, y[va])
    s = np.asarray(BNNMeanScorer().fit(framework=fw).score(Xall)).flatten().astype(float)

    ca_f, te_f = ca[y[ca] == 1], te[y[te] == 1]
    classes = sorted(set(fid[ca_f]) | set(fid[te_f]))
    tau_pool = conformal_tau(s[ca_f], ALPHA)
    tau_c = {c: conformal_tau(s[ca_f][fid[ca_f] == c], ALPHA)
             for c in classes if (fid[ca_f] == c).sum() >= 1}
    tau_cc = min(tau_c.values())

    def pcr(tau):
        return {c: float((s[te_f[fid[te_f] == c]] >= tau).mean())
                for c in classes if (fid[te_f] == c).sum() > 0}

    return dict(rec_pool=pcr(tau_pool), rec_cc=pcr(tau_cc),
                keep_pool=float((s[te] >= tau_pool).mean()),
                keep_cc=float((s[te] >= tau_cc).mean()),
                agg_pool=float((s[te_f] >= tau_pool).mean()),
                agg_cc=float((s[te_f] >= tau_cc).mean()),
                ncal={int(c): int((fid[ca_f] == c).sum()) for c in classes})


if __name__ == "__main__":
    import statistics as st
    seeds = list(range(42, 52))  # paper-grade: 10 seeds
    R = [run(seed=sd) for sd in seeds]
    cls = sorted(R[0]["rec_pool"].keys())
    print(f"\n=== Class-conditional vs pooled conformal (BNN-mean, {len(seeds)} seeds, target 95%) ===")
    print("cal positives/class (seed42):", R[0]["ncal"])
    print(f"\n{'fault':>6} {'recall_POOLED':>14} {'recall_CLASSWISE':>17}")
    wp, wc = [], []
    for c in cls:
        rp = st.mean([r["rec_pool"][c] for r in R if c in r["rec_pool"]])
        rc = st.mean([r["rec_cc"][c] for r in R if c in r["rec_cc"]])
        print(f"{int(c):>6} {rp*100:>13.1f}% {rc*100:>16.1f}%{'  <-- was <95%' if rp<0.95 else ''}")
        wp.append(rp); wc.append(rc)
    print(f"\nMIN per-class recall:  pooled={min(wp)*100:.1f}%   classwise={min(wc)*100:.1f}%")
    print(f"AGG fault recall:      pooled={st.mean([r['agg_pool'] for r in R])*100:.1f}%   "
          f"classwise={st.mean([r['agg_cc'] for r in R])*100:.1f}%")
    kp = st.mean([r['keep_pool'] for r in R]); kc = st.mean([r['keep_cc'] for r in R])
    kp_sd = st.pstdev([r['keep_pool'] for r in R]); kc_sd = st.pstdev([r['keep_cc'] for r in R])
    print(f"PRESERVE fraction:     pooled={kp*100:.1f}%   classwise={kc*100:.1f}%   (lower => more savings)")
    print(f"END-TO-END SAVINGS:    pooled={(1-kp)*100:.1f}% +/-{kp_sd*100:.1f}   "
          f"classwise={(1-kc)*100:.1f}% +/-{kc_sd*100:.1f}   "
          f"(strict-floor operating point; classwise costs savings for per-class protection)")
    minwp_sd = st.pstdev([min(r['rec_pool'].values()) for r in R])
    minwc_sd = st.pstdev([min(r['rec_cc'].values()) for r in R])
    print(f"MIN per-class recall stdev across seeds: pooled +/-{minwp_sd*100:.1f}  classwise +/-{minwc_sd*100:.1f}")
