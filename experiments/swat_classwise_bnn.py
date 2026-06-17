#!/usr/bin/env python3
"""Class-conditional conformal calibration on SWaT (per attack type) with the BNN-mean scorer; compares pooled vs per-attack recall and savings."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments")); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from swat_loader import load_swat                         # noqa: E402
from anomaly_scorers import BNNMeanScorer                 # noqa: E402
from data_splitter import stratified_split                # noqa: E402
from core.framework import UncertaintyAwareFramework      # noqa: E402

ALPHA = 0.05


def conformal_tau(scores, alpha):
    return float(np.quantile(scores, alpha, method="lower"))


def run(seed=42):
    df, y, raw = load_swat(random_state=seed)
    X = df.values.astype(np.float32)
    y = np.asarray(y).astype(int)
    attack = raw["attack_id"].to_numpy()        # 'normal' or attack name
    m = stratified_split(labels=y, seed=seed, episode_ids=np.arange(len(y)),
                         mode="stratified", ratios=(0.5, 0.1, 0.2, 0.2))
    def as_idx(a):
        a = np.asarray(a); return np.where(a)[0] if a.dtype == bool else a
    tr, va, ca, te = (as_idx(m[k]) for k in ("train", "val", "cal", "test"))

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    cfg["model"] = dict(cfg["model"]); cfg["model"]["input_dim"] = X.shape[1]
    fw = UncertaintyAwareFramework(cfg)
    Xtr = fw.scaler.fit_transform(X[tr]); Xva = fw.scaler.transform(X[va]); Xall = fw.scaler.transform(X)
    fw.train(Xtr, y[tr], Xva, y[va])
    s = np.asarray(BNNMeanScorer().fit(framework=fw).score(Xall)).flatten().astype(float)

    ca_f, te_f = ca[y[ca] == 1], te[y[te] == 1]
    classes = sorted(set(attack[ca_f]) | set(attack[te_f]))
    tau_pool = conformal_tau(s[ca_f], ALPHA)
    tau_c = {c: conformal_tau(s[ca_f][attack[ca_f] == c], ALPHA)
             for c in classes if (attack[ca_f] == c).sum() >= 1}
    tau_cc = min(tau_c.values())

    def pcr(tau):
        return {c: float((s[te_f[attack[te_f] == c]] >= tau).mean())
                for c in classes if (attack[te_f] == c).sum() > 0}
    return dict(rec_pool=pcr(tau_pool), rec_cc=pcr(tau_cc),
                keep_pool=float((s[te] >= tau_pool).mean()),
                keep_cc=float((s[te] >= tau_cc).mean()),
                ncal={str(c): int((attack[ca_f] == c).sum()) for c in classes})


if __name__ == "__main__":
    import statistics as st
    R = [run(seed=sd) for sd in range(42, 52)]  # paper-grade: 10 seeds
    cls = sorted(R[0]["rec_pool"].keys())
    print(f"\n=== SWaT per-attack class-conditional vs pooled (BNN-mean, {len(R)} seeds, target 95%) ===")
    print("cal positives/attack (seed42):", R[0]["ncal"])
    print(f"\n{'attack':>16} {'recall_POOLED':>14} {'recall_CLASSWISE':>17}")
    wp, wc = [], []
    for c in cls:
        rp = st.mean([r["rec_pool"][c] for r in R if c in r["rec_pool"]])
        rc = st.mean([r["rec_cc"][c] for r in R if c in r["rec_cc"]])
        print(f"{str(c):>16} {rp*100:>13.1f}% {rc*100:>16.1f}%{'  <-- was <95%' if rp<0.95 else ''}")
        wp.append(rp); wc.append(rc)
    print(f"\nMIN per-attack recall:  pooled={min(wp)*100:.1f}%   classwise={min(wc)*100:.1f}%")
    kp = st.mean([r['keep_pool'] for r in R]); kc = st.mean([r['keep_cc'] for r in R])
    kp_sd = st.pstdev([r['keep_pool'] for r in R]); kc_sd = st.pstdev([r['keep_cc'] for r in R])
    print(f"PRESERVE fraction:      pooled={kp*100:.1f}%   classwise={kc*100:.1f}%   (lower => more savings)")
    print(f"END-TO-END SAVINGS:     pooled={(1-kp)*100:.1f}% +/-{kp_sd*100:.1f}   "
          f"classwise={(1-kc)*100:.1f}% +/-{kc_sd*100:.1f}")
    print(f"COST of per-attack protection (savings drop): {((kc-kp))*100:.1f} pp")
    print("\n(If classwise stays near pooled preserve-fraction here, per-class protection is")
    print(" CHEAP on sparse SWaT -> demonstrates the density-dependent cost claim.)")
