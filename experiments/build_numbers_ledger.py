#!/usr/bin/env python3
"""Aggregate the headline numbers from the result files into a single ledger for verification."""
import io, sys, os, json, datetime, re
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = Path(__file__).resolve().parent / "NUMBERS_LEDGER.md"
lines = []
def w(s=""):
    lines.append(s); print(s)

def mt(p):
    p = Path(p)
    return datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if p.exists() else "MISSING"

def modeA(csv, scorer="bnn_mean", tr="0.95"):
    df = pd.read_csv(csv)
    m = df[(df.method == "cap_dedup") & (df.scorer == scorer) & (df.conformal_target_recall.astype(str) == tr)]
    rec, sav = [], []
    for s, g in m.groupby("seed"):
        r = g.loc[g.storage_savings_pct.idxmax()]
        rec.append(r.safety_recall); sav.append(r.storage_savings_pct)
    rec, sav = np.array(rec), np.array(sav)
    return dict(n=len(rec), rec_m=rec.mean()*100, rec_s=rec.std()*100,
                sav_m=sav.mean(), sav_s=sav.std(), ge95=int((rec>=0.95).sum()))

w(f"# NUMBERS LEDGER — authoritative, built {datetime.datetime.now():%Y-%m-%d %H:%M}")
w("Every manuscript number must come from here. Each block names its source file + mtime.\n")

# ---- 1. Headline Mode A guarantee (3 datasets) ----
w("## 1. Headline: Mode A (conformal gate, BNN-mean, target 95%), per-seed strict point, 10 seeds")
for name, csv in [("TEP", R/"pareto/pareto_sweep_tep.csv"),
                  ("SKAB", R/"pareto_skab/pareto_sweep_skab.csv"),
                  ("SWaT", R/"pareto_swat/pareto_sweep_swat.csv")]:
    if Path(csv).exists():
        d = modeA(csv)
        w(f"- **{name}** [{csv.name}, mtime {mt(csv)}]: recall {d['rec_m']:.2f}%±{d['rec_s']:.2f}, "
          f"savings {d['sav_m']:.2f}%±{d['sav_s']:.2f}, seeds>=95%: {d['ge95']}/{d['n']}")
    else:
        w(f"- **{name}**: MISSING {csv}")

# ---- 2. Downstream utility ----
w("\n## 2. Downstream utility (independent RandomForest, held-out AUPRC vs savings)")
for name in ["tep", "swat"]:
    p = R/f"downstream/downstream_utility_{name}.json"
    if p.exists():
        j = json.load(open(p)); w(f"- **{name.upper()}** [{p.name}, mtime {mt(p)}]:")
        for mth, cells in j["per_method"].items():
            row = "  ".join(f"sav{(1-float(b))*100:.0f}%={cells[b]['auprc_mean']:.3f}" for b in sorted(cells))
            w(f"    {mth:16s} {row}")

# ---- 3. Component ablation (coreset vs random) ----
w("\n## 3. Component ablation: gate+random vs gate+coreset (held-out AUPRC)")
for name in ["tep", "swat"]:
    for mode in ["supervised", "oneclass"]:
        p = R/f"ablation/component_ablation_{name}_{mode}.csv"
        if p.exists():
            df = pd.read_csv(p); w(f"- **{name.upper()} / {mode}** [{p.name}, mtime {mt(p)}]:")
            for bf in sorted(df[df.method!='gate_only'].budget_frac.unique()):
                gr = df[(df.method=='gate_random')&(df.budget_frac==bf)].auprc.mean()
                gcc = df[(df.method=='gate_coreset')&(df.budget_frac==bf)].auprc.mean()
                w(f"    sav~{(1-bf)*100:.0f}%: random={gr:.3f} coreset={gcc:.3f} delta={gcc-gr:+.3f}")

# ---- 4. Streaming ----
w("\n## 4. Streaming (online gate, real time-ordered stream)")
for name in ["swat", "tep"]:
    p = R/f"streaming/streaming_dedup_{name}.json"
    if p.exists():
        j = json.load(open(p)); w(f"- **{name.upper()}** [{p.name}, mtime {mt(p)}]:")
        rs, ss = j["recall_static"], j["savings_static"]
        w(f"    static: recall {rs[0]*100:.2f}%±{rs[1]*100:.2f}  savings {ss[0]:.2f}%±{ss[1]:.2f}")
        for win, d in j.get("rolling", {}).items():
            w(f"    rolling[{win}]: recall {d['recall'][0]*100:.2f}%  savings {d['savings'][0]:.2f}%")
        w(f"    gate latency: {j['gate_latency_ms'][0]:.2f} ms/sample")

# ---- 5. Class-conditional (parsed from this-session logs) ----
w("\n## 5. Class-conditional (per-class vs pooled) — parsed from class-cond logs")
for name, lg in [("TEP", R/"_classcond_tep.log"), ("SWaT", R/"_classcond_swat.log")]:
    if Path(lg).exists():
        t = open(lg, encoding="utf-8", errors="replace").read()
        w(f"- **{name}** [{lg.name}, mtime {mt(lg)}]:")
        for key in ["MIN per-class recall", "MIN per-attack recall", "AGG fault recall",
                    "PRESERVE fraction", "END-TO-END SAVINGS"]:
            m = re.search(rf"{re.escape(key)}.*", t)
            if m: w(f"    {m.group(0).strip()[:110]}")

OUT.write_text("\n".join(lines), encoding="utf-8")
w(f"\n-> written {OUT}")
