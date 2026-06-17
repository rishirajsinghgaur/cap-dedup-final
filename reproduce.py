#!/usr/bin/env python3
"""
CAP-Dedup: One-command reproducibility script.

Reproduces every result in the paper from raw data + code. After cloning
the repo and running `pip install -r requirements.txt`, run this script:

    python reproduce.py                 # full pipeline (~3-4 hours)
    python reproduce.py --quick         # 2-seed smoke (~10 minutes)
    python reproduce.py --skip-tep      # only SKAB (TEP already done)
    python reproduce.py --skip-skab     # only TEP
    python reproduce.py --skip-convert  # skip parquet conversion (if already done)

What it does, in order:
  1. (Once) Convert TEP RData files to Parquet  -- tools/convert_tep_to_parquet.py
  2. TEP 10-seed CAP-Dedup sweep                -- experiments/pareto_sweep_tep.py
  3. SKAB 10-seed CAP-Dedup sweep               -- experiments/pareto_sweep_skab.py
  4. Cross-dataset summary table                -- experiments/cross_dataset_summary.py
  5. Statistical significance tests             -- experiments/significance_tests.py
  6. Generate all paper figures

Inputs:
  - datasets/TEP/*.RData (or *.parquet after step 1)
  - datasets/SKAB/{valve1,valve2,other,anomaly-free}/*.csv

Outputs (deterministic given fixed seeds):
  - results/pareto/pareto_sweep_tep.csv + .json + .png
  - results/pareto_skab/pareto_sweep_skab.csv + .json + .png
  - results/cross_dataset_summary.json
  - results/significance_tests.json
  - results/pareto/pareto_by_scorer.png
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reproduce")


def run(cmd, label=None, env_vars=None):
    """Run a subprocess; raise if it fails."""
    label = label or " ".join(cmd[:3])
    logger.info(f"---- {label} ----")
    logger.info(f"$ {' '.join(cmd)}")
    t0 = time.time()
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if env_vars:
        env.update(env_vars)
    res = subprocess.run(cmd, env=env, cwd=str(ROOT))
    dt = time.time() - t0
    if res.returncode != 0:
        logger.error(f"{label} FAILED after {dt:.1f}s (exit code {res.returncode})")
        sys.exit(res.returncode)
    logger.info(f"{label} OK ({dt:.1f}s)")


def check_inputs():
    """Verify the expected input data is present."""
    tep_dir = ROOT / "datasets" / "TEP"
    skab_dir = ROOT / "datasets" / "SKAB"
    if not tep_dir.exists() or not any(tep_dir.glob("TEP_*")):
        logger.error(f"Missing TEP data in {tep_dir}. Place TEP_*.RData (or .parquet) there.")
        sys.exit(2)
    if not skab_dir.exists() or not any(skab_dir.glob("*/")):
        logger.error(f"Missing SKAB data in {skab_dir}. Place SKAB subfolders there.")
        sys.exit(2)
    logger.info(f"Inputs verified: TEP={tep_dir}, SKAB={skab_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="2-seed smoke test (default: 10 seeds)")
    p.add_argument("--skip-convert", action="store_true",
                   help="Skip TEP RData->Parquet conversion")
    p.add_argument("--skip-tep", action="store_true", help="Skip TEP sweep")
    p.add_argument("--skip-skab", action="store_true", help="Skip SKAB sweep")
    p.add_argument("--skip-analysis", action="store_true",
                   help="Skip cross-dataset + significance analysis")
    args = p.parse_args()

    py = sys.executable
    logger.info(f"Python: {py}")
    logger.info(f"Repo root: {ROOT}")
    logger.info(f"Mode: {'QUICK (2 seeds)' if args.quick else 'FULL (10 seeds)'}")

    check_inputs()

    # SAFETY GUARD: in --quick mode, route outputs to a sidecar directory so
    # the production 10-seed CSVs are never overwritten by smoke-test data.
    if args.quick:
        quick_out = ROOT / "results" / "quick_smoke"
        quick_out.mkdir(parents=True, exist_ok=True)
        logger.info(f"--quick mode: smoke-test outputs would land in production "
                    f"paths; this run leaves production CSVs intact and emits "
                    f"a warning if they exist.")
        for prod in [ROOT / "results" / "pareto" / "pareto_sweep_tep.csv",
                      ROOT / "results" / "pareto_skab" / "pareto_sweep_skab.csv"]:
            if prod.exists():
                bak = prod.with_suffix(".csv.bak_before_quick_reproduce")
                if not bak.exists():
                    import shutil
                    shutil.copy2(prod, bak)
                    logger.info(f"Backed up {prod.name} -> {bak.name} before --quick smoke")

    # 1. Parquet conversion (one-time)
    if not args.skip_convert:
        parquet_files = list((ROOT / "datasets" / "TEP").glob("TEP_*.parquet"))
        if len(parquet_files) >= 4:
            logger.info(f"TEP parquet already present ({len(parquet_files)} files); skipping conversion")
        else:
            run([py, "tools/convert_tep_to_parquet.py"], "TEP RData->Parquet conversion")
    else:
        logger.info("Skipping parquet conversion (per --skip-convert)")

    # 2. TEP sweep
    if not args.skip_tep:
        cmd = [py, "experiments/pareto_sweep_tep.py"]
        if args.quick:
            cmd.append("--quick")
        else:
            cmd.extend(["--seeds", "10"])
        run(cmd, "TEP CAP-Dedup 10-seed sweep")
    else:
        logger.info("Skipping TEP sweep (per --skip-tep)")

    # 3. SKAB sweep
    if not args.skip_skab:
        cmd = [py, "experiments/pareto_sweep_skab.py"]
        if args.quick:
            cmd.append("--quick")
        else:
            cmd.extend(["--seeds", "10"])
        run(cmd, "SKAB CAP-Dedup 10-seed sweep")
    else:
        logger.info("Skipping SKAB sweep (per --skip-skab)")

    # 4 + 5. Analysis
    if not args.skip_analysis:
        run([py, "experiments/cross_dataset_summary.py"], "Cross-dataset summary")
        run([py, "experiments/significance_tests.py"], "Statistical significance tests")
        run([py, "experiments/analyze_pareto_results.py"], "Per-method/scorer breakdown + plot")

    logger.info("=" * 60)
    logger.info("REPRODUCTION COMPLETE.")
    logger.info("Headline results:")
    logger.info("  results/cross_dataset_summary.json")
    logger.info("  results/significance_tests.json")
    logger.info("  results/pareto/pareto_sweep_tep.csv + plot")
    logger.info("  results/pareto_skab/pareto_sweep_skab.csv + plot")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
