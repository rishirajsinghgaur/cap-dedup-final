#!/usr/bin/env python3
"""
One-time RData -> Parquet conversion for TEP dataset.

WHY: pyreadr reads RData files in pure Python (5-10 min per 800MB file).
     Parquet is columnar, compressed, and reads in <1 second.

WHAT: Reads each TEP_*.RData file from datasets/TEP/, writes a sibling
      TEP_*.parquet file. Idempotent — skips already-converted files
      unless --force is passed.

USAGE:
  python tools/convert_tep_to_parquet.py             # convert all, skip existing
  python tools/convert_tep_to_parquet.py --force     # re-convert everything
  python tools/convert_tep_to_parquet.py --files TEP_FaultFree_Training.RData
"""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tep_convert")

# Canonical TEP location is datasets/TEP/.
# We also accept the legacy junction path dataverse_files/ for safety.
TEP_DIRS = [
    ROOT / "datasets" / "TEP",
    ROOT / "dataverse_files",  # backward-compat junction
]

DEFAULT_FILES = [
    "TEP_FaultFree_Training.RData",
    "TEP_FaultFree_Testing.RData",
    "TEP_Faulty_Training.RData",
    "TEP_Faulty_Testing.RData",
]


def find_data_dir():
    for d in TEP_DIRS:
        if d.exists() and any(d.glob("TEP_*.RData")):
            return d
    raise FileNotFoundError(
        f"No TEP RData files found in any of: {[str(d) for d in TEP_DIRS]}"
    )


def convert_one(rdata_path: Path, force: bool = False) -> tuple[bool, float]:
    """Convert a single RData file to Parquet. Returns (did_convert, seconds)."""
    parquet_path = rdata_path.with_suffix(".parquet")
    if parquet_path.exists() and not force:
        logger.info(f"SKIP (exists): {parquet_path.name}")
        return False, 0.0

    import pyreadr

    t0 = time.time()
    logger.info(f"READ : {rdata_path.name} ({rdata_path.stat().st_size / 1e6:.1f} MB)")
    result = pyreadr.read_r(str(rdata_path))
    if not result:
        raise ValueError(f"No dataframes found in {rdata_path}")
    df_name = next(iter(result))
    df = result[df_name]
    t_read = time.time() - t0
    logger.info(f"       loaded {len(df):,} rows x {df.shape[1]} cols in {t_read:.1f}s "
                f"(R variable name: {df_name!r})")

    t1 = time.time()
    df.to_parquet(parquet_path, compression="snappy", index=False)
    t_write = time.time() - t1
    out_mb = parquet_path.stat().st_size / 1e6
    ratio = out_mb / (rdata_path.stat().st_size / 1e6)
    logger.info(f"WRITE: {parquet_path.name} ({out_mb:.1f} MB, "
                f"{ratio*100:.0f}% of RData size) in {t_write:.1f}s")

    total = time.time() - t0
    return True, total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true",
                   help="Re-convert even if .parquet already exists")
    p.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                   help="RData filenames to convert (default: all 4 TEP files)")
    args = p.parse_args()

    data_dir = find_data_dir()
    logger.info(f"TEP data directory: {data_dir}")

    targets = [data_dir / f for f in args.files]
    missing = [t for t in targets if not t.exists()]
    if missing:
        logger.error(f"Missing files: {[str(m) for m in missing]}")
        sys.exit(2)

    total_t = 0.0
    n_converted = 0
    n_skipped = 0
    for t in targets:
        try:
            converted, secs = convert_one(t, force=args.force)
            total_t += secs
            if converted:
                n_converted += 1
            else:
                n_skipped += 1
        except Exception as e:
            logger.error(f"FAILED to convert {t.name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("=" * 60)
    logger.info(f"DONE. Converted={n_converted}, Skipped(existing)={n_skipped}, "
                f"Total time={total_t:.1f}s")
    if n_converted > 0:
        logger.info("Future TEP loads will use Parquet automatically "
                    "(see tep_data_loader.py).")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
