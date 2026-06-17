#!/usr/bin/env python3
"""
SKAB (Skoltech Anomaly Benchmark) data loader for CAP-Dedup sweeps.

Layout (./datasets/SKAB/):
  anomaly-free/anomaly-free.csv   (1 file, ~10k rows, all normal)
  other/*.csv                     (14 files, mixed normal+anomaly)
  valve1/*.csv                    (16 files, mixed)
  valve2/*.csv                    (4 files, mixed)

Each CSV: semicolon-separated, columns
  [datetime, Accelerometer1RMS, Accelerometer2RMS, Current, Pressure,
   Temperature, Thermocouple, Voltage, Volume Flow RateRMS, anomaly, changepoint]

We use the 8 sensor features, binary `anomaly` label as ground truth.
Total dataset: ~46,860 rows, ~28% anomalies.

Returns the same interface as TEPDataLoader for drop-in compatibility:
  load_skab(...) -> (df_features, y_labels, file_id_per_row, columns)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("skab_loader")

ROOT = Path(__file__).resolve().parent.parent
SKAB_DIR = ROOT / "datasets" / "SKAB"

# 8 sensor features (in canonical order)
SENSOR_COLUMNS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS",
]


def _read_one(path: Path, file_id: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["file_id"] = file_id  # acts like simulationRun for split-by-file
    return df


def load_skab(
    sample_size: Optional[int] = None,
    random_state: int = 42,
    include_anomaly_free: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load the full SKAB dataset and return (features_df, labels, raw_df).

    Args:
        sample_size: if given, randomly sample this many rows (stratified by
                     file_id then label). If None, return everything.
        random_state: seed for sampling.
        include_anomaly_free: include the anomaly-free directory (always
                              label=0). Default True.

    Returns
    -------
    df_features : DataFrame of shape (n, 8)  — the 8 sensor signals
    safety_labels : ndarray of int (0/1), len n
    raw_df : DataFrame with the original columns + 'file_id'
             (used by sweep for split-by-file)
    """
    if not SKAB_DIR.exists():
        raise FileNotFoundError(f"SKAB directory not found: {SKAB_DIR}")

    file_id = 0
    frames = []
    subfolders = ["valve1", "valve2", "other"]
    if include_anomaly_free:
        subfolders.append("anomaly-free")
    for sub in subfolders:
        sub_dir = SKAB_DIR / sub
        if not sub_dir.exists():
            logger.warning(f"missing subfolder {sub_dir}")
            continue
        for csv_path in sorted(sub_dir.glob("*.csv")):
            df = _read_one(csv_path, file_id)
            frames.append(df)
            file_id += 1

    if not frames:
        raise ValueError("No SKAB CSV files loaded")

    raw = pd.concat(frames, ignore_index=True)
    logger.info(f"SKAB loaded: {len(raw)} rows from {file_id} files, "
                f"anomaly rate={raw['anomaly'].mean()*100:.1f}%")

    # Drop rows with missing values in the sensor columns (rare in SKAB but possible)
    missing = raw[SENSOR_COLUMNS].isna().any(axis=1)
    if missing.any():
        logger.info(f"dropping {int(missing.sum())} rows with NaN sensor values")
        raw = raw.loc[~missing].reset_index(drop=True)

    # SKAB's anomaly-free CSV has NaN in the `anomaly` column (since by
    # definition every row is normal). Treat those as label=0.
    raw["anomaly"] = raw["anomaly"].fillna(0).astype(int)
    raw["changepoint"] = raw["changepoint"].fillna(0).astype(int)

    # Subsample stratified by file_id + label so the split-by-file logic remains
    # valid afterwards.
    if sample_size is not None and len(raw) > sample_size:
        rng = np.random.default_rng(random_state)
        # Approximate stratification: take ceil(sample_size / n_files) per file
        per_file = max(1, sample_size // raw["file_id"].nunique())
        keep_idx = []
        for fid, grp in raw.groupby("file_id"):
            n_keep = min(len(grp), per_file)
            keep_idx.extend(rng.choice(grp.index.values, size=n_keep, replace=False))
        raw = raw.loc[keep_idx].sort_values(["file_id"]).reset_index(drop=True)
        if len(raw) > sample_size:
            raw = raw.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        logger.info(f"SKAB subsampled to {len(raw)} rows (target {sample_size})")

    df_features = raw[SENSOR_COLUMNS].copy()
    safety_labels = raw["anomaly"].to_numpy()
    return df_features, safety_labels, raw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df_feat, y, raw = load_skab()
    print(f"features shape: {df_feat.shape}")
    print(f"labels shape: {y.shape}, anomaly rate: {y.mean()*100:.2f}%")
    print(f"raw cols: {list(raw.columns)}")
    print(f"per-file row counts (first 5): {raw['file_id'].value_counts().head().to_dict()}")
