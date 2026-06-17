#!/usr/bin/env python3
"""
SWaT (Secure Water Treatment) data loader for CAP-Dedup sweeps.

This loader processes the "SWaT.A4 & A5 (Jul 2019)" release:
  - Main Excel file: datasets/SWaT/extracted/SWaT.A4 & A5_Jul 2019/SWaT_dataset_Jul 19 v2.xlsx
  - Single sheet "MST v2" with ~15,000 rows × 78 cols
  - Row 1 contains group labels (P1, P2, ..., P6 -- the 6 process stages)
  - Row 2 contains sensor names (FIT 101, LIT 101, ...)
  - Row 3 is a literal "value" placeholder
  - Rows 4+ are 1-Hz sensor readings
  - First column is GMT+0 timestamps
  - Some columns are categorical: "Active"/"Inactive" sensor states

Attack labels (from the companion PDF "SWaT data collection_20-07-2019 v2.pdf"):
  Six attacks were carried out during the test, all timestamps in GMT+8.
  We convert to GMT+0 (subtract 8 hours) to match the data timestamps.

  1. FIT401 spoof:        15:08:46 - 15:10:31  GMT+8  (07:08:46 - 07:10:31 GMT+0)
  2. LIT301 spoof:        15:15:00 - 15:19:32  GMT+8  (07:15:00 - 07:19:32 GMT+0)
  3. P601 ON:             15:26:57 - 15:30:48  GMT+8  (07:26:57 - 07:30:48 GMT+0)
  4. MV201+P101 multi:    15:38:50 - 15:46:20  GMT+8  (07:38:50 - 07:46:20 GMT+0)
  5. MV501 CLOSE:         15:54:00 - 15:56:00  GMT+8  (07:54:00 - 07:56:00 GMT+0)
  6. P301 OFF:            16:02:56 - 16:16:18  GMT+8  (08:02:56 - 08:16:18 GMT+0)

Returns the same interface as skab_loader.load_skab() for drop-in compatibility:
  load_swat(...) -> (df_features, y_labels, raw_df_with_attack_id)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("swat_loader")

ROOT = Path(__file__).resolve().parent.parent
SWAT_XLSX = (ROOT / "datasets" / "SWaT" / "extracted"
             / "SWaT.A4 & A5_Jul 2019"
             / "SWaT_dataset_Jul 19 v2.xlsx")
SWAT_PARQUET = SWAT_XLSX.with_suffix(".parquet")
SWAT_LABELS_PARQUET = SWAT_XLSX.with_name("SWaT_labels.parquet")

# Attack windows in GMT+0 (already timezone-adjusted from PDF's GMT+8 spec)
ATTACK_WINDOWS_GMT0 = [
    ("FIT401",      "2019-07-20 07:08:46", "2019-07-20 07:10:31"),
    ("LIT301",      "2019-07-20 07:15:00", "2019-07-20 07:19:32"),
    ("P601",        "2019-07-20 07:26:57", "2019-07-20 07:30:48"),
    ("MV201_P101",  "2019-07-20 07:38:50", "2019-07-20 07:46:20"),
    ("MV501",       "2019-07-20 07:54:00", "2019-07-20 07:56:00"),
    ("P301",        "2019-07-20 08:02:56", "2019-07-20 08:16:18"),
]


def _load_xlsx_raw() -> pd.DataFrame:
    """Load the SWaT Excel, normalising header/value rows."""
    if not SWAT_XLSX.exists():
        raise FileNotFoundError(f"SWaT main xlsx not found: {SWAT_XLSX}")

    # Row 0 = group labels (P1..P6); row 1 = sensor names; row 2 = "value" placeholder;
    # actual data starts row 3.
    # We use sensor names (row 1) as the header.
    df = pd.read_excel(SWAT_XLSX, sheet_name="MST v2", header=1, skiprows=[2], engine="openpyxl")
    # Rename first column to "timestamp"
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "timestamp"})
    # Parse timestamps. SWaT uses ISO 8601 with variable-precision microseconds
    # (e.g. "2019-07-20T04:30:02.004013Z") — need format='ISO8601' explicitly,
    # otherwise pandas's default parser fails and returns NaT.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True, errors="coerce")
    # Strip timezone for cleaner downstream comparisons (data is already in GMT+0)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    logger.info(f"SWaT raw loaded: {len(df)} rows x {df.shape[1]} cols, "
                f"time range {df['timestamp'].min()} -> {df['timestamp'].max()}")
    return df


def _label_attacks(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'anomaly' (0/1) and 'attack_id' (string) columns based on attack windows."""
    df = df.copy()
    df["anomaly"] = 0
    df["attack_id"] = "normal"
    for name, start, end in ATTACK_WINDOWS_GMT0:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        mask = (df["timestamp"] >= s) & (df["timestamp"] <= e)
        df.loc[mask, "anomaly"] = 1
        df.loc[mask, "attack_id"] = name
    return df


def _coerce_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """SWaT has some string columns ('Active'/'Inactive', etc.) — map to numeric."""
    df = df.copy()
    mapping = {
        "Active": 1, "Inactive": 0,
        "Open": 1, "OPEN": 1, "Closed": 0, "CLOSE": 0, "Close": 0,
        "ON": 1, "On": 1, "OFF": 0, "Off": 0,
    }
    for col in df.columns:
        if col in ("timestamp", "anomaly", "attack_id", "file_id"):
            continue
        if df[col].dtype == object:
            mapped = df[col].map(mapping)
            # If most values were mappable, replace
            if mapped.notna().mean() > 0.5:
                df[col] = mapped
            # Fill remaining NaNs from this column with 0 (Inactive-equivalent)
            df[col] = df[col].fillna(0)
    return df


def _prepare_cached() -> pd.DataFrame:
    """Build (or load cached) parquet of the labeled, cleaned SWaT data."""
    if SWAT_PARQUET.exists() and SWAT_LABELS_PARQUET.exists():
        logger.info(f"SWaT cached parquet found: {SWAT_PARQUET}")
        df = pd.read_parquet(SWAT_PARQUET)
        labels = pd.read_parquet(SWAT_LABELS_PARQUET)
        df["anomaly"] = labels["anomaly"].values
        df["attack_id"] = labels["attack_id"].values
        return df

    logger.info("Building SWaT cache (one-time, ~30 sec)")
    raw = _load_xlsx_raw()
    raw = _label_attacks(raw)
    raw = _coerce_categorical_columns(raw)

    # Drop columns that are all-NaN or all-constant (carry no information)
    keep = []
    for col in raw.columns:
        if col in ("timestamp", "anomaly", "attack_id"):
            keep.append(col); continue
        # Coerce to numeric where possible
        s = pd.to_numeric(raw[col], errors="coerce")
        if s.notna().sum() == 0:
            continue  # all NaN
        if s.nunique(dropna=True) <= 1:
            continue  # constant column
        raw[col] = s.fillna(s.median())
        keep.append(col)
    raw = raw[keep]
    logger.info(f"After cleaning: {len(raw)} rows x {raw.shape[1]} cols, "
                f"{raw['anomaly'].sum()} anomaly rows ({raw['anomaly'].mean()*100:.1f}%)")

    # Cache: features-only parquet + labels parquet (separating helps fast reload)
    SWAT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    raw.drop(columns=["anomaly", "attack_id"]).to_parquet(SWAT_PARQUET, compression="snappy")
    raw[["anomaly", "attack_id"]].to_parquet(SWAT_LABELS_PARQUET, compression="snappy")
    logger.info(f"Cached: {SWAT_PARQUET}")
    return raw


def load_swat(
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load SWaT and return (features_df, labels, raw_df).

    Args:
        sample_size: if given, randomly subsample this many rows (stratified by
                     anomaly label to preserve the ~14% anomaly rate).
        random_state: seed for sampling.

    Returns
    -------
    df_features : DataFrame of shape (n, n_features) — all numeric sensor cols
    safety_labels : ndarray of int (0/1), len n
    raw_df : DataFrame with the original columns including timestamp + attack_id
             (used by sweep for stratified splitting via file_id assignment)
    """
    raw = _prepare_cached()

    if sample_size is not None and len(raw) > sample_size:
        # Stratified subsample by anomaly label to preserve class balance
        rng = np.random.default_rng(random_state)
        anom_idx = raw.index[raw["anomaly"] == 1].to_numpy()
        norm_idx = raw.index[raw["anomaly"] == 0].to_numpy()
        anom_frac = len(anom_idx) / len(raw)
        n_anom_keep = max(1, int(round(sample_size * anom_frac)))
        n_norm_keep = sample_size - n_anom_keep
        n_anom_keep = min(n_anom_keep, len(anom_idx))
        n_norm_keep = min(n_norm_keep, len(norm_idx))
        keep = np.concatenate([
            rng.choice(anom_idx, size=n_anom_keep, replace=False),
            rng.choice(norm_idx, size=n_norm_keep, replace=False),
        ])
        keep.sort()  # keep temporal order
        raw = raw.loc[keep].reset_index(drop=True)
        logger.info(f"SWaT subsampled to {len(raw)} rows "
                    f"({raw['anomaly'].sum()} anomaly, {(raw['anomaly']==0).sum()} normal)")

    # Build file_id for split-by-episode (analogous to TEP's simulationRun / SKAB's file_id).
    # Step 1: episode boundary = attack_id transition
    raw["file_id"] = (raw["attack_id"] != raw["attack_id"].shift()).cumsum() - 1

    # Step 2: SWaT has one HUGE "normal" episode before any attack (~13k rows of 15k).
    # Leaving it as a single file_id makes the 70/10/20 split wildly unbalanced
    # (one big file may dominate train OR test). We chunk any file with > CHUNK_SIZE
    # rows into multiple pseudo-files of CHUNK_SIZE each. This is a standard
    # preprocessing step (USAD, Sener & Savarese) - documented in the paper.
    CHUNK_SIZE = 500
    raw = raw.reset_index(drop=True)
    new_fid = []
    next_pseudo = raw["file_id"].max() + 1
    for fid, grp in raw.groupby("file_id", sort=False):
        n = len(grp)
        if n <= CHUNK_SIZE:
            new_fid.extend([fid] * n)
        else:
            # Split into ceil(n / CHUNK_SIZE) chunks
            n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
            chunk_sizes = [n // n_chunks + (1 if i < n % n_chunks else 0)
                           for i in range(n_chunks)]
            # First sub-chunk keeps original fid; rest get fresh pseudo-fids
            new_fid.extend([fid] * chunk_sizes[0])
            for cs in chunk_sizes[1:]:
                new_fid.extend([int(next_pseudo)] * cs)
                next_pseudo += 1
    raw["file_id"] = new_fid
    logger.info(f"After chunking large episodes (CHUNK_SIZE={CHUNK_SIZE}): "
                f"{raw['file_id'].nunique()} episodes (was 13)")

    feature_cols = [c for c in raw.columns
                    if c not in ("timestamp", "anomaly", "attack_id", "file_id")]
    df_features = raw[feature_cols].copy()
    safety_labels = raw["anomaly"].to_numpy().astype(int)

    return df_features, safety_labels, raw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df_feat, y, raw = load_swat()
    print(f"\nfeatures shape: {df_feat.shape}")
    print(f"labels shape: {y.shape}, anomaly rate: {y.mean()*100:.2f}%")
    print(f"unique attack_ids: {raw['attack_id'].unique().tolist()}")
    print(f"file_id range: {raw['file_id'].min()}..{raw['file_id'].max()} ({raw['file_id'].nunique()} unique)")
    print(f"per-attack rows:")
    print(raw['attack_id'].value_counts())
    print(f"\nfeature dtypes (first 10): {df_feat.dtypes.head(10).to_dict()}")
