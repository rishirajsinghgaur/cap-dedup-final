#!/usr/bin/env python3
"""
Tennessee Eastman Process (TEP) Data Loader

Loads TEP dataset from RData files and prepares it for CAP-Dedup framework.

Dataset Structure:
- 4 RData files: FaultFree_Training, FaultFree_Testing, Faulty_Training, Faulty_Testing
- Each file contains: faultNumber, simulationRun, sample, and 52 process variables
- faultNumber: 0=normal, 1-20=different fault types
- 52 variables: 41 measured + 11 manipulated variables
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd

# Try to import pyreadr for RData files
try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False
    logging.warning("pyreadr not installed. Install with: pip install pyreadr")

# Project paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.evaluator import DuplicateDetector
from baselines.improved_definitions import IndustrialIoTDuplicateDefiner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TEPDataLoader:
    """Load and preprocess Tennessee Eastman Process (TEP) data"""
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize TEP Data Loader
        
        Args:
            data_dir: Directory containing TEP RData files (default: ./datasets/TEP)
        """
        if data_dir is None:
            data_dir = ROOT / "dataverse_files"
        self.data_dir = Path(data_dir)
        self.detector = DuplicateDetector()
        
        # Expected RData files
        self.files = {
            'fault_free_training': self.data_dir / 'TEP_FaultFree_Training.RData',
            'fault_free_testing': self.data_dir / 'TEP_FaultFree_Testing.RData',
            'faulty_training': self.data_dir / 'TEP_Faulty_Training.RData',
            'faulty_testing': self.data_dir / 'TEP_Faulty_Testing.RData',
        }
        
        # Verify files exist
        for name, path in self.files.items():
            if not path.exists():
                raise FileNotFoundError(f"TEP file not found: {path}")
        
        logger.info(f"TEP DataLoader initialized with data_dir: {self.data_dir}")
    
    def load_rdata_file(self, filepath: Path, max_samples: int = None, random_state: int = None) -> pd.DataFrame:
        """
        Load a single RData file, with optional sampling for very large files.

        Performance optimization: if a sibling .parquet file exists (created by
        tools/convert_tep_to_parquet.py), it is loaded INSTEAD of the RData file.
        Parquet loads in <1 sec vs 1-10 min for RData.

        Args:
            filepath: Path to RData file
            max_samples: If specified, sample this many samples (for memory efficiency)
            random_state: Random seed for deterministic sampling

        Returns:
            DataFrame with TEP data
        """
        # FAST PATH: prefer parquet sibling if present (created by
        # tools/convert_tep_to_parquet.py). Parquet reads are columnar + compressed
        # and complete in <1 sec even for 800MB RData equivalents.
        parquet_path = filepath.with_suffix(".parquet")
        if parquet_path.exists():
            logger.info(f"Loading Parquet (fast path): {parquet_path.name}")
            df = pd.read_parquet(parquet_path)
            logger.info(f"✓ File loaded into memory: {len(df)} total samples")
            if max_samples is not None and len(df) > max_samples:
                logger.info(f"Sampling {max_samples} from {len(df)} samples in {parquet_path.name}")
                df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
            logger.info(f"✓ Loaded {len(df)} samples from {parquet_path.stem}")
            return df

        # SLOW FALLBACK: RData via pyreadr (used until convert_tep_to_parquet.py is run)
        if not HAS_PYREADR:
            raise ImportError("pyreadr is required. Install with: pip install pyreadr")

        logger.info(f"Loading RData file: {filepath}")
        logger.info(f"⚠ NOTE: Large RData files (millions of rows) may take 1-5 minutes to load...")
        logger.info(f"⚠ Tip: run `python tools/convert_tep_to_parquet.py` once to get the fast path.")
        
        try:
            # Load RData file
            # CRITICAL: pyreadr.read_r() loads the ENTIRE file into memory first
            # This is slow for large files (millions of rows) but necessary
            # The max_samples parameter only affects POST-LOAD sampling, not the initial file read
            # If MemoryError occurs here, the file is too large to load regardless of max_samples
            logger.info(f"Reading RData file structure (this may take a while for large files)...")
            result = pyreadr.read_r(str(filepath))
            
            # RData files contain dataframes with specific names
            # Based on TEP documentation: fault_free_training, fault_free_testing, etc.
            # The key in the result dict is the R variable name
            if len(result) == 0:
                raise ValueError(f"No data found in {filepath}")
            
            # Get the first (and usually only) dataframe
            df_name = list(result.keys())[0]
            df = result[df_name]
            logger.info(f"✓ File loaded into memory: {len(df)} total samples")
            
            # If max_samples is specified and file is larger, sample it
            # NOTE: This sampling happens AFTER the entire file is loaded
            if max_samples is not None and len(df) > max_samples:
                logger.info(f"Sampling {max_samples} from {len(df)} samples in {filepath.name}")
                df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
            
            logger.info(f"✓ Loaded {len(df)} samples from {df_name}")
            return df
            
        except MemoryError as e:
            # CRITICAL FIX: MemoryError at pyreadr.read_r() means the file is too large to load
            # max_samples doesn't help here because pyreadr must load entire file first
            # The retry logic was misleading - it would fail again at the same line
            logger.error(f"❌ Memory error loading {filepath}: {e}")
            logger.error(f"❌ File is too large to load into memory (pyreadr must load entire file)")
            logger.error(f"❌ The max_samples parameter cannot help - it only affects post-load sampling")
            logger.error(f"❌ Solutions:")
            logger.error(f"   1. Use a system with more RAM")
            logger.error(f"   2. Convert RData to CSV and use pd.read_csv() with nrows parameter")
            logger.error(f"   3. Use R to sample the file first, then load the sampled version")
            logger.error(f"   4. Skip this file and use other TEP files")
            # Get file size for error message (if available)
            try:
                file_size_mb = filepath.stat().st_size / (1024 * 1024)
                file_size_str = f"{file_size_mb:.1f} MB"
            except:
                file_size_str = "unknown size"
            
            raise MemoryError(
                f"Cannot load {filepath.name}: File too large for available memory ({file_size_str}). "
                f"pyreadr.read_r() must load entire file into memory before sampling. "
                f"max_samples parameter cannot help because it only affects post-load sampling. "
                f"Solutions: Use more RAM, convert RData to CSV and use pd.read_csv(nrows=...), "
                f"or use R to pre-sample the file."
            ) from e
    
    def load_all_data(self, sample_size: int = None, random_state: int = 42) -> pd.DataFrame:
        """
        Load all TEP data files and combine into single dataset
        
        Strategy: Sample from each file before combining to avoid memory issues
        (TEP files are very large - up to 9.6M samples)
        
        Args:
            sample_size: If specified, randomly sample this many samples total
            random_state: Random seed for sampling
            
        Returns:
            Combined DataFrame with all TEP data
        """
        logger.info("Loading all TEP data files...")
        
        rng = np.random.default_rng(random_state)

        def next_seed() -> int:
            """Generate deterministic per-operation seeds."""
            return int(rng.integers(0, np.iinfo(np.int32).max))

        all_dfs = []
        total_samples = 0
        
        # Calculate samples per file if sample_size is specified
        samples_per_file = None
        if sample_size is not None:
            samples_per_file = max(1, sample_size // len(self.files))
            logger.info(f"Will sample {samples_per_file} samples from each file (target: {sample_size} total)")
        
        # Load each file and sample immediately to save memory
        file_count = len(self.files)
        for file_idx, (name, filepath) in enumerate(self.files.items(), 1):
            try:
                logger.info(f"[{file_idx}/{file_count}] Loading {name}...")
                # For very large files, pass max_samples to load_rdata_file to handle memory errors
                # This allows the loader to sample during load if needed
                # CRITICAL: pyreadr.read_r() loads ENTIRE file first (slow for 5M+ row files)
                # For quick testing with <500 samples, consider using sample_size=50-100
                max_samples_for_file = samples_per_file * 2 if samples_per_file is not None else None
                file_seed = next_seed()
                df = self.load_rdata_file(filepath, max_samples=max_samples_for_file, random_state=file_seed)
                df['source_file'] = name  # Track source
                
                # Sample from this file if needed (before combining to save memory)
                if samples_per_file is not None and len(df) > samples_per_file:
                    logger.info(f"Sampling {samples_per_file} from {len(df)} samples in {name}")
                    df = df.sample(n=samples_per_file, random_state=next_seed()).reset_index(drop=True)
                
                all_dfs.append(df)
                total_samples += len(df)
                logger.info(f"Loaded {len(df)} samples from {name} (total so far: {total_samples})")
                
                # Clear memory hint
                del df
                
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                # For very large files, try to sample directly during load
                if "Unable to allocate" in str(e) or "MemoryError" in str(e) or "memory" in str(e).lower():
                    logger.warning(f"Memory error loading {name}. This file may be too large.")
                    logger.warning(f"Attempting to load with aggressive sampling...")
                    try:
                        # Try loading with very aggressive sampling (10k samples max)
                        df = self.load_rdata_file(filepath, max_samples=10000, random_state=file_seed)
                        df['source_file'] = name
                        if samples_per_file is not None and len(df) > samples_per_file:
                            df = df.sample(n=samples_per_file, random_state=next_seed()).reset_index(drop=True)
                        all_dfs.append(df)
                        total_samples += len(df)
                        logger.info(f"Successfully loaded {len(df)} samples from {name} (with aggressive sampling)")
                        del df
                    except Exception as e2:
                        logger.warning(f"Failed to load {name} even with aggressive sampling: {e2}")
                        logger.warning(f"Skipping {name} - will use other files only")
                        continue
                else:
                    raise
        
        if len(all_dfs) == 0:
            raise ValueError("No TEP files could be loaded")
        
        # Combine all dataframes
        logger.info("Combining dataframes...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} samples, {combined_df.shape[1]} columns")
        
        # Final sample if we have more than requested
        if sample_size is not None and len(combined_df) > sample_size:
            logger.info(f"Final sampling: {sample_size} samples from {len(combined_df)} total")
            combined_df = combined_df.sample(n=sample_size, random_state=next_seed()).reset_index(drop=True)
        elif sample_size is not None and len(combined_df) < sample_size:
            logger.warning(f"⚠️ WARNING: Only loaded {len(combined_df)} samples, requested {sample_size}")
            logger.warning(f"⚠️ This may be due to memory errors loading large TEP files")
            if len(combined_df) == 0:
                raise ValueError("Insufficient samples: no data available after loading TEP files")
            if len(combined_df) < sample_size * 0.5:
                logger.error(f"❌ CRITICAL: Only {len(combined_df)} samples available, need {sample_size}")
                logger.error("❌ This is insufficient for reliable experiments. Consider:")
                logger.error("❌ 1. Reducing sample_size to available data")
                logger.error("❌ 2. Fixing memory issues to load all TEP files")
                raise ValueError(f"Insufficient samples: only {len(combined_df)} available, need {sample_size}")
            deficit = sample_size - len(combined_df)
            logger.warning(f"⚠️ Only loaded {len(combined_df)} samples (requested {sample_size})")
            logger.warning(f"⚠️ Proceeding with available data. DO NOT upsample with replacement for deduplication tasks.")
            logger.warning(f"⚠️ Upsampling would create artificial duplicates that invalidate deduplication metrics.")
            # FIXED: Do NOT upsample - artificial duplicates would inflate deduplication metrics
        
        logger.info(f"Final dataset size: {len(combined_df)} samples")
        return combined_df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for framework (remove metadata columns, keep process variables)
        
        Args:
            df: Raw TEP dataframe
            
        Returns:
            DataFrame with only process variables (52 columns)
        """
        # TEP structure: faultNumber, simulationRun, sample, then 52 process variables
        # Keep only numeric columns (process variables)
        # Remove metadata columns: faultNumber, simulationRun, sample, source_file
        
        feature_df = df.copy()
        
        # Remove metadata columns
        metadata_cols = ['faultNumber', 'simulationRun', 'sample', 'source_file']
        for col in metadata_cols:
            if col in feature_df.columns:
                feature_df = feature_df.drop(col, axis=1)
        
        # Keep only numeric columns
        feature_df = feature_df.select_dtypes(include=[np.number])
        
        # Fill NaN values
        feature_df = feature_df.fillna(feature_df.mean())
        
        logger.info(f"Prepared features: {feature_df.shape[1]} process variables")
        return feature_df
    
    def create_safety_labels(self, df: pd.DataFrame, label_type: str = 'gt_faults') -> np.ndarray:
        """
        Create safety-critical labels for TEP data
        
        Args:
            df: TEP dataframe with faultNumber column
            label_type: Type of labels to create:
                - 'gt_faults': Use only fault IDs (faultNumber > 0) - PRIMARY for breaking circularity
                - 'gt_subset': Use curated challenging faults (Faults 4, 9, 11)
                - 'proxy': Use 3-sigma outliers (original method, kept for contrast)
            
        Returns:
            Binary array (1 = safety-critical, 0 = normal)
        """
        from scipy import stats
        
        safety_labels = np.zeros(len(df), dtype=int)
        
        if label_type == 'gt_faults':
            # PRIMARY: Use only real fault IDs (breaks circularity)
            if 'faultNumber' in df.columns:
                fault_mask = df['faultNumber'] > 0
                safety_labels[fault_mask] = 1
                logger.info(f"GT-Faults labels: {fault_mask.sum()} samples ({100*fault_mask.sum()/len(df):.1f}%)")
            else:
                logger.warning("faultNumber column not found, cannot create GT-Faults labels")
        
        elif label_type == 'gt_subset':
            # Use curated challenging faults (Faults 4, 9, 11)
            if 'faultNumber' in df.columns:
                challenging_faults = [4, 9, 11]
                fault_mask = df['faultNumber'].isin(challenging_faults)
                safety_labels[fault_mask] = 1
                logger.info(f"GT-Subset labels (Faults {challenging_faults}): {fault_mask.sum()} samples ({100*fault_mask.sum()/len(df):.1f}%)")
            else:
                logger.warning("faultNumber column not found, cannot create GT-Subset labels")
        
        elif label_type == 'proxy':
            # Original 3-sigma method (kept for contrast/comparison)
            # Get process variables (exclude metadata)
            process_vars = df.select_dtypes(include=[np.number])
            metadata_cols = ['faultNumber', 'simulationRun', 'sample', 'source_file']
            for col in metadata_cols:
                if col in process_vars.columns:
                    process_vars = process_vars.drop(col, axis=1)
            
            if len(process_vars.columns) > 0:
                z_scores = np.abs(stats.zscore(process_vars, nan_policy='omit'))
                outlier_mask = (z_scores > 3).any(axis=1)
                safety_labels[outlier_mask] = 1
                logger.info(f"Proxy (3-sigma) labels: {outlier_mask.sum()} samples ({100*outlier_mask.sum()/len(df):.1f}%)")
        else:
            raise ValueError(f"Unknown label_type: {label_type}. Must be 'gt_faults', 'gt_subset', or 'proxy'")
        
        # Ensure reasonable percentage (5-30% safety-critical) for GT labels
        safety_pct = 100 * safety_labels.sum() / len(df) if len(df) > 0 else 0
        logger.info(f"Total safety-critical: {safety_labels.sum()} samples ({safety_pct:.1f}%)")
        
        # Only adjust if using GT labels (not proxy, which should remain as-is for comparison)
        if label_type in ['gt_faults', 'gt_subset']:
            if safety_pct < 5 and len(df) > 0:
                # Too few - add top 5% most critical based on fault severity
                logger.warning(f"Too few safety-critical samples ({safety_pct:.1f}%), adding top 5%")
                if 'faultNumber' in df.columns:
                    # Prioritize by fault number (higher = more severe)
                    fault_scores = df['faultNumber'].values
                    top_5pct = int(0.05 * len(df))
                    top_indices = np.argsort(fault_scores)[-top_5pct:]
                    safety_labels[top_indices] = 1
            
            if safety_pct > 30 and len(df) > 0:
                # Too many - keep top 30% most critical
                logger.warning(f"Too many safety-critical samples ({safety_pct:.1f}%), keeping top 30%")
                if 'faultNumber' in df.columns:
                    fault_scores = df['faultNumber'].values
                    top_30pct = int(0.30 * len(df))
                    top_indices = np.argsort(fault_scores)[-top_30pct:]
                    safety_labels = np.zeros(len(df), dtype=int)
                    safety_labels[top_indices] = 1
        
        return safety_labels
    
    def find_duplicates(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Find duplicate pairs in TEP data
        
        Duplicate types:
        1. Stuck sensors: consecutive identical readings (>5 samples)
        2. Network duplicates: exact copies with different timestamps
        3. Near-duplicates: cosine similarity > 0.95
        
        Args:
            df: Process variables dataframe
            
        Returns:
            List of duplicate pairs (i, j)
        """
        logger.info("Finding duplicates in TEP data...")
        
        all_duplicates = []
        
        # Method 1: Temporal duplicates (stuck sensors)
        temporal_dups = self.detector.find_temporal_duplicates(
            df, time_window=5, similarity_threshold=0.95
        )
        logger.info(f"Temporal duplicates: {len(temporal_dups)} pairs")
        all_duplicates.extend(temporal_dups)
        
        # Method 2: Stuck sensor duplicates
        stuck_dups = self.detector.find_stuck_sensor_duplicates(
            df, window_size=5, max_variance=0.01
        )
        logger.info(f"Stuck sensor duplicates: {len(stuck_dups)} pairs")
        all_duplicates.extend(stuck_dups)
        
        # Method 3: Improved duplicate detection
        try:
            improved_dups = IndustrialIoTDuplicateDefiner.detect_real_duplicates(
                df, method='all'
            )
            logger.info(f"Improved duplicates: {len(improved_dups)} pairs")
            all_duplicates.extend(improved_dups)
        except Exception as e:
            logger.warning(f"Improved duplicate detection failed: {e}")
        
        # Remove duplicates from list
        all_duplicates = list(set(all_duplicates))
        
        logger.info(f"Total duplicate pairs: {len(all_duplicates)}")
        return all_duplicates
    
    def load_data(self, sample_size: int = None, random_state: int = 42, label_type: str = 'gt_faults') -> Tuple[pd.DataFrame, np.ndarray, List[Tuple[int, int]], pd.DataFrame]:
        """
        Main method to load TEP data for framework
        
        Args:
            sample_size: Number of samples to use (None = use all)
            random_state: Random seed for sampling
            label_type: Type of safety labels ('gt_faults', 'gt_subset', 'proxy')
            
        Returns:
            Tuple of (features_df, safety_labels, duplicate_pairs, raw_df)
            raw_df is included to preserve metadata (faultNumber, simulationRun) for train/test splitting
        """
        logger.info("=" * 80)
        logger.info("TEP Data Loader: Loading Tennessee Eastman Process Dataset")
        logger.info(f"Label type: {label_type}")
        logger.info("=" * 80)
        
        # Load all data
        raw_df = self.load_all_data(sample_size=sample_size, random_state=random_state)
        
        # Prepare features (remove metadata, keep process variables)
        features_df = self.prepare_features(raw_df)
        
        # Create safety labels (need raw_df for faultNumber)
        safety_labels = self.create_safety_labels(raw_df, label_type=label_type)
        
        # Find duplicates (use features_df)
        duplicate_pairs = self.find_duplicates(features_df)
        
        logger.info("=" * 80)
        logger.info(f"TEP Data Loaded Successfully:")
        logger.info(f"  Samples: {len(features_df)}")
        logger.info(f"  Features: {features_df.shape[1]}")
        logger.info(f"  Safety-critical: {safety_labels.sum()} ({100*safety_labels.sum()/len(features_df):.1f}%)")
        logger.info(f"  Duplicate pairs: {len(duplicate_pairs)}")
        logger.info("=" * 80)
        
        return features_df, safety_labels, duplicate_pairs, raw_df


def test_tep_loader():
    """Test TEP data loader"""
    loader = TEPDataLoader()
    
    # Test with small sample
    features_df, safety_labels, duplicate_pairs, raw_df = loader.load_data(sample_size=1000, random_state=42)
    
    print(f"\nTest Results:")
    print(f"  Features shape: {features_df.shape}")
    print(f"  Safety labels: {safety_labels.sum()} / {len(safety_labels)} ({100*safety_labels.sum()/len(safety_labels):.1f}%)")
    print(f"  Duplicate pairs: {len(duplicate_pairs)}")
    print(f"  Feature columns: {list(features_df.columns[:5])}...")


if __name__ == "__main__":
    test_tep_loader()

