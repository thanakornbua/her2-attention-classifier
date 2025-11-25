#!/usr/bin/env python3
"""
Create stratified train/val/test splits for Zarr-based HER2 classification.

This script handles:
- Class-balanced stratification (preserves HER2+/HER2- ratios)
- Slide-level splitting (no data leakage)
- Configurable split ratios
- Quality checks (minimum samples per class)
- Verbose statistics reporting

Usage:
    python create_zarr_splits.py --zarr-dir zarr_norm --output-dir outputs
    python create_zarr_splits.py --train 0.7 --val 0.15 --test 0.15 --seed 42
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from sklearn.model_selection import train_test_split


def load_tcga_labels(tcga_csv_path: Path) -> dict:
    """
    Load TCGA clinical labels from HER2_TCGA_clean.csv.
    
    Returns:
        Dict mapping slide_id (first portion before '.') to label (0=Negative, 1=Positive)
    """
    try:
        df = pd.read_csv(tcga_csv_path)
        if 'Slide' not in df.columns or 'Clinical.HER2.status' not in df.columns:
            raise ValueError(f"Expected columns 'Slide' and 'Clinical.HER2.status' in {tcga_csv_path}")
        
        label_map = {}
        for _, row in df.iterrows():
            slide_id = str(row['Slide']).split('.')[0]  # Take first portion before '.'
            status = str(row['Clinical.HER2.status']).strip().lower()
            
            if 'positive' in status:
                label_map[slide_id] = 1
            elif 'negative' in status:
                label_map[slide_id] = 0
            else:
                warnings.warn(f"Unknown HER2 status '{status}' for {slide_id}")
        
        print(f"✓ Loaded {len(label_map)} TCGA labels from {tcga_csv_path}")
        return label_map
    except Exception as e:
        warnings.warn(f"Could not load TCGA labels: {e}")
        return {}


def determine_label(slide_id: str, tcga_label_map: dict) -> int | None:
    """
    Determine HER2 label based on slide naming convention.
    
    Rules:
    - Her2Pos_* → class 1 (Positive)
    - Her2Neg_* → class 0 (Negative)
    - TCGA-XX-AAA* → lookup first portion (before '.') in tcga_label_map
    - Unknown format → class 1 (Positive) by default
    
    Args:
        slide_id: Slide identifier (filename without .zarr)
        tcga_label_map: Dict mapping TCGA case IDs to labels
    
    Returns:
        0 (Negative) or 1 (Positive)
    """
    # Rule 1: Her2Pos/Her2Neg
    if slide_id.startswith('Her2Pos') or slide_id.startswith('Pos_'):
        return 1
    elif slide_id.startswith('Her2Neg') or slide_id.startswith('Neg_'):
        return 0
    
    # Rule 2: TCGA slides
    if slide_id.startswith('TCGA-'):
        # Extract case ID (first portion before '.')
        case_id = slide_id.split('.')[0]
        
        if case_id in tcga_label_map:
            return tcga_label_map[case_id]
        else:
            warnings.warn(f"TCGA slide {slide_id} not found in clinical data, defaulting to class 1")
            return 1
    
    # Rule 3: Default to class 1 (Positive) for unknown formats
    warnings.warn(f"Unknown slide format '{slide_id}', defaulting to class 1 (Positive)")
    return 1


def scan_zarr_directory(zarr_dir: Path, tcga_label_map: dict = None) -> pd.DataFrame:
    """
    Scan directory for .zarr archives and extract metadata.
    
    Args:
        zarr_dir: Directory containing .zarr archives
        tcga_label_map: Dict mapping TCGA case IDs to labels (from clinical data)
    
    Returns:
        DataFrame with columns: zarr_path, slide_id, label, num_patches
    """
    zarr_dir = Path(zarr_dir)
    if not zarr_dir.exists():
        raise FileNotFoundError(f"Directory not found: {zarr_dir}")
    
    if tcga_label_map is None:
        tcga_label_map = {}
    
    records = []
    zarr_paths = sorted(zarr_dir.glob("*.zarr"))
    
    if not zarr_paths:
        raise ValueError(f"No .zarr archives found in {zarr_dir}")
    
    print(f"Scanning {len(zarr_paths)} zarr archives in {zarr_dir.name}...")
    
    skipped_count = 0
    for zarr_path in zarr_paths:
        slide_id = zarr_path.stem
        
        # Determine label using naming convention + TCGA lookup
        # Note: determine_label now always returns a valid label (0 or 1)
        label = determine_label(slide_id, tcga_label_map)
        
        # Count patches
        try:
            z = zarr.open(str(zarr_path), mode='r')
            if 'patches' in z:
                num_patches = z['patches'].shape[0]
            else:
                warnings.warn(f"No 'patches' array in {slide_id}, skipping")
                continue
        except Exception as e:
            warnings.warn(f"Error reading {slide_id}: {e}")
            continue
        
        records.append({
            'zarr_path': str(zarr_path.absolute()),
            'slide_id': slide_id,
            'label': int(label),
            'num_patches': num_patches
        })
    
    df = pd.DataFrame(records)
    
    if df.empty:
        raise ValueError("No valid zarr archives found!")
    
    if skipped_count > 0:
        print(f"⚠ Skipped {skipped_count} slides (could not determine label)")
    
    print(f"✓ Found {len(df)} valid slides from {zarr_dir.name}")
    return df


def create_stratified_splits(
    metadata_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    min_samples_per_class: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits at slide level.
    
    Args:
        metadata_df: DataFrame with zarr_path, slide_id, label, num_patches
        train_ratio: Training set proportion (default: 0.7)
        val_ratio: Validation set proportion (default: 0.15)
        test_ratio: Test set proportion (default: 0.15)
        random_state: Random seed for reproducibility
        min_samples_per_class: Minimum slides per class per split
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Check class distribution
    class_counts = metadata_df['label'].value_counts().sort_index()
    print("\nOriginal class distribution:")
    for label, count in class_counts.items():
        print(f"  Class {label}: {count} slides ({count/len(metadata_df)*100:.1f}%)")
    
    # Check if we have enough samples
    min_class_size = class_counts.min()
    if min_class_size < min_samples_per_class * 3:
        warnings.warn(
            f"Smallest class has only {min_class_size} slides. "
            f"May not have enough for all splits."
        )
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        metadata_df,
        test_size=(val_ratio + test_ratio),
        stratify=metadata_df['label'],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_df['label'],
        random_state=random_state
    )
    
    # Validation checks
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        class_dist = df['label'].value_counts().sort_index()
        print(f"\n{name} set:")
        print(f"  Total slides: {len(df)}")
        print(f"  Total patches: {df['num_patches'].sum()}")
        for label, count in class_dist.items():
            patches = df[df['label'] == label]['num_patches'].sum()
            print(f"  Class {label}: {count} slides, {patches} patches ({count/len(df)*100:.1f}%)")
        
        # Check minimum samples
        if class_dist.min() < min_samples_per_class:
            warnings.warn(
                f"{name} set has only {class_dist.min()} samples in smallest class!"
            )
    
    return train_df, val_df, test_df


def save_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Save train/val/test manifests as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "zarr_train_manifest.csv"
    val_path = output_dir / "zarr_val_manifest.csv"
    test_path = output_dir / "zarr_test_manifest.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Saved manifests:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")


def print_statistics(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print comprehensive statistics about the splits."""
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)
    
    total_slides = len(train_df) + len(val_df) + len(test_df)
    total_patches = (
        train_df['num_patches'].sum() +
        val_df['num_patches'].sum() +
        test_df['num_patches'].sum()
    )
    
    print(f"\nTotal dataset:")
    print(f"  Slides:  {total_slides}")
    print(f"  Patches: {total_patches:,}")
    
    print(f"\nSlide distribution:")
    print(f"  Train: {len(train_df):3d} ({len(train_df)/total_slides*100:5.1f}%)")
    print(f"  Val:   {len(val_df):3d} ({len(val_df)/total_slides*100:5.1f}%)")
    print(f"  Test:  {len(test_df):3d} ({len(test_df)/total_slides*100:5.1f}%)")
    
    print(f"\nPatch distribution:")
    train_patches = train_df['num_patches'].sum()
    val_patches = val_df['num_patches'].sum()
    test_patches = test_df['num_patches'].sum()
    print(f"  Train: {train_patches:6,} ({train_patches/total_patches*100:5.1f}%)")
    print(f"  Val:   {val_patches:6,} ({val_patches/total_patches*100:5.1f}%)")
    print(f"  Test:  {test_patches:6,} ({test_patches/total_patches*100:5.1f}%)")
    
    # Class balance check
    print(f"\nClass balance (slides):")
    for label in sorted(train_df['label'].unique()):
        train_n = (train_df['label'] == label).sum()
        val_n = (val_df['label'] == label).sum()
        test_n = (test_df['label'] == label).sum()
        total_n = train_n + val_n + test_n
        
        print(f"  Class {label}:")
        print(f"    Train: {train_n:3d} ({train_n/total_n*100:5.1f}%)")
        print(f"    Val:   {val_n:3d} ({val_n/total_n*100:5.1f}%)")
        print(f"    Test:  {test_n:3d} ({test_n/total_n*100:5.1f}%)")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits for Zarr datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--zarr-dirs",
        type=str,
        nargs='+',
        help="Directories containing .zarr archives (e.g., zarr_norm zarr_raw)"
    )
    parser.add_argument(
        "--zarr-dir",
        type=str,
        help="Single directory containing .zarr archives (deprecated, use --zarr-dirs)"
    )
    parser.add_argument(
        "--tcga-csv",
        type=str,
        default="data/TCGA_BRCA_Filtered/HER2_TCGA_clean.csv",
        help="Path to TCGA clinical data CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for manifest CSV files"
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.15,
        help="Test set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=2,
        help="Minimum slides per class per split (warns if violated)"
    )
    parser.add_argument(
        "--existing-manifest",
        type=str,
        help="Use existing manifest CSV instead of scanning directory"
    )
    
    args = parser.parse_args()
    
    # Handle zarr directories
    if args.zarr_dirs:
        zarr_directories = [Path(d) for d in args.zarr_dirs]
    elif args.zarr_dir:
        zarr_directories = [Path(args.zarr_dir)]
    else:
        parser.error("Must specify either --zarr-dirs or --zarr-dir")
    
    print("=" * 60)
    print("Creating Zarr Dataset Splits")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Zarr directories: {', '.join(str(d) for d in zarr_directories)}")
    print(f"  TCGA clinical data: {args.tcga_csv}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Split ratios: train={args.train}, val={args.val}, test={args.test}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Load or scan data
    if args.existing_manifest:
        print(f"Loading existing manifest: {args.existing_manifest}")
        metadata_df = pd.read_csv(args.existing_manifest)
        required_cols = {'zarr_path', 'slide_id', 'label', 'num_patches'}
        if not required_cols.issubset(metadata_df.columns):
            raise ValueError(f"Manifest must have columns: {required_cols}")
    else:
        # Load TCGA clinical labels
        tcga_label_map = load_tcga_labels(Path(args.tcga_csv))
        
        # Scan all zarr directories
        all_dfs = []
        for zarr_dir in zarr_directories:
            df = scan_zarr_directory(zarr_dir, tcga_label_map)
            all_dfs.append(df)
        
        # Combine all dataframes
        metadata_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n✓ Combined {len(metadata_df)} slides from {len(zarr_directories)} directories")
    
    # Create splits
    train_df, val_df, test_df = create_stratified_splits(
        metadata_df,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_state=args.seed,
        min_samples_per_class=args.min_samples_per_class
    )
    
    # Save manifests
    save_manifests(train_df, val_df, test_df, Path(args.output_dir))
    
    # Print statistics
    print_statistics(train_df, val_df, test_df)
    
    print("\n✓ Split creation complete!")
    print("\nNext steps:")
    print(f"  1. Verify manifests in {args.output_dir}/")
    print(f"  2. Run training: ./train.sh")


if __name__ == "__main__":
    main()
