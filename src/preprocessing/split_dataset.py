"""Split dataset into train/val/test sets with stratification for medical data."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(metadata_csv, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                 export_dir='data/splits', stratify_by='label', random_state=42):
    """
    Split dataset with stratification to preserve label distribution.
    
    Args:
        metadata_csv: Path to metadata CSV
        train_ratio: Training set ratio (default: 0.7)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        export_dir: Output directory for split CSVs
        stratify_by: Column to stratify by (default: 'label' for HER2 grades)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = pd.read_csv(metadata_csv)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Stratified split: preserve label distribution across splits
    stratify_col = df[stratify_by] if stratify_by and stratify_by in df.columns else None
    
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio),
        stratify=stratify_col,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_df[stratify_by] if stratify_by and stratify_by in temp_df.columns else None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        stratify=stratify_temp,
        random_state=random_state
    )
    
    # Export splits
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(export_path / 'train_split.csv', index=False)
    val_df.to_csv(export_path / 'val_split.csv', index=False)
    test_df.to_csv(export_path / 'test_split.csv', index=False)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset with stratification")
    parser.add_argument("metadata_csv", help="Path to metadata CSV")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio")
    parser.add_argument("-o", "--output", default="data/splits", help="Output directory")
    parser.add_argument("--stratify", default="label", help="Column to stratify by")
    args = parser.parse_args()
    
    split_dataset(args.metadata_csv, args.train, args.val, args.test, args.output, args.stratify)

