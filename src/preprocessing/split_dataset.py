import yaml
import pandas as pd
from tqdm import tqdm
from IPython import display

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def split_dataset(metadata_csv, train_ratio, val_ratio, test_ratio, export_dir):
    
    met_df = pd.read_csv(metadata_csv)
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    # Shuffle the dataset
    tqdm.pandas(desc="Shuffling dataset")
    shuffled_df = met_df.sample(frac=1, random_state=42).progress_reset_index(drop=True)
    
    # Calculate split indices
    total = len(shuffled_df)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split the dataset
    train_df = shuffled_df.iloc[:train_end]
    val_df = shuffled_df.iloc[train_end:val_end]
    test_df = shuffled_df.iloc[val_end:]
    
    # Export to CSV
    train_df.to_csv(f'{export_dir}/train_split.csv', index=False)
    val_df.to_csv(f'{export_dir}/val_split.csv', index=False)
    test_df.to_csv(f'{export_dir}/test_split.csv', index=False)
    
    # Print resulting DataFrames
    print("Train DataFrame:")
    display(train_df.head())
    print("\nValidation DataFrame:")
    display(val_df.head())
    print("\nTest DataFrame:")
    display(test_df.head())
    
    return train_df, val_df, test_df
