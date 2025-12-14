"""
Slide-level dataset for Multiple Instance Learning (MIL).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional


class SlideFeatureDataset(Dataset):
    """
    Dataset for slide-level MIL training using pre-extracted patch features.
    
    Expected structure:
        features_dir/
        ├── slide1.npy  # [num_patches, feature_dim]
        ├── slide2.npy
        └── ...
        
        labels.csv:
        ├── slide_id, label
        ├── slide1, 0
        ├── slide2, 1
        └── ...
    """
    
    def __init__(
        self,
        features_dir: str,
        labels_csv: str,
        slide_ids: Optional[np.ndarray] = None
    ):
        """
        Args:
            features_dir: Directory containing .npy feature files
            labels_csv: Path to CSV with slide IDs and labels
            slide_ids: Optional array of slide IDs to subset dataset
        """
        self.features_dir = Path(features_dir)
        
        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        
        # Subset if requested
        if slide_ids is not None:
            self.labels_df = self.labels_df[
                self.labels_df['slide_id'].isin(slide_ids)
            ].reset_index(drop=True)
        
        self.slide_ids = self.labels_df['slide_id'].values
        self.labels = self.labels_df['label'].values
    
    def __len__(self):
        return len(self.slide_ids)
    
    def __getitem__(self, idx):
        """
        Get slide features and label.
        
        Returns:
            features: Patch features [num_patches, feature_dim]
            label: Slide-level label (int)
            slide_id: Slide identifier (str)
        """
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]
        
        # Load features
        feature_path = self.features_dir / f"{slide_id}.npy"
        features = np.load(feature_path)
        
        # Convert to tensors
        features_t = torch.from_numpy(features).float()
        label_t = torch.tensor(label, dtype=torch.long)
        
        return features_t, label_t, slide_id


def collate_slide_features(batch):
    """
    Custom collate function for variable-length slide features.
    
    Pads features to max length in batch.
    
    Args:
        batch: List of (features, label, slide_id) tuples
        
    Returns:
        features_padded: [batch_size, max_patches, feature_dim]
        labels: [batch_size]
        slide_ids: List of slide identifiers
        lengths: [batch_size] - actual number of patches per slide
    """
    features_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    slide_ids = [item[2] for item in batch]
    
    # Get max length
    lengths = torch.tensor([f.size(0) for f in features_list])
    max_len = lengths.max().item()
    feature_dim = features_list[0].size(1)
    
    # Pad features
    features_padded = torch.zeros(len(batch), max_len, feature_dim)
    for i, features in enumerate(features_list):
        features_padded[i, :features.size(0)] = features
    
    return features_padded, labels, slide_ids, lengths
