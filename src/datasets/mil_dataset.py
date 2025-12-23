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
        ├── slide1_roi_confidence.npy  # [num_patches] (optional)
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
        slide_ids: Optional[np.ndarray] = None,
        use_roi_confidence: bool = False
    ):
        """
        Args:
            features_dir: Directory containing .npy feature files
            labels_csv: Path to CSV with slide IDs and labels
            slide_ids: Optional array of slide IDs to subset dataset
            use_roi_confidence: Whether to load and use ROI confidence weights
        """
        self.features_dir = Path(features_dir)
        self.use_roi_confidence = use_roi_confidence
        
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
            roi_confidence: ROI confidence weights [num_patches] (if use_roi_confidence=True, else None)
        """
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]
        
        # Load features
        feature_path = self.features_dir / f"{slide_id}.npy"
        features = np.load(feature_path)
        
        # Load ROI confidence if available
        roi_confidence = None
        if self.use_roi_confidence:
            roi_path = self.features_dir / f"{slide_id}_roi_confidence.npy"
            if roi_path.exists():
                roi_confidence = np.load(roi_path)  # [num_patches]
            else:
                # Default to uniform confidence if file not found
                roi_confidence = np.ones(features.shape[0])
        
        # Convert to tensors
        features_t = torch.from_numpy(features).float()
        label_t = torch.tensor(label, dtype=torch.long)
        
        if roi_confidence is not None:
            roi_confidence_t = torch.from_numpy(roi_confidence).float()
            return features_t, label_t, slide_id, roi_confidence_t
        
        return features_t, label_t, slide_id, None


def collate_slide_features(batch):
    """
    Custom collate function for variable-length slide features.
    
    Pads features to max length in batch.
    
    Args:
        batch: List of (features, label, slide_id, roi_confidence) tuples
        
    Returns:
        features_padded: [batch_size, max_patches, feature_dim]
        labels: [batch_size]
        slide_ids: List of slide identifiers
        lengths: [batch_size] - actual number of patches per slide
        roi_confidence_padded: [batch_size, max_patches] or None
    """
    features_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    slide_ids = [item[2] for item in batch]
    
    # Check if ROI confidence is provided
    has_roi = len(batch[0]) > 3 and batch[0][3] is not None
    roi_list = [item[3] for item in batch] if has_roi else None
    
    # Get max length
    lengths = torch.tensor([f.size(0) for f in features_list])
    max_len = lengths.max().item()
    feature_dim = features_list[0].size(1)
    
    # Pad features
    features_padded = torch.zeros(len(batch), max_len, feature_dim)
    for i, features in enumerate(features_list):
        features_padded[i, :features.size(0)] = features
    
    # Pad ROI confidence if available
    roi_confidence_padded = None
    if has_roi and roi_list[0] is not None:
        roi_confidence_padded = torch.zeros(len(batch), max_len)
        for i, roi in enumerate(roi_list):
            roi_confidence_padded[i, :roi.size(0)] = roi
    
    return features_padded, labels, slide_ids, lengths, roi_confidence_padded
