"""
ZarrPatchDataset: PyTorch Dataset for Zarr-Stored Histopathology Patches

This module provides a memory-efficient dataset loader for pre-extracted histopathology
patches stored in Zarr format. It supports loading normalized or raw patches with
associated labels for multi-task classification and localization training.

Key Features:
    - Memory-efficient lazy loading via Zarr arrays
    - ImageNet-compatible preprocessing (normalization and channel ordering)
    - Support for train/val/test splits via index subsetting
    - Support for dual Zarr archives (normalized + raw) with automatic routing
    - Mock localization targets for multi-task learning

Expected Zarr Structure:
    zarr_root/
        (N, H, W, 3) uint8 array of RGB patches (flat structure)

Usage Example:
    >>> import zarr
    >>> from torch.utils.data import DataLoader
    >>> 
    >>> # Load dataset with specific indices
    >>> train_indices = [0, 1, 2, 5, 10]
    >>> dataset = ZarrPatchDataset('path/to/patches.zarr', indices=train_indices)
    >>> loader = DataLoader(dataset, batch_size=16, shuffle=True)
    >>> 
    >>> # Iterate
    >>> for images, labels, boxes in loader:
    >>>     # images: (B, 3, H, W) float32, ImageNet-normalized
    >>>     # labels: (B,) int64 class indices
    >>>     # boxes: (B, 4) float32 mock localization targets
    >>>     pass

Author: T. Buathongtanakarn et al. (2025)
"""

import zarr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class ZarrPatchDataset(Dataset):
    """
    Loads pre-extracted patches from one or two Zarr archives.
    
    Supports two usage modes:
    1. Single Zarr: zarr_root only (traditional mode)
    2. Dual Zarr: zarr_root (normalized) + zarr_root_secondary (raw) with patch_metadata for routing
    
    Args:
        zarr_root: Path to primary Zarr archive (or only Zarr if not using dual mode)
        indices: Array of indices to load
        zarr_root_secondary: Optional path to secondary Zarr archive (for raw patches)
        patch_metadata: Optional DataFrame with 'source' column for routing between Zarrs
    """
    def __init__(self, zarr_root: str, indices, zarr_root_secondary=None, patch_metadata=None):
        super().__init__()
        
        # Open primary Zarr
        self.root_primary = zarr.open(zarr_root, mode="r")
        self.patches_primary = self.root_primary
        self.zarr_root = zarr_root
        
        # Setup for dual Zarr mode
        self.use_dual_zarr = zarr_root_secondary is not None
        if self.use_dual_zarr:
            self.root_secondary = zarr.open(zarr_root_secondary, mode="r")
            self.patches_secondary = self.root_secondary
            self.zarr_root_secondary = zarr_root_secondary
            
            # Require metadata for routing
            if patch_metadata is None:
                raise ValueError("patch_metadata required when using dual Zarr mode")
            self.metadata = patch_metadata
            
            # Calculate cumulative sizes for index mapping
            self.primary_size = self.patches_primary.shape[0]
        else:
            self.root_secondary = None
            self.patches_secondary = None
            self.metadata = None
            self.primary_size = self.patches_primary.shape[0]
        
        self.indices = np.array(list(indices), dtype=np.int64)
        
        # Validate patch dimensions
        assert self.patches_primary.ndim == 4 and self.patches_primary.shape[-1] == 3, \
            f"patches must be (N, H, W, 3), got {self.patches_primary.shape}"

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        idx = int(self.indices[i])
        
        # Route to correct Zarr archive and map index
        if self.use_dual_zarr:
            # Get metadata for this global index
            source = self.metadata.iloc[idx]['source']
            patch_global_index = int(self.metadata.iloc[idx]['patch_global_index'])
            
            # Select Zarr and get patch
            if source == 'normalized':
                patch_u8 = self.patches_primary[patch_global_index]
            else:  # 'raw'
                patch_u8 = self.patches_secondary[patch_global_index]
        else:
            # Single Zarr mode: direct indexing
            patch_u8 = self.patches_primary[idx]
        
        # Get label from metadata if available, otherwise from Zarr
        if self.metadata is not None:
            label = int(self.metadata.iloc[idx]['her2_status'])
        else:
            # Fallback: try to get from Zarr labels
            if 'labels' in self.root_primary:
                label = int(self.root_primary['labels'][idx])
            else:
                label = 0  # Default
        
        # Preprocess: normalize to ImageNet stats
        img = patch_u8.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # C,H,W
        img_t = torch.from_numpy(img).float()
        
        cls_t = torch.tensor(label, dtype=torch.long)
        loc_t = torch.tensor([0.25, 0.25, 0.5, 0.5], dtype=torch.float32)
        return img_t, cls_t, loc_t