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
    Loads pre-extracted patches using EXPLICIT ZARR ROUTING.
    
    Each row in patch_metadata contains:
    - zarr_path: Full path to the Zarr file containing this patch
    - zarr_index: The exact index within that Zarr file
    - her2_status: The label for this patch
    
    This eliminates ambiguity and prevents silent data corruption from index misalignment.
    
    Args:
        zarr_root: DEPRECATED (can pass None). Use patch_metadata['zarr_path'] instead.
        indices: Array of indices into patch_metadata DataFrame
        zarr_root_secondary: DEPRECATED (can pass None). Use patch_metadata['zarr_path'] instead.
        patch_metadata: DataFrame with ['zarr_path', 'zarr_index', 'her2_status'] columns
        return_metadata: If True, __getitem__ also returns slide_name and case_name.
    """
    def __init__(self, zarr_root: str, indices, zarr_root_secondary=None, patch_metadata=None, return_metadata: bool = False):
        super().__init__()
        self.return_metadata = return_metadata
        
        # NEW: Explicit routing mode (recommended)
        if patch_metadata is not None and 'zarr_path' in patch_metadata.columns:
            self.metadata = patch_metadata
            self.indices = np.array(list(indices), dtype=np.int64)
            
            # Validate metadata structure
            required_cols = ['zarr_path', 'zarr_index', 'her2_status']
            for col in required_cols:
                if col not in self.metadata.columns:
                    raise ValueError(f"patch_metadata missing required column: {col}")
            
            # Open all unique Zarr files and cache them
            self.zarr_cache = {}
            unique_zarrs = self.metadata['zarr_path'].unique()
            print(f"Opening {len(unique_zarrs)} Zarr archive(s) for dataset...")
            for zarr_path in unique_zarrs:
                self.zarr_cache[zarr_path] = zarr.open(str(zarr_path), mode="r")
                print(f"  Cached: {zarr_path} (shape={self.zarr_cache[zarr_path].shape})")
            
            self.use_explicit_routing = True
            print(f"✓ Dataset initialized with EXPLICIT routing ({len(self)} samples)")
            
        # LEGACY: Old dual-Zarr mode (deprecated but kept for backwards compatibility)
        elif zarr_root is not None:
            print("⚠️  Using LEGACY dual-Zarr mode. Consider updating to explicit routing.")
            self.root_primary = zarr.open(zarr_root, mode="r")
            self.patches_primary = self.root_primary
            self.zarr_root = zarr_root
            
            self.use_dual_zarr = zarr_root_secondary is not None
            if self.use_dual_zarr:
                self.root_secondary = zarr.open(zarr_root_secondary, mode="r")
                self.patches_secondary = self.root_secondary
                self.zarr_root_secondary = zarr_root_secondary
                
                if patch_metadata is None:
                    raise ValueError("patch_metadata required when using dual Zarr mode")
                self.metadata = patch_metadata
                self.primary_size = self.patches_primary.shape[0]
            else:
                self.root_secondary = None
                self.patches_secondary = None
                self.metadata = patch_metadata
                self.primary_size = self.patches_primary.shape[0]
            
            self.indices = np.array(list(indices), dtype=np.int64)
            self.use_explicit_routing = False
            
            # Validate patch dimensions
            assert self.patches_primary.ndim == 4 and self.patches_primary.shape[-1] == 3, \
                f"patches must be (N, H, W, 3), got {self.patches_primary.shape}"
        else:
            raise ValueError("Must provide either zarr_root or patch_metadata with zarr_path column")

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        idx = int(self.indices[i])
        
        # NEW: Explicit routing - each row knows exactly where it lives
        if self.use_explicit_routing:
            row = self.metadata.iloc[idx]
            zarr_path = row['zarr_path']
            zarr_index = int(row['zarr_index'])
            label = int(row['her2_status'])
            slide_name = row['slide_name'] if 'slide_name' in row else None
            case_name = row['case_name'] if 'case_name' in row else None
            
            # Load from the correct Zarr file at the correct index
            zarr_array = self.zarr_cache[zarr_path]
            patch_u8 = zarr_array[zarr_index]
            
        # LEGACY: Old dual-Zarr mode
        else:
            if self.use_dual_zarr:
                row = self.metadata.iloc[idx]
                source = row['source']
                patch_global_index = int(row['patch_global_index'])
                slide_name = row['slide_name'] if 'slide_name' in row else None
                case_name = row['case_name'] if 'case_name' in row else None
                
                if source == 'normalized':
                    patch_u8 = self.patches_primary[patch_global_index]
                else:
                    patch_u8 = self.patches_secondary[patch_global_index]
            else:
                patch_u8 = self.patches_primary[idx]
                slide_name = None
                case_name = None
            
            if self.metadata is not None:
                label = int(self.metadata.iloc[idx]['her2_status'])
            else:
                if 'labels' in self.root_primary:
                    label = int(self.root_primary['labels'][idx])
                else:
                    label = 0
        
        # Preprocess: normalize to ImageNet stats
        img = patch_u8.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # C,H,W
        img_t = torch.from_numpy(img).float()
        
        cls_t = torch.tensor(label, dtype=torch.long)
        loc_t = torch.tensor([0.25, 0.25, 0.5, 0.5], dtype=torch.float32)
        if self.return_metadata:
            return img_t, cls_t, loc_t, slide_name, case_name
        return img_t, cls_t, loc_t