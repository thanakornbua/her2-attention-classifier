"""
ZarrPatchDataset: PyTorch Dataset for Zarr-Stored Histopathology Patches

This module provides a memory-efficient dataset loader for pre-extracted histopathology
patches stored in Zarr format. It supports loading normalized or raw patches with
associated labels for multi-task classification and localization training.

Key Features:
    - Memory-efficient lazy loading via Zarr arrays
    - ImageNet-compatible preprocessing (normalization and channel ordering)
    - Support for train/val/test splits via index subsetting
    - Mock localization targets for multi-task learning

Expected Zarr Structure:
    zarr_root/
        patches/    # (N, H, W, 3) uint8 array of RGB patches
        labels/     # (N,) int64 array of class labels

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
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class ZarrPatchDataset(Dataset):
    """
    Loads pre-extracted, Macenko-normalized patches from a Zarr archive.
    Expects Zarr groups:
      - patches: (N, 256, 256, 3) uint8
      - labels:  (N,) int64 class indices
    """
    def __init__(self, zarr_root: str, indices):
        super().__init__()
        self.root = zarr.open(zarr_root, mode="r")
        self.patches = self.root["patches"]
        self.labels = self.root["labels"]
        self.indices = np.array(list(indices), dtype=np.int64)
        assert self.patches.ndim == 4 and self.patches.shape[-1] == 3, "patches must be (N, H, W, 3)"
        assert self.labels.shape[0] == self.patches.shape[0], "labels length must match patches"

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        idx = int(self.indices[i])
        patch_u8 = self.patches[idx]  # (H, W, 3) uint8
        label = int(self.labels[idx])

        img = patch_u8.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # C,H,W
        img_t = torch.from_numpy(img).float()

        cls_t = torch.tensor(label, dtype=torch.long)
        loc_t = torch.tensor([0.25, 0.25, 0.5, 0.5], dtype=torch.float32)
        return img_t, cls_t, loc_t