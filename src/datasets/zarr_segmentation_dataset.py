"""
Zarr-based segmentation dataset for U-Net training.

Pre-extracted patches and masks for fast training.
"""

import zarr
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List


class ZarrSegmentationDataset(Dataset):
    """
    Load pre-extracted image patches and masks from Zarr for U-Net training.
    
    Expected Zarr structure:
        patches.zarr/
            images/     # (N, H, W, 3) uint8 RGB patches
            masks/      # (N, H, W) uint8 binary masks (0=background, 255=tumor)
    
    Args:
        zarr_path: Path to Zarr archive containing images/ and masks/
        indices: Optional list of indices to use (for train/val split)
        transform: Optional image transforms
    
    Returns:
        image: (C, H, W) float32 tensor normalized to [0, 1]
        mask: (1, H, W) long tensor with class indices [0, 1]
    """
    
    def __init__(
        self,
        zarr_path: str,
        indices: Optional[List[int]] = None,
        transform=None
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.transform = transform
        
        # Open Zarr archive
        self.root = zarr.open(zarr_path, mode='r')
        
        # Verify structure
        if 'images' not in self.root:
            raise ValueError(f"Zarr archive missing 'images' array: {zarr_path}")
        if 'masks' not in self.root:
            raise ValueError(f"Zarr archive missing 'masks' array: {zarr_path}")
        
        self.images = self.root['images']
        self.masks = self.root['masks']
        
        # Validate shapes
        if self.images.shape[0] != self.masks.shape[0]:
            raise ValueError(f"Shape mismatch: images={self.images.shape}, masks={self.masks.shape}")
        
        # Set indices
        if indices is None:
            self.indices = np.arange(self.images.shape[0])
        else:
            self.indices = np.array(indices, dtype=np.int64)
        
        print(f"✓ ZarrSegmentationDataset loaded: {len(self)} samples")
        print(f"  Images: {self.images.shape} (dtype={self.images.dtype})")
        print(f"  Masks:  {self.masks.shape} (dtype={self.masks.dtype})")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        idx = int(self.indices[i])
        
        # Load from Zarr (lazy loading)
        image = self.images[idx]  # (H, W, 3) uint8
        mask = self.masks[idx]    # (H, W) uint8
        
        # Convert to tensors
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (3, H, W) [0, 1]
        
        # Normalize mask: 255 -> 1 (tumor), 0 -> 0 (background)
        mask_binary = (mask > 127).astype(np.uint8)
        mask_t = torch.from_numpy(mask_binary).long().unsqueeze(0)  # (1, H, W)
        
        if self.transform is not None:
            image_t = self.transform(image_t)
        
        return image_t, mask_t
