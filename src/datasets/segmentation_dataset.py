"""
Dataset for semantic segmentation (U-Net training).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class SegmentationDataset(Dataset):
    """
    Dataset for tumor segmentation using paired images and masks.
    
    Expected structure:
        data_dir/
        ├── images/
        │   ├── slide1.npy  # [H, W, 3] uint8
        │   ├── slide2.npy
        │   └── ...
        └── masks/
            ├── slide1.npy  # [H, W] uint8 (0=background, 1=tumor)
            ├── slide2.npy
            └── ...
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        patch_size: int = 256,
        stride: int = 256,
        transform=None
    ):
        """
        Args:
            images_dir: Directory with image files
            masks_dir: Directory with mask files
            patch_size: Patch size for tiling
            stride: Stride for tiling
            transform: Image transforms
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        
        # Find all image files
        self.image_files = sorted(self.images_dir.glob('*.npy'))
        
        # Build patch list
        self.patches = []
        for img_file in self.image_files:
            mask_file = self.masks_dir / img_file.name
            if not mask_file.exists():
                continue
            
            # Load to get size
            img = np.load(img_file)
            h, w = img.shape[:2]
            
            # Generate patches
            for y in range(0, max(0, h - patch_size + 1), stride):
                for x in range(0, max(0, w - patch_size + 1), stride):
                    self.patches.append({
                        'image': img_file,
                        'mask': mask_file,
                        'x': x,
                        'y': y
                    })
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get patch pair.
        
        Returns:
            image: [C, H, W] float32 normalized to [0, 1]
            mask: [1, H, W] long tensor
        """
        patch_info = self.patches[idx]
        
        # Load patch
        img = np.load(patch_info['image'])
        mask = np.load(patch_info['mask'])
        
        # Extract patch
        x, y = patch_info['x'], patch_info['y']
        img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
        
        # Convert to tensors
        img_t = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask_patch).long().unsqueeze(0)
        
        if self.transform is not None:
            img_t = self.transform(img_t)
        
        return img_t, mask_t
