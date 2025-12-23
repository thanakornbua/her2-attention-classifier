"""
Dataset for semantic segmentation (U-Net training).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import cv2

try:
    from cucim import CuImage
    USE_CUCIM = True
except ImportError:
    USE_CUCIM = False
    import openslide


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
        # Normalize mask: 255 -> 1 (tumor), 0 -> 0 (background)
        mask_patch = (mask_patch > 127).astype(np.uint8)
        mask_t = torch.from_numpy(mask_patch).long().unsqueeze(0)
        
        if self.transform is not None:
            img_t = self.transform(img_t)
        
        return img_t, mask_t


class WSISegmentationDataset(Dataset):
    """
    On-the-fly WSI tile extraction for U-Net training.
    
    Memory-efficient approach:
    - Opens WSI only when needed
    - Extracts tiles on-demand
    - No pre-tiling required
    - Automatic memory cleanup
    
    Expected structure:
        wsi_dir/
        ├── Her2Neg_Case_01.svs
        ├── Her2Pos_Case_01.svs
        └── ...
        
        mask_dir/
        ├── Her2Neg_Case_01.npy  # [H, W] uint8
        ├── Her2Pos_Case_01.npy
        └── ...
    """
    
    def __init__(
        self,
        wsi_paths: List[str],
        mask_paths: List[str],
        patch_size: int = 256,
        stride: int = 256,
        level: int = 0,
        tissue_threshold: float = 0.1,
        transform=None,
        use_cucim: bool = None
    ):
        """
        Args:
            wsi_paths: List of WSI file paths
            mask_paths: List of corresponding mask file paths (numpy arrays)
            patch_size: Patch size for tiling
            stride: Stride for tiling
            level: Pyramid level to extract from (0 = highest resolution)
            tissue_threshold: Minimum tissue ratio to include patch
            transform: Optional transforms
            use_cucim: Use cuCIM (auto-detect if None)
        """
        self.patch_size = patch_size
        self.stride = stride
        self.level = level
        self.tissue_threshold = tissue_threshold
        self.transform = transform
        self.use_cucim = USE_CUCIM if use_cucim is None else use_cucim
        
        # Build patch index without loading images
        self.patches = []
        print(f"Building patch index from {len(wsi_paths)} WSI files...")
        
        for wsi_path, mask_path in zip(wsi_paths, mask_paths):
            wsi_path = Path(wsi_path)
            mask_path = Path(mask_path)
            
            if not wsi_path.exists():
                print(f"  Warning: WSI not found: {wsi_path}")
                continue
            if not mask_path.exists():
                print(f"  Warning: Mask not found: {mask_path}")
                continue
            
            # Get dimensions WITHOUT loading the whole image
            if self.use_cucim:
                with CuImage(str(wsi_path)) as slide:
                    w, h = slide.resolutions['level_dimensions'][level]
            else:
                with openslide.OpenSlide(str(wsi_path)) as slide:
                    w, h = slide.level_dimensions[level]
            
            # Generate patch coordinates
            n_patches = 0
            for y in range(0, max(1, h - patch_size + 1), stride):
                for x in range(0, max(1, w - patch_size + 1), stride):
                    self.patches.append({
                        'wsi_path': str(wsi_path),
                        'mask_path': str(mask_path),
                        'x': x,
                        'y': y,
                        'level': level,
                        'w': w,
                        'h': h,
                    })
                    n_patches += 1
            
            print(f"  {wsi_path.name}: {w}x{h} → {n_patches} patches")
        
        print(f"Total patches: {len(self.patches)}")
    
    def __len__(self):
        return len(self.patches)
    
    def _read_region_cucim(self, wsi_path: str, x: int, y: int, size: int, level: int) -> np.ndarray:
        """Read region using cuCIM (GPU-accelerated)."""
        with CuImage(wsi_path) as slide:
            # cuCIM uses (x, y, width, height, level)
            region = slide.read_region(
                location=(x, y),
                size=(size, size),
                level=level
            )
            return np.array(region)
    
    def _read_region_openslide(self, wsi_path: str, x: int, y: int, size: int, level: int) -> np.ndarray:
        """Read region using OpenSlide."""
        with openslide.OpenSlide(wsi_path) as slide:
            # OpenSlide level 0 coordinates need scaling
            scale = slide.level_downsamples[level]
            x0 = int(x * scale)
            y0 = int(y * scale)
            
            region = slide.read_region(
                location=(x0, y0),
                level=level,
                size=(size, size)
            )
            return np.array(region.convert('RGB'))
    
    def _has_sufficient_tissue(self, img: np.ndarray) -> bool:
        """Check if patch contains sufficient tissue (not background)."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Otsu thresholding to separate tissue from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate tissue ratio
        tissue_ratio = np.sum(binary < 128) / binary.size
        
        return tissue_ratio >= self.tissue_threshold
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patch on-the-fly from WSI.
        
        Memory management:
        - Opens WSI in context manager (auto-closes)
        - Extracts only requested patch
        - Immediately releases file handle
        
        Returns:
            image: [C, H, W] float32 tensor
            mask: [1, H, W] long tensor
        """
        patch_info = self.patches[idx]
        
        # Extract patch from WSI (memory-efficient - opens and closes immediately)
        if self.use_cucim:
            img_patch = self._read_region_cucim(
                patch_info['wsi_path'],
                patch_info['x'],
                patch_info['y'],
                self.patch_size,
                patch_info['level']
            )
        else:
            img_patch = self._read_region_openslide(
                patch_info['wsi_path'],
                patch_info['x'],
                patch_info['y'],
                self.patch_size,
                patch_info['level']
            )
        
        # Load corresponding mask patch (mask may be downsampled vs WSI)
        mask_full = np.load(patch_info['mask_path'])
        if mask_full.size == 0:
            # Guard against corrupt/empty masks; return empty tile
            mask_full = np.zeros((1, 1), dtype=np.uint8)

        # Compute scale between mask and WSI dimensions
        wsi_w, wsi_h = patch_info['w'], patch_info['h']
        mask_h, mask_w = mask_full.shape[:2]
        scale_x = mask_w / max(1, wsi_w)
        scale_y = mask_h / max(1, wsi_h)

        # Map WSI coords to mask coords
        y_wsi, x_wsi = patch_info['y'], patch_info['x']
        y_m = int(y_wsi * scale_y)
        x_m = int(x_wsi * scale_x)
        ph_m = int(self.patch_size * scale_y)
        pw_m = int(self.patch_size * scale_x)

        mask_patch = mask_full[y_m:y_m+ph_m, x_m:x_m+pw_m]

        # If slice is empty (e.g., coords outside mask), fill with zeros
        if mask_patch.size == 0 or mask_patch.shape[0] == 0 or mask_patch.shape[1] == 0:
            mask_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        # Resize mask patch back to patch_size if scaling changed dims
        elif mask_patch.shape[0] != self.patch_size or mask_patch.shape[1] != self.patch_size:
            mask_patch = cv2.resize(
                mask_patch,
                (self.patch_size, self.patch_size),
                interpolation=cv2.INTER_NEAREST,
            )

        # Handle edge cases (incomplete image patches)
        if img_patch.shape[0] == 0 or img_patch.shape[1] == 0:
            img_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        elif img_patch.shape[0] < self.patch_size or img_patch.shape[1] < self.patch_size:
            padded_img = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            h, w = img_patch.shape[:2]
            padded_img[:h, :w] = img_patch
            img_patch = padded_img
        
        # Convert to tensors
        img_t = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
        # Normalize mask: 255 -> 1 (tumor), 0 -> 0 (background)
        mask_patch = (mask_patch > 127).astype(np.uint8)
        mask_t = torch.from_numpy(mask_patch).long().unsqueeze(0)
        
        if self.transform is not None:
            img_t = self.transform(img_t)
        
        return img_t, mask_t
