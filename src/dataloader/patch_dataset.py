"""
Patch Dataset for HER2 Classification

Supports two modes:
1. Legacy: Individual PNG files (from outputs/patches/)
2. Zarr: Zarr archives (one per slide, memory-efficient)
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import zarr
import pandas as pd
from typing import Optional, List


class PatchDataset(Dataset):
    """
    Dataset for loading patches from either individual files or Zarr archives.
    
    Args:
        patch_paths: List of patch file paths (legacy mode) or None (Zarr mode)
        labels: List of labels (legacy mode) or None (Zarr mode)
        transform: Optional torchvision transforms
        zarr_manifest_csv: Path to CSV with columns: zarr_path, slide_id, label, num_patches
        mode: 'files' for individual files, 'zarr' for Zarr archives
    """
    
    def __init__(self, 
                 patch_paths: Optional[List[str]] = None,
                 labels: Optional[List[int]] = None,
                 transform=None,
                 zarr_manifest_csv: Optional[str] = None,
                 mode: str = 'files'):
        
        self.transform = transform
        self.mode = mode
        
        if mode == 'files':
            # Legacy mode: individual patch files
            if patch_paths is None or labels is None:
                raise ValueError("patch_paths and labels required for 'files' mode")
            self.patch_paths = patch_paths
            self.labels = labels
            self.zarr_handles = None
            
        elif mode == 'zarr':
            # Zarr mode: read from manifest
            if zarr_manifest_csv is None:
                raise ValueError("zarr_manifest_csv required for 'zarr' mode")
            
            self._init_zarr_mode(zarr_manifest_csv)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'files' or 'zarr'")
    
    def _init_zarr_mode(self, manifest_csv: str):
        """Initialize Zarr mode by reading manifest and opening Zarr handles."""
        df = pd.read_csv(manifest_csv)
        
        # Build index: for each patch, store (zarr_idx, patch_idx_within_zarr, label)
        self.patch_index = []
        self.zarr_handles = []
        self.zarr_paths = []
        
        for _, row in df.iterrows():
            zarr_path = row['zarr_path']
            label = int(row['label'])
            num_patches = int(row['num_patches'])
            
            # Open Zarr in read-only mode
            z = zarr.open(str(zarr_path), mode='r')
            zarr_idx = len(self.zarr_handles)
            self.zarr_handles.append(z)
            self.zarr_paths.append(zarr_path)
            
            # Add all patches from this Zarr to index
            for patch_idx in range(num_patches):
                self.patch_index.append((zarr_idx, patch_idx, label))
        
        print(f"Loaded {len(self.zarr_handles)} Zarr files with {len(self.patch_index)} total patches")
    
    def __len__(self):
        if self.mode == 'files':
            return len(self.patch_paths)
        else:  # zarr
            return len(self.patch_index)
    
    def __getitem__(self, idx):
        if self.mode == 'files':
            return self._get_file_item(idx)
        else:  # zarr
            return self._get_zarr_item(idx)
    
    def _get_file_item(self, idx):
        """Load patch from individual file (legacy mode)."""
        img_path = self.patch_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return black image on error
            print(f"Warning: Failed to load {img_path}: {e}")
            img = Image.new('RGB', (256, 256), color=(0, 0, 0))
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def _get_zarr_item(self, idx):
        """Load patch from Zarr archive."""
        zarr_idx, patch_idx, label = self.patch_index[idx]
        
        # Get Zarr handle
        z = self.zarr_handles[zarr_idx]
        
        # Load patch (already normalized during preprocessing)
        try:
            patch = z['patches'][patch_idx]  # uint8 array (H, W, 3)
            img = Image.fromarray(patch)
        except Exception as e:
            # Return black image on error
            print(f"Warning: Failed to load patch {patch_idx} from {self.zarr_paths[zarr_idx]}: {e}")
            img = Image.new('RGB', (256, 256), color=(0, 0, 0))
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_slide_patches(self, slide_id: str):
        """
        Get all patches for a specific slide (Zarr mode only).
        
        Args:
            slide_id: slide identifier
            
        Returns:
            Tuple of (patches, labels) as numpy arrays
        """
        if self.mode != 'zarr':
            raise ValueError("get_slide_patches only supported in Zarr mode")
        
        # Find Zarr for this slide
        for zarr_idx, z in enumerate(self.zarr_handles):
            meta_path = Path(self.zarr_paths[zarr_idx]) / 'meta.json'
            if meta_path.exists():
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    if meta['slide_id'] == slide_id:
                        patches = z['patches'][:]
                        labels = z['labels'][:]
                        return patches, labels
        
        raise ValueError(f"Slide {slide_id} not found in dataset")


class ZarrPatchDataset(Dataset):
    """
    Memory-efficient dataset for loading patches from Zarr archives.
    Optimized for training with minimal memory overhead.
    """
    
    def __init__(self, zarr_paths: List[str], transform=None):
        """
        Args:
            zarr_paths: List of paths to .zarr directories
            transform: Optional torchvision transforms
        """
        self.transform = transform
        self.zarr_paths = zarr_paths
        self.zarr_handles = []
        self.patch_index = []  # (zarr_idx, patch_idx, label)
        
        # Open all Zarr files and build index
        for zarr_path in zarr_paths:
            z = zarr.open(str(zarr_path), mode='r')
            zarr_idx = len(self.zarr_handles)
            self.zarr_handles.append(z)
            
            num_patches = z['patches'].shape[0]
            labels = z['labels'][:]
            
            for patch_idx in range(num_patches):
                self.patch_index.append((zarr_idx, patch_idx, int(labels[patch_idx])))
        
    def __len__(self):
        return len(self.patch_index)
    
    def __getitem__(self, idx):
        zarr_idx, patch_idx, label = self.patch_index[idx]
        z = self.zarr_handles[zarr_idx]
        
        # Load patch
        patch = z['patches'][patch_idx]
        img = Image.fromarray(patch)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
