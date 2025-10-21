import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image


def extract_patches(
    wsi_slide, 
    mask: Optional[np.ndarray] = None, 
    size: int = 512, 
    stride: int = 512,
    tissue_threshold: float = 0.1,
    level: int = 0
) -> List[Dict]:
    """
    Extract image patches from Whole Slide Images (WSI).
    Supports both ROI-only extraction (with mask) and full tissue extraction.
    
    Args:
        wsi_slide: OpenSlide object of the loaded WSI
        mask: Optional binary mask (numpy array) indicating regions to extract from.
              - If provided: Extract only from ROI where mask > 0 (ROI-only mode)
              - If None: Extract from entire WSI (full tissue mode)
              Shape should correspond to WSI dimensions (may be downsampled)
        size: Patch size in pixels (default: 512x512)
        stride: Step size between patch centers in pixels (default: 512)
                - stride = size: No overlap
                - stride < size: Overlapping patches
                - stride > size: Gap between patches
        tissue_threshold: Minimum ratio of tissue/ROI pixels required in mask 
                         to keep a patch (0.0 to 1.0). Default: 0.1 (10%)
        level: Pyramid level to extract from (0 = highest resolution)
    
    Returns:
        List of dictionaries, each containing:
            - 'patch': PIL.Image object of the extracted patch
            - 'x': x-coordinate of top-left corner in level-0 coordinates
            - 'y': y-coordinate of top-left corner in level-0 coordinates
            - 'level': pyramid level the patch was extracted from
    
    Example:
        >>> import openslide
        >>> wsi = openslide.open_slide('slide.svs')
        >>> 
        >>> # Full tissue mode (no mask)
        >>> patches = extract_patches(wsi, mask=None, size=512, stride=512)
        >>> 
        >>> # ROI-only mode (with tissue mask)
        >>> tissue_mask = create_tissue_mask(wsi)  # Binary mask
        >>> patches = extract_patches(wsi, mask=tissue_mask, size=512, stride=256)
        >>> 
        >>> # Save extracted patches
        >>> for i, p in enumerate(patches):
        >>>     p['patch'].save(f'patch_{i}.png')
    """
    patches = []
    
    # Get WSI dimensions at the specified level
    level_dimensions = wsi_slide.level_dimensions[level]
    width, height = level_dimensions
    
    # Get dimensions at level 0 for coordinate tracking
    level0_width, level0_height = wsi_slide.level_dimensions[0]
    
    # Calculate downsample factor for this level
    level_downsample = wsi_slide.level_downsamples[level]
    
    # Calculate downsample factors between mask and WSI if mask exists
    if mask is not None:
        mask_height, mask_width = mask.shape
        # Downsample from level-0 to mask resolution
        downsample_x = level0_width / mask_width
        downsample_y = level0_height / mask_height
    
    # Iterate through WSI in a sliding window pattern
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            
            # Convert current position to level-0 coordinates
            x_level0 = int(x * level_downsample)
            y_level0 = int(y * level_downsample)
            
            # If mask is provided, check if patch contains sufficient tissue/ROI
            if mask is not None:
                # Convert level-0 coordinates to mask coordinates
                mask_x_start = int(x_level0 / downsample_x)
                mask_y_start = int(y_level0 / downsample_y)
                mask_x_end = int((x_level0 + size * level_downsample) / downsample_x)
                mask_y_end = int((y_level0 + size * level_downsample) / downsample_y)
                
                # Ensure coordinates are within mask bounds
                mask_x_start = max(0, min(mask_x_start, mask_width - 1))
                mask_y_start = max(0, min(mask_y_start, mask_height - 1))
                mask_x_end = max(0, min(mask_x_end, mask_width))
                mask_y_end = max(0, min(mask_y_end, mask_height))
                
                # Extract corresponding mask region
                mask_region = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                
                # Skip if mask region is empty
                if mask_region.size == 0:
                    continue
                
                # Calculate tissue/ROI ratio
                tissue_ratio = np.sum(mask_region > 0) / mask_region.size
                
                # Skip patch if insufficient tissue/ROI content
                if tissue_ratio < tissue_threshold:
                    continue
            
            # Extract patch from WSI
            # read_region uses level-0 coordinates regardless of level parameter
            patch = wsi_slide.read_region(
                location=(x_level0, y_level0),
                level=level,
                size=(size, size)
            )
            
            # Convert RGBA to RGB (OpenSlide returns RGBA)
            if patch.mode == 'RGBA':
                patch = patch.convert('RGB')
            
            # Store patch with metadata
            patches.append({
                'patch': patch,
                'x': x_level0,
                'y': y_level0,
                'level': level
            })
    
    return patches