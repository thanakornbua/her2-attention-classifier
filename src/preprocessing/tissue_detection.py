import numpy as np
import cv2
from typing import Tuple, Optional


def otsu_thresholding(wsi_slide, level: int = 2, return_mask: bool = True) -> np.ndarray:
    """
    Detect tissue regions using Otsu thresholding on grayscale WSI thumbnail.
    
    This is part of MIL preprocessing: extracts tissue from background automatically.
    Uses Otsu's method (adaptive thresholding) to separate tissue from white background.
    
    Args:
        wsi_slide: OpenSlide object of the loaded WSI
        level (int): Pyramid level to use for thumbnail (default: 2, ~4x downsampled).
                    Higher level = faster but less precise
        return_mask (bool): If True, returns binary mask. If False, returns tissue image
    
    Returns:
        np.ndarray: Binary tissue mask (H x W) where 255=tissue, 0=background
                   OR tissue image (H x W x 3) if return_mask=False
    
    Practical notes:
        - Clinically validated for H&E slides
        - Fast and robust for automatic tissue detection
        - Works well with Otsu threshold (no manual tuning needed)
    
    Example:
        >>> import openslide
        >>> slide = openslide.open_slide('slide.svs')
        >>> tissue_mask = otsu_thresholding(slide, level=2)
        >>> # Use mask with extract_patches()
        >>> patches = extract_patches(slide, mask=tissue_mask)
    """
    # Read thumbnail at specified level
    thumb = wsi_slide.read_region((0, 0), level, wsi_slide.level_dimensions[level])
    thumb = np.array(thumb.convert('RGB'))
    
    # Convert to grayscale
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    
    # Apply Otsu thresholding (automatic threshold selection)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    
    if return_mask:
        return binary
    else:
        # Apply mask to original image
        tissue = cv2.bitwise_and(thumb, thumb, mask=binary)
        return tissue


def HSV_filtering(wsi_slide, level: int = 2, 
                 h_range: Tuple[int, int] = (130, 180),
                 s_range: Tuple[int, int] = (20, 255),
                 v_range: Tuple[int, int] = (20, 255),
                 return_mask: bool = True) -> np.ndarray:
    """
    Detect tissue regions using HSV color space filtering.
    
    HSV filtering is more specific for H&E stained tissue:
    - Hue (H): Filters for purple/pink tissue colors (H&E staining)
    - Saturation (S): Removes white/gray background
    - Value (V): Removes dark artifacts
    
    Args:
        wsi_slide: OpenSlide object of the loaded WSI
        level (int): Pyramid level for thumbnail (default: 2)
        h_range (tuple): Hue range (0-180). Default: (130, 180) for purple/pink H&E
        s_range (tuple): Saturation range (0-255). Default: (20, 255)
        v_range (tuple): Value range (0-255). Default: (20, 255)
        return_mask (bool): If True, returns binary mask. If False, returns tissue image
    
    Returns:
        np.ndarray: Binary tissue mask (H x W) where 255=tissue, 0=background
                   OR tissue image (H x W x 3) if return_mask=False
    
    Practical notes:
        - More specific than Otsu (color-based vs intensity-based)
        - Default h_range optimized for H&E stained slides
        - Adjust ranges for different staining protocols
    
    Example:
        >>> slide = openslide.open_slide('slide.svs')
        >>> # Standard H&E detection
        >>> tissue_mask = HSV_filtering(slide)
        >>> 
        >>> # Custom ranges for darker staining
        >>> tissue_mask = HSV_filtering(slide, h_range=(120, 180), s_range=(30, 255))
    """
    # Read thumbnail
    thumb = wsi_slide.read_region((0, 0), level, wsi_slide.level_dimensions[level])
    thumb = np.array(thumb.convert('RGB'))
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    
    # Define color ranges
    lower_bound = np.array([h_range[0], s_range[0], v_range[0]])
    upper_bound = np.array([h_range[1], s_range[1], v_range[1]])
    
    # Create mask using HSV thresholds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if return_mask:
        return mask
    else:
        tissue = cv2.bitwise_and(thumb, thumb, mask=mask)
        return tissue


def combine_tissue_masks(otsu_mask: np.ndarray, hsv_mask: np.ndarray, 
                        method: str = 'intersection') -> np.ndarray:
    """
    Combine Otsu and HSV masks for more robust tissue detection.
    
    Args:
        otsu_mask (np.ndarray): Binary mask from Otsu thresholding
        hsv_mask (np.ndarray): Binary mask from HSV filtering
        method (str): Combination method:
                     - 'intersection': Tissue detected by both (most conservative)
                     - 'union': Tissue detected by either (most inclusive)
                     - 'weighted': Weighted average (balanced)
    
    Returns:
        np.ndarray: Combined binary mask
    
    Practical notes:
        - 'intersection' recommended for clean tissue detection
        - 'union' useful when staining is uneven
    
    Example:
        >>> otsu_mask = otsu_thresholding(slide)
        >>> hsv_mask = HSV_filtering(slide)
        >>> combined = combine_tissue_masks(otsu_mask, hsv_mask, method='intersection')
        >>> patches = extract_patches(slide, mask=combined)
    """
    if method == 'intersection':
        # Both methods must agree (conservative)
        return cv2.bitwise_and(otsu_mask, hsv_mask)
    elif method == 'union':
        # Either method detects tissue (inclusive)
        return cv2.bitwise_or(otsu_mask, hsv_mask)
    elif method == 'weighted':
        # Weighted combination
        combined = (otsu_mask.astype(np.float32) + hsv_mask.astype(np.float32)) / 2
        return (combined > 127).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown method: {method}. Use 'intersection', 'union', or 'weighted'.")