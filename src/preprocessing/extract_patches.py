import os
import numpy as np
from typing import List, Dict, Optional, Callable, Any
from PIL import Image
import gc


def extract_patches(
    wsi_slide, 
    mask: Optional[np.ndarray] = None, 
    size: int = 512, 
    stride: int = 512,
    tissue_threshold: float = 0.1,
    level: int = 0,
    save_dir: Optional[str] = None,
    save_prefix: Optional[str] = None,
    save_format: str = 'png',
    use_gpu: bool = False,
    validator: Optional[Callable[[Image.Image, Dict[str, Any]], bool]] = None,
) -> List[Dict]:
    """
    Extract image patches from Whole Slide Images (WSI).
    Supports both ROI-only extraction (with mask) and full tissue extraction.
    
    Args:
        wsi_slide: OpenSlide object of the loaded WSI
        mask: Optional binary mask (numpy array) indicating regions to extract from.
        size: Patch size in pixels (default: 512x512)
        stride: Step size between patch centers in pixels (default: 512)
        tissue_threshold: Minimum ratio of tissue/ROI pixels required in mask 
        level: Pyramid level to extract from (0 = highest resolution)
        save_dir: Optional directory to save patches
        save_prefix: Optional prefix for saved patch filenames
        save_format: Image format for saving (default: 'png')
        use_gpu: If True, request GPU memory from CuCIM (requires CuPy for processing)
        validator: Optional callable that receives (patch, metadata) and returns True
            if the patch should be kept. Returning False discards the patch.
    
    Returns:
        List of dictionaries, each containing:
            - 'patch': PIL.Image object of the extracted patch
            - 'x': x-coordinate of top-left corner in level-0 coordinates
            - 'y': y-coordinate of top-left corner in level-0 coordinates
            - 'level': pyramid level the patch was extracted from
            - 'path': optional, filesystem path where the patch was saved (if save_dir provided)
    """
    patches = []
    do_save = save_dir is not None
    if do_save:
        os.makedirs(save_dir, exist_ok=True)

    # try to infer a basename for saved files if prefix not provided
    wsi_basename = None
    if not save_prefix:
        for attr in ('path', 'filename', 'name'):
            if hasattr(wsi_slide, attr):
                try:
                    candidate = getattr(wsi_slide, attr)
                    wsi_basename = os.path.splitext(os.path.basename(str(candidate)))[0]
                    break
                except Exception:
                    pass
        if not wsi_basename:
            # fallback
            wsi_basename = 'wsi'
    level_dimensions = wsi_slide.level_dimensions[level]
    width, height = level_dimensions
    level0_width, level0_height = wsi_slide.level_dimensions[0]
    level_downsample = wsi_slide.level_downsamples[level]

    if mask is not None:
        mask_height, mask_width = mask.shape
        downsample_x = level0_width / mask_width
        downsample_y = level0_height / mask_height

    attempt_count = 0
    valid_patch_count = 0
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            attempt_count += 1
            x_level0 = int(x * level_downsample)
            y_level0 = int(y * level_downsample)

            if mask is not None:
                mask_x_start = int(x_level0 / downsample_x)
                mask_y_start = int(y_level0 / downsample_y)
                mask_x_end = int((x_level0 + size * level_downsample) / downsample_x)
                mask_y_end = int((y_level0 + size * level_downsample) / downsample_y)
                mask_x_start = max(0, min(mask_x_start, mask_width - 1))
                mask_y_start = max(0, min(mask_y_start, mask_height - 1))
                mask_x_end = max(0, min(mask_x_end, mask_width))
                mask_y_end = max(0, min(mask_y_end, mask_height))
                mask_region = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                if mask_region.size == 0:
                    continue
                tissue_ratio = np.sum(mask_region > 0) / mask_region.size
                if tissue_ratio < tissue_threshold:
                    continue

            # Read patch from WSI, optionally on GPU (CuCIM backend only)
            if use_gpu and hasattr(wsi_slide, 'backend') and wsi_slide.backend == 'cucim':
                patch = wsi_slide.read_region(
                    location=(x_level0, y_level0),
                    level=level,
                    size=(size, size),
                    device='cuda'
                )
            else:
                patch = wsi_slide.read_region(
                    location=(x_level0, y_level0),
                    level=level,
                    size=(size, size)
                )
            if patch.mode == 'RGBA':
                patch = patch.convert('RGB')
            metadata = {
                'x': x_level0,
                'y': y_level0,
                'level': level,
                'attempt_index': attempt_count - 1,
                'downsample': level_downsample,
                'size': size,
                'stride': stride,
            }
            if validator:
                try:
                    is_valid = validator(patch, metadata)
                except Exception:
                    is_valid = False
                if not is_valid:
                    # Release resources associated with the discarded patch
                    try:
                        patch.close()
                    except Exception:
                        pass
                    continue
            entry = {
                'patch': patch,
                'x': x_level0,
                'y': y_level0,
                'level': level
            }
            if do_save:
                idx = valid_patch_count
                base = save_prefix if save_prefix else wsi_basename
                fname = f"{base}_x{x_level0}_y{y_level0}_lv{level}_{idx:04d}.{save_format.lstrip('.')}"
                out_path = os.path.join(save_dir, fname)
                try:
                    # PIL will infer format from extension; pass explicit format for clarity
                    patch.save(out_path)
                    entry['path'] = out_path
                    # Close the image to free memory when saving
                    patch.close()
                    entry['patch'] = None  # Don't keep in memory if saved
                except Exception:
                    # if saving fails, still return the patch object
                    entry['path'] = None
            patches.append(entry)
            valid_patch_count += 1

            # Periodic garbage collection to prevent memory buildup with large WSI
            if attempt_count % 100 == 0:
                gc.collect()
                
    return patches