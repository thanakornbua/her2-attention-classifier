import os
import gc
import logging
from typing import List, Dict, Optional, Callable, Any
import numpy as np
from PIL import Image
from glob import glob

from src.preprocessing.xml_to_mask import get_mask
from src.preprocessing.annotation_utils import resolve_annotation_path
from src.preprocessing.load_wsi import load_wsi

# Placeholder for BASE_DIR and create_patch_validator if not imported elsewhere
try:
    from src.config import BASE_DIR
except ImportError:
    BASE_DIR = None
try:
    from src.preprocessing.patch_validator import create_patch_validator
except ImportError:
    def create_patch_validator():
        class Dummy:
            stats = {'discarded': 0}
            def __call__(self, patch, metadata): return True
        return Dummy()

# Configure logging
log_dir = 'outputs/preprocessing/logs'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'preprocessing.log')

logger = logging.getLogger('preprocessing')
if not logger.handlers:
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

def log(msg):
    """Log message using Python logging module (proper file handling)."""
    logger.info(msg)

def extract_patches(
    wsi_slide,
    mask: Optional[np.ndarray] = None,
    size: int = 512,
    stride: int = 512,
    tissue_threshold: float = 0.2,
    level: int = 0,
    save_dir: Optional[str] = None,
    save_prefix: Optional[str] = None,
    save_format: str = 'png',
    use_gpu: bool = False,
    validator: Optional[Callable[[Image.Image, Dict[str, Any]], bool]] = None,
) -> List[Dict]:
    """
    Extract image patches from Whole Slide Images (WSI).

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

    # Infer basename for saved files if prefix not provided
    wsi_basename = save_prefix
    if not wsi_basename:
        for attr in ('path', 'filename', 'name'):
            if hasattr(wsi_slide, attr):
                try:
                    candidate = getattr(wsi_slide, attr)
                    wsi_basename = os.path.splitext(os.path.basename(str(candidate)))[0]
                    break
                except Exception:
                    pass
        if not wsi_basename:
            wsi_basename = 'wsi'

    width, height = wsi_slide.level_dimensions[level]
    level0_width, level0_height = wsi_slide.level_dimensions[0]
    level_downsample = wsi_slide.level_downsamples[level]

    # Determine patch extraction bounds
    if mask is not None:
        mask = _prepare_mask(mask)
        mask_height, mask_width = mask.shape
        
        # Calculate downsample factors between mask and level0
        # Mask might be downsampled from level0 to save memory
        downsample_x = level0_width / mask_width
        downsample_y = level0_height / mask_height

        mask01 = (mask > 0).astype(np.uint8)
        
        # Create integral image with explicit dtype to prevent overflow
        # Use int64 for large images to prevent integer overflow
        ii = np.zeros((mask_height + 1, mask_width + 1), dtype=np.int64)
        ii[1:, 1:] = mask01.cumsum(axis=0).cumsum(axis=1)

        ys, xs = np.nonzero(mask01)
        if ys.size > 0 and xs.size > 0:
            min_mx, max_mx = int(xs.min()), int(xs.max())
            min_my, max_my = int(ys.min()), int(ys.max())
            lvl0_x_min = int(min_mx * downsample_x)
            lvl0_x_max = int((max_mx + 1) * downsample_x)
            lvl0_y_min = int(min_my * downsample_y)
            lvl0_y_max = int((max_my + 1) * downsample_y)
            x_min_lvl = max(0, int(lvl0_x_min / level_downsample) - size)
            x_max_lvl = min(width - size, int(lvl0_x_max / level_downsample))
            y_min_lvl = max(0, int(lvl0_y_min / level_downsample) - size)
            y_max_lvl = min(height - size, int(lvl0_y_max / level_downsample))
        else:
            return []
    else:
        x_min_lvl, y_min_lvl = 0, 0
        x_max_lvl, y_max_lvl = width - size, height - size

    attempt_count = 0
    valid_patch_count = 0

    for y in range(y_min_lvl, y_max_lvl + 1, stride):
        for x in range(x_min_lvl, x_max_lvl + 1, stride):
            attempt_count += 1
            x_level0 = int(x * level_downsample)
            y_level0 = int(y * level_downsample)

            if mask is not None:
                if not _passes_tissue_threshold(
                    ii, x_level0, y_level0, size, level_downsample,
                    downsample_x, downsample_y, mask_width, mask_height, tissue_threshold
                ):
                    continue

            patch = None
            is_valid = True
            try:
                patch = _read_patch(wsi_slide, x_level0, y_level0, level, size, use_gpu)
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
                        patch.close()
                        patch = None
                        continue

                entry = {
                    'patch': None,
                    'x': x_level0,
                    'y': y_level0,
                    'level': level,
                    'path': None
                }

                if do_save:
                    idx = valid_patch_count
                    base = save_prefix if save_prefix else wsi_basename
                    fname = f"{base}_x{x_level0}_y{y_level0}_lv{level}_{idx:04d}.{save_format.lstrip('.')}"
                    out_path = os.path.join(save_dir, fname)
                    try:
                        patch.save(out_path)
                        entry['path'] = out_path
                    except Exception:
                        entry['path'] = None
                    finally:
                        # Always close patch after saving to free memory
                        if patch is not None:
                            try:
                                patch.close()
                            except Exception:
                                pass
                        patch = None
                else:
                    # If not saving, close patch immediately to free memory
                    if patch is not None:
                        try:
                            patch.close()
                        except Exception:
                            pass
                        patch = None

                patches.append(entry)
                valid_patch_count += 1

            except Exception as e:
                # Ensure patch is closed on exception
                if patch is not None:
                    try:
                        patch.close()
                    except Exception:
                        pass
                patch = None
                raise

            finally:
                # Double-check patch is always closed
                if patch is not None:
                    try:
                        patch.close()
                    except Exception:
                        pass

            if attempt_count % 50 == 0:
                gc.collect()

    # Explicitly free mask and integral image memory
    del mask
    if 'ii' in locals():
        del ii
    gc.collect()

    return patches


def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is 2D uint8 binary {0,255}."""
    if mask.ndim != 2:
        mask = mask.squeeze()
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    return mask


def _passes_tissue_threshold(
    ii: np.ndarray,
    x_level0: int,
    y_level0: int,
    size: int,
    level_downsample: float,
    downsample_x: float,
    downsample_y: float,
    mask_width: int,
    mask_height: int,
    tissue_threshold: float
) -> bool:
    """Check if patch passes tissue threshold using integral image."""
    mask_x_start = int(x_level0 / downsample_x)
    mask_y_start = int(y_level0 / downsample_y)
    mask_x_end = int((x_level0 + size * level_downsample) / downsample_x)
    mask_y_end = int((y_level0 + size * level_downsample) / downsample_y)
    mask_x_start = max(0, min(mask_x_start, mask_width))
    mask_y_start = max(0, min(mask_y_start, mask_height))
    mask_x_end = max(0, min(mask_x_end, mask_width))
    mask_y_end = max(0, min(mask_y_end, mask_height))
    if (mask_x_end <= mask_x_start) or (mask_y_end <= mask_y_start):
        return False
    s = (
        ii[mask_y_end, mask_x_end]
        - ii[mask_y_start, mask_x_end]
        - ii[mask_y_end, mask_x_start]
        + ii[mask_y_start, mask_x_start]
    )
    region_area = (mask_y_end - mask_y_start) * (mask_x_end - mask_x_start)
    if region_area <= 0:
        return False
    tissue_ratio = s / float(region_area)
    return tissue_ratio >= tissue_threshold


def _read_patch(
    wsi_slide,
    x_level0: int,
    y_level0: int,
    level: int,
    size: int,
    use_gpu: bool
) -> Image.Image:
    """Read a patch from the WSI, optionally using GPU."""
    if use_gpu and hasattr(wsi_slide, 'backend') and wsi_slide.backend == 'cucim':
        return wsi_slide.read_region(
            location=(x_level0, y_level0),
            level=level,
            size=(size, size),
            device='cuda'
        )
    return wsi_slide.read_region(
        location=(x_level0, y_level0),
        level=level,
        size=(size, size)
    )

def process_slide(row, base_dir=None):
    """Process a single slide to extract patches.
    
    Args:
        row: DataFrame row containing wsi_path and annotation_path
        base_dir: Base directory for resolving annotation paths (defaults to 'data')
    """
    if base_dir is None:
        base_dir = BASE_DIR if BASE_DIR is not None else 'data'
    
    wsi_path = getattr(row, 'wsi_path', None)
    if not isinstance(wsi_path, str) or not wsi_path:
        return

    annotation_path = resolve_annotation_path(
        getattr(row, 'annotation_path', None),
        wsi_path,
        base_dir=base_dir
    )
    if not annotation_path:
        # Skip silently if no annotation found
        return

    log(f"Processing slide: {wsi_path}")

    mask = None
    wsi_slide = None
    try:
        try:
            mask = get_mask(annotation_path, wsi_path)
        except Exception as e:
            log(f"Failed to generate mask for {wsi_path}: {e}")
            return
        if mask is None:
            log(f"No mask generated for {wsi_path}")
            return

        log(f'Mask shape: {mask.shape}')

        try:
            wsi_slide = load_wsi(wsi_path)
        except Exception as e:
            log(f"Failed to load WSI ({wsi_path}): {e}")
            return
        if wsi_slide is None:
            log(f"Failed to load WSI: {wsi_path}")
            return

        backend = getattr(wsi_slide, 'backend', None)
        log(f'Loaded WSI backend: {backend}')

        slide_base = os.path.splitext(os.path.basename(wsi_path))[0]
        out_dir_patches = os.path.join('outputs', 'patches', slide_base)
        existing_paths = glob(os.path.join(out_dir_patches, '*.png')) if os.path.isdir(out_dir_patches) else []
        if existing_paths:
            log(f"Skipping {wsi_path}; found {len(existing_paths)} existing patches in {out_dir_patches}")
            return

        validator = create_patch_validator()
        patches = []
        try:
            patches = extract_patches(
                wsi_slide,
                mask=mask,
                size=512,
                stride=512,
                save_dir=out_dir_patches,
                save_prefix=slide_base,
                save_format='png',
                validator=validator
            )
        except Exception as e:
            log(f"Failed to extract patches for {wsi_path}: {e}")
            return

        saved = sum(1 for entry in patches if entry.get('path') and os.path.exists(entry.get('path')))
        missing = len(patches) - saved
        log(f"Extracted {len(patches)} patches from {wsi_path}; saved {saved} to {out_dir_patches}")
        if validator.stats['discarded']:
            log(f"Discarded {validator.stats['discarded']} low-quality patches for {wsi_path}")
        if missing:
            log(f"Warning: {missing} patches reported saved but missing on disk for {wsi_path}")

    finally:
        # Clean up resources in finally block to ensure cleanup on all code paths
        if wsi_slide is not None:
            if hasattr(wsi_slide, 'close'):
                try:
                    wsi_slide.close()
                except Exception:
                    pass
            wsi_slide = None
        
        if mask is not None:
            del mask
        
        gc.collect()