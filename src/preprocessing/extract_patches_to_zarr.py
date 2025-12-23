"""src.preprocessing.extract_patches_to_zarr

Pre-extract patches from WSI + full-slide masks into a Zarr archive for fast U-Net training.

Key goals:
- Avoid OOM: stream patches to disk (no giant in-RAM lists)
  * Small chunk_size (default 10-20) for frequent disk writes
  * Explicit garbage collection after each flush
  * Immediate cleanup of stacked arrays after writing
  * WSI and mask cleanup after each slide
- Provide progress bars via tqdm (with a safe fallback)
- Match WSISegmentationDataset semantics for (patch_size, stride, level)
  - Iterate in *level coordinates* using level dimensions
  - For OpenSlide, scale (x,y) to level-0 when calling read_region
  - For mask, map coordinates using scale based on mask vs WSI level dimensions
  
Memory management strategy:
  1. Extract patches one-by-one (generator pattern)
  2. Buffer small batches (chunk_size patches)
  3. Write to Zarr and immediately clear buffer + gc.collect()
  4. Close WSI reader and delete mask after each slide
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import zarr

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

try:
    from cucim import CuImage
    CUCIM_AVAILABLE = True
except Exception:
    CUCIM_AVAILABLE = False

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except Exception:
    OPENSLIDE_AVAILABLE = False


def _open_wsi(wsi_path: str):
    """Open a WSI reader (CuImage preferred), return (reader, use_cucim)."""
    if CUCIM_AVAILABLE:
        try:
            return CuImage(wsi_path), True
        except Exception:
            pass

    if not OPENSLIDE_AVAILABLE:
        raise RuntimeError("Neither cuCIM nor OpenSlide is available to read WSI files.")
    return openslide.OpenSlide(wsi_path), False


def _get_level_dimensions(wsi_reader, level: int, use_cucim: bool) -> Tuple[int, int]:
    """Return (w, h) at a given pyramid level."""
    if use_cucim:
        # cuCIM exposes a resolutions dict in this repo (see WSISegmentationDataset)
        w, h = wsi_reader.resolutions['level_dimensions'][level]
        return int(w), int(h)
    w, h = wsi_reader.level_dimensions[level]
    return int(w), int(h)


def _read_wsi_patch(
    wsi_reader,
    x: int,
    y: int,
    patch_size: int,
    level: int,
    use_cucim: bool,
) -> np.ndarray:
    """Read an RGB patch [H,W,3] at given (x,y) in *level coordinates*."""
    if use_cucim:
        region = wsi_reader.read_region(location=(x, y), size=(patch_size, patch_size), level=level)
        arr = np.asarray(region)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3]
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        return arr

    # OpenSlide expects location in level-0 coordinates
    scale = float(wsi_reader.level_downsamples[level])
    x0 = int(x * scale)
    y0 = int(y * scale)
    region = wsi_reader.read_region(location=(x0, y0), level=level, size=(patch_size, patch_size))
    return np.asarray(region.convert('RGB'))


def _extract_mask_patch(
    mask: np.ndarray,
    x: int,
    y: int,
    patch_size: int,
    wsi_w: int,
    wsi_h: int,
) -> np.ndarray:
    """Extract a mask patch aligned with (x,y) in WSI level-coordinates.

    Mask may be a different resolution than WSI level dimensions.
    Output is (patch_size, patch_size) uint8 in {0,1}.
    """
    mask_h, mask_w = mask.shape[:2]
    scale_x = mask_w / max(1, wsi_w)
    scale_y = mask_h / max(1, wsi_h)

    x_m = int(x * scale_x)
    y_m = int(y * scale_y)
    pw_m = max(1, int(patch_size * scale_x))
    ph_m = max(1, int(patch_size * scale_y))

    patch = mask[y_m:y_m + ph_m, x_m:x_m + pw_m]
    if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    return patch.astype(np.uint8)


def iter_patches_from_wsi(
    wsi_path: str,
    mask_path: str,
    patch_size: int = 256,
    stride: int = 256,
    level: int = 0,
    show_inner_pbar: bool = False,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield (image_patch, mask_patch) without storing everything in RAM."""
    # Load mask once per slide
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")
    mask = (mask > 127).astype(np.uint8)

    wsi, use_cucim = _open_wsi(wsi_path)
    try:
        w, h = _get_level_dimensions(wsi, level=level, use_cucim=use_cucim)
        xs = list(range(0, max(1, w - patch_size + 1), stride))
        ys = list(range(0, max(1, h - patch_size + 1), stride))
        total = len(xs) * len(ys)

        it = ((x, y) for y in ys for x in xs)
        if show_inner_pbar:
            slide_short = Path(wsi_path).stem[:30]  # Truncate long names
            it = tqdm(it, total=total, desc=f"  ├─ {slide_short}", leave=False, 
                     unit="patch", ncols=100, position=1)

        for x, y in it:
            img = _read_wsi_patch(wsi, x=x, y=y, patch_size=patch_size, level=level, use_cucim=use_cucim)
            if img.shape[0] == 0 or img.shape[1] == 0:
                img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            elif img.shape[0] != patch_size or img.shape[1] != patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                hh, ww = img.shape[:2]
                padded[:hh, :ww] = img[:, :, :3]
                img = padded
            else:
                img = img[:, :, :3]

            m = _extract_mask_patch(mask, x=x, y=y, patch_size=patch_size, wsi_w=w, wsi_h=h)
            yield img.astype(np.uint8), m.astype(np.uint8)
    finally:
        try:
            wsi.close()
        except Exception:
            pass
        # Clean up mask from memory
        del mask
        gc.collect()
        pass


def extract_dataset_to_zarr(
    wsi_paths: List[str],
    mask_paths: List[str],
    output_path: str,
    patch_size: int = 256,
    stride: int = 256,
    level: int = 0,
    chunk_size: int = 100
):
    """
    Extract all patches from dataset and save to Zarr format.
    
    Args:
        wsi_paths: List of WSI file paths
        mask_paths: List of mask file paths (must match wsi_paths order)
        output_path: Output Zarr file path (e.g., 'dataset.zarr')
        patch_size: Patch size in pixels
        stride: Stride between patches
        level: WSI pyramid level to extract from
        chunk_size: Zarr chunk size (affects compression and read speed)
    """
    if len(wsi_paths) != len(mask_paths):
        raise ValueError(f"Mismatch: {len(wsi_paths)} WSIs vs {len(mask_paths)} masks")
    
    print(f"Extracting patches from {len(wsi_paths)} WSI/mask pairs...")
    print(f"  Patch size: {patch_size}")
    print(f"  Stride: {stride}")
    print(f"  Level: {level}")
    print(f"  Output: {output_path}")
    
    # Create Zarr arrays (RESIZABLE) and stream-write to disk to avoid OOM
    print("\nStreaming extraction (single pass): extracting and writing chunks to disk...")
    # Zarr v3 API: use open_group directly with path string
    root = zarr.open_group(output_path, mode='w')

    # Zarr v3: use create_array instead of create_dataset
    images_zarr = root.create_array(
        'images',
        shape=(0, patch_size, patch_size, 3),
        chunks=(chunk_size, patch_size, patch_size, 3),
        dtype=np.uint8,
    )

    masks_zarr = root.create_array(
        'masks',
        shape=(0, patch_size, patch_size),
        chunks=(chunk_size, patch_size, patch_size),
        dtype=np.uint8,
    )

    def _flush(buf_imgs: List[np.ndarray], buf_masks: List[np.ndarray], idx: int) -> int:
        """Write buffered patches to Zarr and immediately clear memory."""
        if not buf_imgs:
            return idx
        n = len(buf_imgs)
        # Zarr v3: resize takes full shape tuple, not axis parameter
        images_zarr.resize((idx + n, patch_size, patch_size, 3))
        masks_zarr.resize((idx + n, patch_size, patch_size))
        
        # Stack and write in one operation, then immediately delete stacked arrays
        img_stack = np.stack(buf_imgs, axis=0)
        mask_stack = np.stack(buf_masks, axis=0)
        
        images_zarr[idx:idx + n] = img_stack
        masks_zarr[idx:idx + n] = mask_stack
        
        # Immediately free memory
        del img_stack, mask_stack
        buf_imgs.clear()
        buf_masks.clear()
        gc.collect()  # Force immediate memory cleanup
        
        return idx + n

    idx = 0
    buf_imgs: List[np.ndarray] = []
    buf_masks: List[np.ndarray] = []
    patches_written = 0
    slides_processed = 0

    print("")
    slide_iter = tqdm(zip(wsi_paths, mask_paths), total=len(wsi_paths), 
                     desc="📊 Processing Slides", unit="slide", 
                     ncols=100, position=0, leave=True)
    
    for wsi_path, mask_path in slide_iter:
        slide_name = Path(wsi_path).name
        slide_short = Path(wsi_path).stem[:30]
        slides_processed += 1
        
        try:
            patch_count_before = idx
            for img, m in iter_patches_from_wsi(
                wsi_path=wsi_path,
                mask_path=mask_path,
                patch_size=patch_size,
                stride=stride,
                level=level,
                show_inner_pbar=True,  # Enable nested progress bar
            ):
                buf_imgs.append(img)
                buf_masks.append(m)
                
                # Flush frequently to avoid OOM
                if len(buf_imgs) >= chunk_size:
                    idx = _flush(buf_imgs, buf_masks, idx)
            
            # Flush remaining patches from this slide
            if buf_imgs:
                idx = _flush(buf_imgs, buf_masks, idx)
            
            patches_from_slide = idx - patch_count_before
            patches_written = idx
            
            # Update main progress bar with detailed stats
            slide_iter.set_postfix({
                'total_patches': f'{patches_written:,}',
                'last_slide': f'{patches_from_slide:,}',
                'avg': f'{patches_written//slides_processed:,}'
            })
                    
        except Exception as e:
            print(f"\n⚠️  Error processing {slide_short}: {e}")
            # Flush any remaining patches even on error
            if buf_imgs:
                idx = _flush(buf_imgs, buf_masks, idx)

    # Final flush (should be empty now, but just in case)
    if buf_imgs:
        idx = _flush(buf_imgs, buf_masks, idx)
    
    print("")  # Clean line after progress bars

    print(f"\n✓ Extraction complete!")
    print(f"  Saved {idx:,} patches to {output_path}")
    print(f"  Images shape: {images_zarr.shape}")
    print(f"  Masks shape: {masks_zarr.shape}")
    print(f"  Size on disk: {_get_zarr_size_mb(output_path):.1f} MB")


def _get_zarr_size_mb(zarr_path: str) -> float:
    """Calculate total size of Zarr directory in MB."""
    total_size = 0
    for path in Path(zarr_path).rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size / (1024 * 1024)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract patches to Zarr')
    parser.add_argument('--wsi-dir', required=True, help='Directory with WSI files')
    parser.add_argument('--mask-dir', required=True, help='Directory with mask files')
    parser.add_argument('--output', required=True, help='Output Zarr path')
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=100)
    
    args = parser.parse_args()
    
    # Discover WSI and mask files
    wsi_paths = sorted(Path(args.wsi_dir).glob('*.svs'))
    mask_paths = []
    for wsi_path in wsi_paths:
        mask_path = Path(args.mask_dir) / f"{wsi_path.stem}_mask.png"
        if mask_path.exists():
            mask_paths.append(str(mask_path))
        else:
            print(f"Warning: No mask found for {wsi_path.name}")
    
    wsi_paths = [str(p) for p in wsi_paths[:len(mask_paths)]]
    
    extract_dataset_to_zarr(
        wsi_paths=wsi_paths,
        mask_paths=mask_paths,
        output_path=args.output,
        patch_size=args.patch_size,
        stride=args.stride,
        level=args.level,
        chunk_size=args.chunk_size
    )
