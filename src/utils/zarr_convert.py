#!/usr/bin/env python3
"""
zarr_convert.py

CLI tool to convert .zarr slide stores (with schema: patches, coords, labels, meta.json)
to per-patch PNG files on disk. Designed to be memory-efficient and safe:
- Streams patches in configurable batch sizes
- Uses atomic writes (temp file + os.replace)
- Skips already-written PNGs
- Optionally writes metadata CSV per-slide

Usage examples:
  python zarr_convert.py --input /path/to/zarr_dir --output /path/to/out_dir
  python zarr_convert.py --input slide_001.zarr --output /out --batch-size 512

"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple
import tempfile

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import zarr
except Exception as e:
    print("Missing dependency 'zarr'. Install with: pip install zarr", file=sys.stderr)
    raise


LOG = logging.getLogger("zarr_convert")


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_image(img: Image.Image, out_path: Path):
    """Write PIL image to out_path atomically."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(out_path.parent))
    os.close(tmp_fd)
    try:
        img.save(tmp_path, format="PNG")
        os.replace(tmp_path, str(out_path))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def normalize_array_uint8(arr: np.ndarray) -> np.ndarray:
    """Ensure arr is uint8 in HWC layout with 3 channels.
    Accepts arrays already in uint8 or float in [0,1] or [0,255]."""
    if arr.dtype == np.uint8:
        out = arr
    else:
        # assume float
        if arr.max() <= 1.0:
            out = (arr * 255.0).round().astype(np.uint8)
        else:
            out = np.clip(arr, 0, 255).round().astype(np.uint8)
    # Ensure 3 channels
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    if out.shape[-1] == 4:
        out = out[..., :3]
    return out


def write_slide_pngs(
    zarr_path: Path,
    out_root: Path,
    batch_size: int = 256,
    skip_existing: bool = True,
    write_meta_csv: bool = True,
):
    """Convert a single .zarr store to PNGs.

    Expects dataset under 'patches' (N,H,W,3). Optionally uses 'coords' (N,2) and 'labels' (N,).
    Writes files to out_root/<slide_name>/patch_{idx:08d}__x{X}_y{Y}__label{lbl}.png
    Returns number of written files.
    """
    slide_name = zarr_path.stem
    LOG.info(f"Processing zarr: {zarr_path} -> {out_root} (batch={batch_size})")

    store = zarr.open(str(zarr_path), mode="r")

    if "patches" not in store:
        raise RuntimeError(f"No 'patches' dataset found in {zarr_path}")

    patches_ds = store["patches"]
    n_patches = patches_ds.shape[0]
    LOG.info(f"Found patches: {n_patches}, shape per-patch: {patches_ds.shape[1:]}")

    coords_ds = store.get("coords", None)
    labels_ds = store.get("labels", None)

    out_slide_dir = ensure_dir(out_root / slide_name)

    # Optional CSV metadata
    meta_lines = []

    def patch_filename(idx: int, coord: Optional[Tuple[int, int]], label: Optional[int]) -> str:
        name = f"patch_{idx:08d}"
        if coord is not None:
            name += f"__x{int(coord[0])}_y{int(coord[1])}"
        if label is not None:
            name += f"__label{int(label)}"
        name += ".png"
        return name

    written = 0
    pbar = tqdm(total=n_patches, desc=f"{slide_name}", unit="patch")

    try:
        for start in range(0, n_patches, batch_size):
            stop = min(n_patches, start + batch_size)
            batch = patches_ds[start:stop]
            # Convert to uint8 on CPU and iterate per-image to keep memory small
            for i_in_batch, arr in enumerate(batch):
                idx = start + i_in_batch
                coord = None
                label = None
                if coords_ds is not None:
                    try:
                        coord = tuple(coords_ds[idx].tolist())
                    except Exception:
                        coord = None
                if labels_ds is not None:
                    try:
                        label = int(labels_ds[idx])
                    except Exception:
                        label = None

                fname = patch_filename(idx, coord, label)
                out_path = out_slide_dir / fname
                if skip_existing and out_path.exists():
                    pbar.update(1)
                    continue

                arr_u8 = normalize_array_uint8(np.asarray(arr))
                img = Image.fromarray(arr_u8)
                # Atomic write
                atomic_write_image(img, out_path)
                written += 1

                if write_meta_csv:
                    meta_lines.append((idx, coord[0] if coord is not None else "", coord[1] if coord is not None else "", label if label is not None else ""))

                pbar.update(1)
    finally:
        pbar.close()

    # write CSV
    if write_meta_csv and meta_lines:
        csv_path = out_slide_dir / f"{slide_name}_meta.csv"
        try:
            with open(csv_path, "w") as fh:
                fh.write("idx,x,y,label\n")
                for t in meta_lines:
                    fh.write(f"{t[0]},{t[1]},{t[2]},{t[3]}\n")
        except Exception:
            LOG.exception("Failed to write meta CSV")

    LOG.info(f"Finished {slide_name}. Written {written} new PNGs (out of {n_patches}).")
    return written


def find_zarr_paths(input_path: Path):
    """Yield zarr path objects. Accepts single .zarr file or directories containing .zarr children."""
    if input_path.is_file() and input_path.suffix == ".zarr":
        yield input_path
        return
    if input_path.is_dir():
        # find directories that end with .zarr or directories with 'patches' dataset
        for child in sorted(input_path.iterdir()):
            if child.is_dir() and child.suffix == ".zarr":
                yield child
            elif child.is_dir():
                # sometimes zarr stores are plain directories without .zarr suffix; check for 'patches'
                try:
                    store = zarr.open(str(child), mode="r")
                    if "patches" in store:
                        yield child
                except Exception:
                    continue


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert .zarr slide stores to PNG patches.")
    parser.add_argument("--input", "-i", required=True, help="Input .zarr file or directory containing zarr stores")
    parser.add_argument("--output", "-o", required=True, help="Output directory for PNGs")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of patches to load in memory per batch")
    parser.add_argument("--no-csv", dest="write_csv", action="store_false", help="Do not write per-slide meta CSV")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", help="Skip existing PNGs")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--max-slides", type=int, default=None, help="Optional: maximum number of slide zarr stores to convert (useful for testing). If omitted, all discovered slides are processed.")
    args = parser.parse_args(argv)

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    input_path = Path(args.input)
    out_root = Path(args.output)
    ensure_dir(out_root)

    total_written = 0
    zarr_paths = list(find_zarr_paths(input_path))
    if not zarr_paths:
        LOG.error("No zarr stores found at input path")
        sys.exit(2)

    # Optionally limit how many slides to process (useful for quick tests)
    if args.max_slides is not None:
        if args.max_slides <= 0:
            LOG.error("--max-slides must be a positive integer")
            sys.exit(2)
        original_count = len(zarr_paths)
        zarr_paths = zarr_paths[: args.max_slides]
        LOG.info(f"Limiting conversion: processing {len(zarr_paths)} slides out of {original_count} discovered (max-slides={args.max_slides})")

    for zpath in zarr_paths:
        try:
            written = write_slide_pngs(zpath, out_root, batch_size=args.batch_size, skip_existing=args.skip_existing, write_meta_csv=args.write_csv)
            total_written += written
        except Exception:
            LOG.exception(f"Failed to convert {zpath}")

    LOG.info(f"All done. Total new PNGs written: {total_written}")


if __name__ == "__main__":
    main()
