#!/usr/bin/env python3
"""sampling_zarr.py

Sample a fraction of available SVS slides (with matching XML annotations) and
extract 512x512 patches based on polygon annotations, writing per-slide Zarr
stores instead of PNGs.

Key properties:
- Discovers .svs files under the project's standard data layout
- Expects a matching XML file per slide (same basename or TCGA id based)
- Randomly samples a fraction of matching slides (default 20%, controllable via CLI)
- For each slide, builds a binary mask from XML polygons
- Extracts patches on a stride grid; keeps patches whose center lies inside mask
- Writes per-slide Zarr stores with:
    patches: (N, H, W, 3) uint8
    coords:  (N, 2) int64 (x, y in level-0 SVS coords)
    meta:    attributes and optional  meta.json sidecar
- No stain normalization is performed
- CUDA can be enabled for light-weight array operations (mask checks, coord packing)

This script is intentionally independent from the main preprocessing
pipeline; it does not touch the main Macenko normalizer. It can be used to
create smaller Zarr datasets for exploration or debugging.
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable, Optional

import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

try:  # GPU optional
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - GPU optional
    cp = None

try:
    import openslide
except Exception as e:  # pragma: no cover - runtime import error surfaced directly
    openslide = None

try:
    import zarr
except Exception as e:  # pragma: no cover
    zarr = None

LOG = logging.getLogger("sampling_svs_xml")


@dataclass
class SlideItem:
    slide_path: Path
    xml_path: Path


# ------------------------ XML / polygon utilities -------------------------


def parse_xml_polygons(xml_path: Path) -> List[np.ndarray]:
    """Parse XML annotation file and return list of polygons as (N,2) float arrays.

    This is a generic parser that looks for sequences of X,Y under any tag
    named 'Vertex' with attributes X and Y (common ASAP-style format).
    Adjust if your XML schema differs.
    """
    polys: List[np.ndarray] = []
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    for vertex_parent in root.iter():
        # find children that look like vertices
        vertices = []
        for v in list(vertex_parent):
            tag_lower = v.tag.lower()
            if "vertex" in tag_lower or "point" in tag_lower:
                x = v.attrib.get("X") or v.attrib.get("x")
                y = v.attrib.get("Y") or v.attrib.get("y")
                if x is None or y is None:
                    continue
                try:
                    vertices.append((float(x), float(y)))
                except ValueError:
                    continue
        if len(vertices) >= 3:
            polys.append(np.asarray(vertices, dtype=np.float32))

    if not polys:
        LOG.warning(f"No polygons parsed from XML: {xml_path}")
    return polys


def polygons_to_mask(polygons: List[np.ndarray], width: int, height: int) -> np.ndarray:
    """Rasterize polygons into a binary mask (H, W) using skimage if available,
    otherwise a simple PIL-based fallback.
    """
    if not polygons:
        return np.zeros((height, width), dtype=bool)

    try:
        from skimage.draw import polygon as sk_polygon
    except Exception:
        sk_polygon = None

    mask = np.zeros((height, width), dtype=bool)

    if sk_polygon is not None:
        for poly in polygons:
            rr, cc = sk_polygon(poly[:, 1], poly[:, 0], shape=mask.shape)
            mask[rr, cc] = True
    else:
        # Fallback using PIL.ImageDraw
        from PIL import Image, ImageDraw

        pil_mask = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(pil_mask)
        for poly in polygons:
            # poly is (N,2) => list of (x,y)
            draw.polygon([tuple(p) for p in poly.tolist()], outline=1, fill=1)
        mask = np.array(pil_mask, dtype=bool)

    return mask


# ------------------------ Slide discovery / sampling ----------------------


def discover_svs_with_xml(data_root: Path) -> List[SlideItem]:
    """Discover .svs slides under data_root and match with XML.

    Heuristics:
    - For SVS file name 'TCGA-XX.....s.svs', we also allow matching XML
      using just the substring before the first '.'
    - For other slides, look for exact basename (without extension) match.
    - XMLs are searched in sibling 'Annotations' directories or alongside SVS.
    """
    svs_files: List[Path] = []
    for root, dirs, files in os.walk(data_root):
        for fn in files:
            if fn.lower().endswith(".svs"):
                svs_files.append(Path(root) / fn)

    LOG.info(f"Discovered {len(svs_files)} SVS files under {data_root}")

    items: List[SlideItem] = []

    for slide in svs_files:
        slide_name = slide.stem
        # TCGA special case: use string before first '.'
        xml_candidates: List[Path] = []

        # candidate search roots: same dir and ../Annotations
        search_roots = {slide.parent}
        annot_dir = slide.parent.parent / "Annotations"
        if annot_dir.is_dir():
            search_roots.add(annot_dir)

        # Build candidate basenames
        basenames = {slide_name}
        if slide_name.startswith("TCGA-") and "." in slide.name:
            tcga_key = slide.name.split(".")[0]
            basenames.add(tcga_key)

        for r in search_roots:
            for b in basenames:
                for ext in (".xml", ".XML"):
                    p = r / f"{b}{ext}"
                    if p.is_file():
                        xml_candidates.append(p)

        if not xml_candidates:
            continue

        # prefer xml next to slide if multiple
        xml_path = sorted(xml_candidates, key=lambda p: (p.parent != slide.parent, str(p)))[0]
        items.append(SlideItem(slide_path=slide, xml_path=xml_path))

    LOG.info(f"Matched {len(items)} SVS slides with XML annotations")
    return items


# ------------------------ Patch extraction to Zarr ------------------------


def generate_patch_centers(width: int, height: int, patch_size: int, stride: int) -> Iterable[Tuple[int, int]]:
    """Yield (x_center, y_center) positions for a regular grid over the slide."""
    half = patch_size // 2
    for y in range(half, height - half + 1, stride):
        for x in range(half, width - half + 1, stride):
            yield x, y


def extract_patches_to_zarr_for_slide(
    slide_item: SlideItem,
    out_root: Path,
    patch_size: int = 512,
    stride: Optional[int] = None,
    level: int = 0,
    limit_patches: Optional[int] = None,
    use_gpu: bool = False,
    zarr_compressor: Optional[object] = None,
):
    """Extract patches for a single slide and write into a per-slide Zarr store.

    - Builds mask from XML polygons (level 0 coordinates)
    - Generates grid of patch centers with given stride (default = patch_size)
    - Keeps patches whose centers fall inside mask
    - Writes a Zarr store under out_root/<slide_basename>.zarr with groups:
        patches: (N, H, W, 3) uint8
        coords:  (N, 2) int64
        meta:    attributes + meta.json sidecar
    """
    if openslide is None:
        raise RuntimeError("openslide-python is not installed")
    if zarr is None:
        raise RuntimeError("zarr is not installed")

    slide = openslide.OpenSlide(str(slide_item.slide_path))
    width, height = slide.level_dimensions[level]

    LOG.info(f"Slide {slide_item.slide_path.name}: size={width}x{height} level={level}")

    polys = parse_xml_polygons(slide_item.xml_path)
    if not polys:
        LOG.warning(f"No polygons for {slide_item.slide_path.name}, skipping")
        slide.close()
        return 0

    mask = polygons_to_mask(polys, width=width, height=height)

    if stride is None:
        stride = patch_size

    # Prepare Zarr store path; skip if already exists and looks complete
    slide_id = slide_item.slide_path.stem
    zarr_path = out_root / f"{slide_id}.zarr"
    if zarr_path.exists():
        LOG.info(f"Zarr store already exists for slide {slide_id}, skipping.")
        slide.close()
        return 0

    # Collect patch centers first so we can pre-allocate datasets efficiently
    centers: List[Tuple[int, int]] = []
    for xc, yc in generate_patch_centers(width, height, patch_size, stride):
        if mask[yc, xc]:
            centers.append((xc, yc))
            if limit_patches is not None and len(centers) >= limit_patches:
                break

    if not centers:
        LOG.warning(f"No valid patch centers inside polygons for {slide_item.slide_path.name}")
        slide.close()
        return 0

    n_patches = len(centers)
    half = patch_size // 2

    # Create Zarr store and datasets
    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=True)

    patches_ds = root.create_dataset(
        "patches",
        shape=(n_patches, patch_size, patch_size, 3),
        chunks=(min(64, n_patches), patch_size, patch_size, 3),
        dtype="u1",
        compressor=zarr_compressor,
    )
    coords_ds = root.create_dataset(
        "coords",
        shape=(n_patches, 2),
        chunks=(min(1024, n_patches), 2),
        dtype="i8",
        compressor=zarr_compressor,
    )

    # Save minimal metadata
    meta = {
        "slide_path": str(slide_item.slide_path),
        "xml_path": str(slide_item.xml_path),
        "level": level,
        "patch_size": patch_size,
        "stride": stride,
        "width": int(width),
        "height": int(height),
        "n_patches": int(n_patches),
        "use_gpu": bool(use_gpu and cp is not None),
    }
    root.attrs.update(meta)
    # Also dump a meta.json file for convenience
    try:
        with open(zarr_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        LOG.exception(f"Failed to write meta.json for {slide_id}")

    # Optionally, move centers to GPU for batched coord writes, but reading patches is CPU-bound
    if use_gpu and cp is not None:
        centers_arr = cp.asarray(centers, dtype=cp.int64)
        centers_cpu = centers_arr.get()  # bring back as NumPy once; cheap
    else:
        centers_cpu = np.asarray(centers, dtype=np.int64)

    pbar = tqdm(total=n_patches, desc=f"{slide_id}", unit="patch")

    try:
        for idx, (xc, yc) in enumerate(centers_cpu):
            x0 = int(xc - half)
            y0 = int(yc - half)

            region = slide.read_region((x0, y0), level, (patch_size, patch_size))
            img = region.convert("RGB")
            patch = np.array(img, dtype=np.uint8)

            patches_ds[idx] = patch
            coords_ds[idx] = (int(xc), int(yc))

            pbar.update(1)
    finally:
        pbar.close()
        slide.close()

    LOG.info(f"Wrote {n_patches} patches for slide {slide_item.slide_path.name} to {zarr_path}")
    return n_patches


# ------------------------ CLI --------------------------------------------


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Sample a fraction of SVS slides with XML annotations and extract 512x512 "
            "patches from tumor polygons into per-slide Zarr stores (no normalization)."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory containing SVS and XML files (default: data)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/sampled_zarr",
        help="Directory to write per-slide Zarr stores (default: outputs/sampled_zarr)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Patch size in pixels (default: 512)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between patch centers in pixels (default: patch_size)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="Fraction of slides to sample (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for slide sampling (default: 42)",
    )
    parser.add_argument(
        "--limit-patches-per-slide",
        type=int,
        default=None,
        help="Optional maximum number of patches per slide (default: None = no cap)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable CUDA (CuPy) for lightweight array ops where applicable (no effect if CuPy is missing)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args(argv)

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if openslide is None:
        LOG.error("openslide-python is not available. Please install it before running this script.")
        raise SystemExit(1)
    if zarr is None:
        LOG.error("zarr is not available. Please install it before running this script.")
        raise SystemExit(1)

    slides = discover_svs_with_xml(data_root)
    if not slides:
        LOG.error("No SVS+XML pairs found. Nothing to do.")
        raise SystemExit(1)

    # Sample fraction of slides
    random.seed(args.seed)
    n_total = len(slides)
    n_sample = max(1, int(math.ceil(args.fraction * n_total)))
    sampled = random.sample(slides, n_sample)

    LOG.info(f"Sampling {n_sample} / {n_total} slides (fraction={args.fraction:.3f})")

    total_patches = 0
    # Wrap slide iteration in tqdm
    for item in tqdm(sampled, desc="Slides", unit="slide"):
        LOG.info(f"Processing slide: {item.slide_path.name} (XML: {item.xml_path.name})")
        n = extract_patches_to_zarr_for_slide(
            slide_item=item,
            out_root=out_root,
            patch_size=args.patch_size,
            stride=args.stride,
            level=0,
            limit_patches=args.limit_patches_per_slide,
            use_gpu=args.use_gpu,
        )
        total_patches += n

    LOG.info(f"Done. Extracted {total_patches} patches into Zarr from {len(sampled)} slides.")


if __name__ == "__main__":  # pragma: no cover
    main()
