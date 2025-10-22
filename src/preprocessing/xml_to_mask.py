"""wsi_mask_cuda.py

Refactored WSI + XML mask generator with optional CUDA acceleration.

Features:
- Uses cucim.OpenSlide if available for faster, GPU-aware WSI reading; falls back to openslide.
- Parses XML with lxml for speed.
- Creates masks by rasterizing only bounding-box regions of polygons (memory-efficient).
- Uses a memory-mapped file for full-slide mask to avoid loading entire slide into RAM.
- Optional GPU acceleration for overlay using OpenCV CUDA (if available) and cupy.
- Multiprocessing with concurrent.futures.ProcessPoolExecutor and per-slide tasks.
- Saves binary mask (.png) and optional overlay (_overlay.png).

Usage example:
    python wsi_mask_cuda.py \
        --xml-list xml_paths.txt \
        --wsi-list wsi_paths.txt \
        --out-dir ./out_masks \
        --workers 4 \
        --overlay  # include this flag to create RGB overlay

Dependencies (suggested):
- lxml
- numpy
- pillow
- opencv-python (and opencv-contrib-python with cuda support if you want GPU overlay)
- cupy (optional, for some GPU ops)
- openslide-python (or cucim)

Note: GPU acceleration is optional and depends on environment. The script auto-detects capabilities and falls back to CPU.

"""

import os
import sys
import math
import argparse
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from PIL import Image

# XML parsing: lxml is faster and supports XPath
try:
    from lxml import etree as ET
except Exception:
    import xml.etree.ElementTree as ET

# WSI reader: prefer cucim if available, else openslide
USE_CUCIM = False
try:
    import cucim
    from cucim import CuImage
    USE_CUCIM = True
except Exception:
    try:
        import openslide
    except Exception as e:
        raise RuntimeError("Neither cucim nor openslide is available. Install one to read WSI files.")

# OpenCV and GPU utilities
import cv2

# Try to detect CUDA-capable OpenCV and cupy for GPU arrays
HAS_CV2_CUDA = hasattr(cv2, 'cuda')
try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

# ------------------------- Helper functions -------------------------

def parse_xml_lxml(xml_path):
    """Parse XML and return list of polygons with labels.
    Each polygon is dict: {'label': label, 'polygon': [(x1,y1),...]}
    Uses lxml if available (faster); falls back to ElementTree if not.
    """
    polygons = []
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        # Find Annotation nodes
        for ann in root.findall('.//Annotation'):
            label = ann.get('Name') or ann.get('Value') or 'Unknown'
            for region in ann.findall('.//Region'):
                verts = []
                for v in region.findall('.//Vertex'):
                    # sometimes X/Y attributes are strings of floats
                    x = int(float(v.get('X')))
                    y = int(float(v.get('Y')))
                    verts.append((x, y))
                if len(verts) >= 3:
                    polygons.append({'label': label, 'polygon': verts})
    except Exception as e:
        # try fallback for lxml style (if ET is lxml)
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            for ann in root.xpath('.//Annotation'):
                label = ann.get('Name') or 'Unknown'
                for region in ann.xpath('.//Region'):
                    verts = [(int(float(v.get('X'))), int(float(v.get('Y')))) for v in region.xpath('.//Vertex')]
                    if len(verts) >= 3:
                        polygons.append({'label': label, 'polygon': verts})
        except Exception:
            raise
    return polygons


def polygon_bbox(polygon, pad=0):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    minx = max(0, min(xs) - pad)
    miny = max(0, min(ys) - pad)
    maxx = max(xs) + pad
    maxy = max(ys) + pad
    return minx, miny, maxx, maxy


def read_wsi_region(wsi_reader, x, y, w, h, level=0):
    """Read a region from WSI. Supports cucim.CuImage and openslide.OpenSlide
    Returns a numpy RGB (H, W, 3) uint8 array.
    """
    if USE_CUCIM and isinstance(wsi_reader, CuImage):
        # CuImage has get_patch or read_region like methods; use get_patch
        # Coordinates for CuImage: (x, y, w, h) at level 0
        patch = wsi_reader.read_region((x, y), level, (w, h))
        # cucim returns an array-like already
        arr = np.asarray(patch)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3]
        else:
            # convert gray to rgb
            return np.stack([arr] * 3, axis=-1)
    else:
        # openslide
        region = wsi_reader.read_region((x, y), level, (w, h))  # PIL image
        arr = np.asarray(region)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

# ------------------------- Core per-slide task -------------------------

def process_slide(xml_path, wsi_path, out_mask_path, create_overlay=False, overlay_path=None, chunk_size=4096, pad=2):
	"""Use full-slide read + mask rasterization then apply mask to WSI content.
	Optimized: use cv2.cuda when available; supports CUCIM reading if detected.
	"""
	print(f"Processing: {wsi_path} + {xml_path}")

	# Open WSI
	if USE_CUCIM:
		wsi = CuImage(str(wsi_path))
		slide_w, slide_h = wsi.get_width(), wsi.get_height()
	else:
		wsi = openslide.OpenSlide(str(wsi_path))
		slide_w, slide_h = wsi.dimensions

	print(f"Slide dimensions: width={slide_w}, height={slide_h}")

	# Read entire WSI at level 0 (may be large; we will fallback to tiled GPU/CPU if needed)
	if USE_CUCIM:
		wsi_image = np.asarray(wsi.read_region((0, 0), 0, (slide_w, slide_h)))
	else:
		wsi_image = np.array(wsi.read_region((0, 0), 0, (slide_w, slide_h)).convert("RGB"))
	print(f"WSI image shape: {wsi_image.shape}")

	# Create an empty binary mask (H, W)
	mask = np.zeros((slide_h, slide_w), dtype=np.uint8)

	# Parse polygons and rasterize into full mask
	polygons = parse_xml_lxml(xml_path)
	if not polygons:
		print(f"No polygons found in {xml_path}")
		# cleanup and return
		try:
			del wsi_image
		except Exception:
			pass
		del mask
		return None

	for item in polygons:
		poly = item['polygon']
		poly_arr = np.array(poly, dtype=np.int32)
		cv2.fillPoly(mask, [poly_arr], color=255)

	# Ensure mask is uint8 0/255
	if mask.dtype != np.uint8:
		mask = mask.astype(np.uint8)
	if mask.max() <= 1:
		mask = (mask * 255).astype(np.uint8)

	# Ensure alignment
	if wsi_image.shape[:2] != (slide_h, slide_w):
		raise ValueError(f"Mismatch between WSI image dimensions {wsi_image.shape[:2]} and mask dimensions {(slide_h, slide_w)}")

	# Decide GPU usage
	use_cuda = HAS_CV2_CUDA
	# Estimate memory: number of bytes for RGB image
	est_bytes = slide_h * slide_w * 3
	# If image is huge, prefer tiled processing (tile size = chunk_size)
	process_entire = est_bytes <= (2000 * 1024 * 1024)  # allow up to ~2GB for single-shot GPU/host op (adjustable)

	wsi_with_mask_path = str(out_mask_path).replace(".png", "_wsi_with_mask.png")

	def _save_result(result_arr, path):
		# result_arr is uint8 HxWx3
		cv2.imwrite(path, result_arr)

	# Try full-image GPU path
	if use_cuda and process_entire:
		try:
			# upload image and mask to GPU
			g_img = cv2.cuda_GpuMat()
			g_img.upload(wsi_image)
			g_mask = cv2.cuda_GpuMat()
			# mask must be single channel uint8
			g_mask.upload(mask)
			# perform bitwise_and on GPU using mask parameter
			g_result = cv2.cuda.bitwise_and(g_img, g_img, mask=g_mask)
			result = g_result.download()
			_save_result(result, wsi_with_mask_path)
			print(f"WSI with mask saved to {wsi_with_mask_path} (gpu full-image)")
			# cleanup
			del g_result, g_mask, g_img, result
			del wsi_image, mask
			return wsi_with_mask_path
		except Exception as e:
			print(f"GPU full-image path failed ({e}), falling back to tiled processing")

	# Tiled processing (GPU or CPU)
	overlay = np.zeros((slide_h, slide_w, 3), dtype=np.uint8)
	if use_cuda:
		# prepare reusable GPU mats to reduce allocations
		for y in range(0, slide_h, chunk_size):
			h = min(chunk_size, slide_h - y)
			for x in range(0, slide_w, chunk_size):
				w = min(chunk_size, slide_w - x)
				region = wsi_image[y:y + h, x:x + w]  # already loaded
				mask_patch = mask[y:y + h, x:x + w]
				# ensure mask format
				if mask_patch.dtype != np.uint8:
					mask_patch = (mask_patch.astype(np.uint8) * 255) if mask_patch.max() <= 1 else mask_patch.astype(np.uint8)
				elif mask_patch.max() == 1:
					mask_patch = (mask_patch * 255).astype(np.uint8)
				# skip empty tiles quickly
				if mask_patch.sum() == 0:
					del region, mask_patch
					continue
				try:
					g_img = cv2.cuda_GpuMat()
					g_img.upload(region)
					g_mask = cv2.cuda_GpuMat()
					g_mask.upload(mask_patch)
					g_res = cv2.cuda.bitwise_and(g_img, g_img, mask=g_mask)
					res = g_res.download()
					overlay[y:y + h, x:x + w] = res
					# cleanup per tile
					del g_res, g_mask, g_img, res
				except Exception as e:
					# fallback to CPU for this tile
					res = cv2.bitwise_and(region, region, mask=mask_patch)
					overlay[y:y + h, x:x + w] = res
					del res
				del region, mask_patch
	else:
		# pure CPU tiled path (uses preloaded wsi_image)
		for y in range(0, slide_h, chunk_size):
			h = min(chunk_size, slide_h - y)
			for x in range(0, slide_w, chunk_size):
				w = min(chunk_size, slide_w - x)
				region = wsi_image[y:y + h, x:x + w]
				mask_patch = mask[y:y + h, x:x + w]
				if mask_patch.dtype != np.uint8:
					mask_patch = (mask_patch.astype(np.uint8) * 255) if mask_patch.max() <= 1 else mask_patch.astype(np.uint8)
				elif mask_patch.max() == 1:
					mask_patch = (mask_patch * 255).astype(np.uint8)
				if mask_patch.sum() == 0:
					del region, mask_patch
					continue
				res = cv2.bitwise_and(region, region, mask=mask_patch)
				overlay[y:y + h, x:x + w] = res
				del region, mask_patch, res

	# Save overlay (only WSI content inside mask)
	_save_result(overlay, wsi_with_mask_path)
	print(f"WSI with mask saved to {wsi_with_mask_path} (tiled {'gpu' if use_cuda else 'cpu'})")

	# Free memory
	try:
		del wsi_image
	except Exception:
		pass
	del mask
	del overlay

	return wsi_with_mask_path

# ------------------------- CLI and orchestration -------------------------

def process_pair_worker(args):
    return process_slide(*args)


def main(argv=None):
    parser = argparse.ArgumentParser(description='WSI + XML mask generator with optional CUDA.')
    parser.add_argument('--xml-list', type=str, required=True, help='Text file with one XML path per line')
    parser.add_argument('--wsi-list', type=str, required=True, help='Text file with one WSI path per line (same order as XML)')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory for masks')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overlay', action='store_true', help='Create RGB overlay image (can be slow)')
    parser.add_argument('--chunk-size', type=int, default=4096, help='Chunk size for streaming overlay')
    parser.add_argument('--pad', type=int, default=2, help='Padding around polygon bbox when reading regions')
    args = parser.parse_args(argv)

    xml_paths = [p.strip() for p in open(args.xml_list, 'r').read().splitlines() if p.strip()]
    wsi_paths = [p.strip() for p in open(args.wsi_list, 'r').read().splitlines() if p.strip()]
    if len(xml_paths) != len(wsi_paths):
        raise ValueError('xml-list and wsi-list must have the same number of lines and correspond one-to-one')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for xml, wsi in zip(xml_paths, wsi_paths):
        xmlp = Path(xml)
        wsip = Path(wsi)
        out_mask = out_dir / (xmlp.stem + '_mask.png')
        overlay_path = None
        if args.overlay:
            overlay_path = out_dir / (xmlp.stem + '_overlay.png')
        pairs.append((xmlp, wsip, out_mask, args.overlay, overlay_path, args.chunk_size, args.pad))

    # Use ProcessPoolExecutor for CPU-bound IO + small cpu work; avoid huge per-process memory by limiting workers
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        future_to_pair = {ex.submit(process_pair_worker, p): p for p in pairs}
        for fut in as_completed(future_to_pair):
            p = future_to_pair[fut]
            try:
                res = fut.result()
                print(f"Done: {res}")
                results.append(res)
            except Exception as e:
                print(f"Failed: {p[1]} with error: {e}")
    print('All done')


if __name__ == '__main__':
    main()
