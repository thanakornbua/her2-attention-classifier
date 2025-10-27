import os
from pathlib import Path
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

import cv2

HAS_CV2_CUDA = hasattr(cv2, 'cuda')
try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

def parse_xml_lxml(xml_path):
    polygons = []
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        for ann in root.findall('.//Annotation'):
            label = ann.get('Name') or ann.get('Value') or 'Unknown'
            for region in ann.findall('.//Region'):
                verts = []
                for v in region.findall('.//Vertex'):
                    x = int(float(v.get('X')))
                    y = int(float(v.get('Y')))
                    verts.append((x, y))
                if len(verts) >= 3:
                    polygons.append({'label': label, 'polygon': verts})
    except Exception:
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
    if USE_CUCIM and isinstance(wsi_reader, CuImage):
        patch = wsi_reader.read_region((x, y), level, (w, h))
        arr = np.asarray(patch)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3]
        else:
            return np.stack([arr] * 3, axis=-1)
    else:
        region = wsi_reader.read_region((x, y), level, (w, h))
        arr = np.asarray(region)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

def process_slide(
    xml_path, wsi_path, out_mask_path=None, create_overlay=False, overlay_path=None, chunk_size=4096, pad=2
):
    print(f"Processing: {wsi_path} + {xml_path}")

    if USE_CUCIM:
        wsi = CuImage(str(wsi_path))
        slide_w, slide_h = wsi.get_width(), wsi.get_height()
    else:
        wsi = openslide.OpenSlide(str(wsi_path))
        slide_w, slide_h = wsi.dimensions

    print(f"Slide dimensions: width={slide_w}, height={slide_h}")

    if USE_CUCIM:
        wsi_image = np.asarray(wsi.read_region((0, 0), 0, (slide_w, slide_h)))
    else:
        wsi_image = np.array(wsi.read_region((0, 0), 0, (slide_w, slide_h)).convert("RGB"))
    print(f"WSI image shape: {wsi_image.shape}")

    mask = np.zeros((slide_h, slide_w), dtype=np.uint8)
    polygons = parse_xml_lxml(xml_path)
    if not polygons:
        print(f"No polygons found in {xml_path}")
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

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    if wsi_image.shape[:2] != (slide_h, slide_w):
        raise ValueError(f"Mismatch between WSI image dimensions {wsi_image.shape[:2]} and mask dimensions {(slide_h, slide_w)}")

    use_cuda = HAS_CV2_CUDA
    est_bytes = slide_h * slide_w * 3
    process_entire = est_bytes <= (2000 * 1024 * 1024)

    wsi_with_mask_path = None
    if out_mask_path:
        wsi_with_mask_path = str(out_mask_path).replace(".png", "_wsi_with_mask.png")

    def _save_result(result_arr, path):
        cv2.imwrite(path, result_arr)

    result_overlay = None
    if use_cuda and process_entire:
        try:
            g_img = cv2.cuda_GpuMat()
            g_img.upload(wsi_image)
            g_mask = cv2.cuda_GpuMat()
            g_mask.upload(mask)
            g_result = cv2.cuda.bitwise_and(g_img, g_img, mask=g_mask)
            result = g_result.download()
            if wsi_with_mask_path:
                _save_result(result, wsi_with_mask_path)
            result_overlay = result
            del g_result, g_mask, g_img, result
            del wsi_image, mask
            return wsi_with_mask_path if wsi_with_mask_path else result_overlay
        except Exception as e:
            print(f"GPU full-image path failed ({e}), falling back to tiled processing")

    overlay = np.zeros((slide_h, slide_w, 3), dtype=np.uint8)
    if use_cuda:
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
                try:
                    g_img = cv2.cuda_GpuMat()
                    g_img.upload(region)
                    g_mask = cv2.cuda_GpuMat()
                    g_mask.upload(mask_patch)
                    g_res = cv2.cuda.bitwise_and(g_img, g_img, mask=g_mask)
                    res = g_res.download()
                    overlay[y:y + h, x:x + w] = res
                    del g_res, g_mask, g_img, res
                except Exception:
                    res = cv2.bitwise_and(region, region, mask=mask_patch)
                    overlay[y:y + h, x:x + w] = res
                    del res
                del region, mask_patch
    else:
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

    if wsi_with_mask_path:
        _save_result(overlay, wsi_with_mask_path)
    result_overlay = overlay

    try:
        del wsi_image
    except Exception:
        pass
    del mask
    del overlay

    return wsi_with_mask_path if wsi_with_mask_path else result_overlay

