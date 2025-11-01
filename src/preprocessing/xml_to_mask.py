import numpy as np
import gc
try:
    from lxml import etree as ET
except Exception:
    import xml.etree.ElementTree as ET

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
        # CuCIM expects (location, size, level); use kwargs for safety across versions
        patch = wsi_reader.read_region(location=(x, y), size=(w, h), level=level)
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

    # Open WSI and determine dimensions in a robust way across CuCIM/OpenSlide
    if USE_CUCIM:
        wsi = CuImage(str(wsi_path))
        # CuImage API varies between versions; try multiple attribute/method patterns
        slide_w = slide_h = None
        # 1) get_width / get_height methods
        if hasattr(wsi, 'get_width') and hasattr(wsi, 'get_height'):
            try:
                slide_w = int(wsi.get_width())
                slide_h = int(wsi.get_height())
            except Exception:
                slide_w = slide_h = None
        # 2) width / height attributes
        if slide_w is None and hasattr(wsi, 'width') and hasattr(wsi, 'height'):
            try:
                slide_w = int(wsi.width)
                slide_h = int(wsi.height)
            except Exception:
                slide_w = slide_h = None
        # 3) shape (H,W,...) -> (W,H)
        if slide_w is None and hasattr(wsi, 'shape'):
            try:
                sh = wsi.shape
                if len(sh) >= 2:
                    slide_h = int(sh[0])
                    slide_w = int(sh[1])
            except Exception:
                slide_w = slide_h = None
        # 4) get_dimensions() or get_size()
        if slide_w is None and hasattr(wsi, 'get_dimensions'):
            try:
                dims = wsi.get_dimensions()
                if isinstance(dims, (tuple, list)) and len(dims) >= 2:
                    slide_w = int(dims[0])
                    slide_h = int(dims[1])
            except Exception:
                slide_w = slide_h = None
        if slide_w is None and hasattr(wsi, 'size'):
            try:
                s = wsi.size
                if isinstance(s, (tuple, list)) and len(s) >= 2:
                    slide_w = int(s[0])
                    slide_h = int(s[1])
            except Exception:
                slide_w = slide_h = None
        if slide_w is None or slide_h is None:
            raise AttributeError(f"Could not determine CuImage dimensions for {wsi_path}; available attrs: {dir(wsi)[:10]}")
    else:
        wsi = openslide.OpenSlide(str(wsi_path))
        slide_w, slide_h = wsi.dimensions

    print(f"Slide dimensions: width={slide_w}, height={slide_h}")

    mask = np.zeros((slide_h, slide_w), dtype=np.uint8)
    polygons = parse_xml_lxml(xml_path)
    if not polygons:
        print(f"No polygons found in {xml_path}")
        del mask
        # Close WSI reader properly
        try:
            if hasattr(wsi, 'close'):
                wsi.close()
        except Exception:
            pass
        return None

    for item in polygons:
        poly = item['polygon']
        poly_arr = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [poly_arr], color=255)
        del poly_arr  # Free immediately after use

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    # Avoid asserting on full-image array shape; we may not read full image at all

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
            # Read full image only for full-image GPU path
            if USE_CUCIM:
                wsi_image = np.asarray(
                    wsi.read_region(location=(0, 0), size=(slide_w, slide_h), level=0)
                )
            else:
                wsi_image = np.array(
                    wsi.read_region((0, 0), 0, (slide_w, slide_h)).convert("RGB")
                )

            # Prepare GPU mats. The CUDA arithm functions require the mask to be
            # a single-channel CV_8UC1 image with the same spatial size as the
            # source image. Convert/rescale/resize the mask if necessary.
            mask_for_cuda = mask
            # If mask is RGB overlay, convert to single-channel
            if hasattr(mask_for_cuda, 'ndim') and mask_for_cuda.ndim == 3:
                try:
                    mask_for_cuda = cv2.cvtColor(mask_for_cuda, cv2.COLOR_BGR2GRAY)
                except Exception:
                    mask_for_cuda = mask_for_cuda[:, :, 0]

            # Ensure dtype is uint8 and values are 0..255
            if mask_for_cuda.dtype != np.uint8:
                try:
                    if mask_for_cuda.max() <= 1:
                        mask_for_cuda = (mask_for_cuda * 255).astype(np.uint8)
                    else:
                        mask_for_cuda = mask_for_cuda.astype(np.uint8)
                except Exception:
                    mask_for_cuda = mask_for_cuda.astype(np.uint8)

            # Ensure mask matches image dimensions (h, w)
            if mask_for_cuda.shape[:2] != wsi_image.shape[:2]:
                mask_for_cuda = cv2.resize(
                    mask_for_cuda,
                    (wsi_image.shape[1], wsi_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            g_img = cv2.cuda_GpuMat()
            g_img.upload(wsi_image)
            g_mask = cv2.cuda_GpuMat()
            g_mask.upload(mask_for_cuda)
            g_result = cv2.cuda.bitwise_and(g_img, g_img, mask=g_mask)
            result = g_result.download()
            if wsi_with_mask_path:
                _save_result(result, wsi_with_mask_path)
            result_overlay = result
            # Clean up all intermediate GPU and CPU arrays
            del g_result, g_mask, g_img
            del wsi_image, mask_for_cuda
            del mask
            # Close WSI reader
            try:
                if hasattr(wsi, 'close'):
                    wsi.close()
            except Exception:
                pass
            gc.collect()
            return wsi_with_mask_path if wsi_with_mask_path else result_overlay
        except Exception as e:
            print(f"GPU full-image path failed ({e}), falling back to tiled processing")
            # Clean up any partially allocated resources
            gc.collect()

    overlay = np.zeros((slide_h, slide_w, 3), dtype=np.uint8)
    chunk_count = 0
    if use_cuda:
        for y in range(0, slide_h, chunk_size):
            h = min(chunk_size, slide_h - y)
            for x in range(0, slide_w, chunk_size):
                w = min(chunk_size, slide_w - x)
                # Read region on-demand to avoid full-image memory usage
                region = read_wsi_region(wsi, x, y, w, h, level=0)
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
                chunk_count += 1
                # Periodic garbage collection for large WSI
                if chunk_count % 50 == 0:
                    gc.collect()
    else:
        for y in range(0, slide_h, chunk_size):
            h = min(chunk_size, slide_h - y)
            for x in range(0, slide_w, chunk_size):
                w = min(chunk_size, slide_w - x)
                region = read_wsi_region(wsi, x, y, w, h, level=0)
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
                chunk_count += 1
                # Periodic garbage collection for large WSI
                if chunk_count % 50 == 0:
                    gc.collect()

    if wsi_with_mask_path:
        _save_result(overlay, wsi_with_mask_path)
    result_overlay = overlay

    # Clean up all resources
    del mask
    del overlay
    # Close WSI reader
    try:
        if hasattr(wsi, 'close'):
            wsi.close()
    except Exception:
        pass
    gc.collect()

    return wsi_with_mask_path if wsi_with_mask_path else result_overlay


def get_mask(xml_path, wsi_path, downsample_factor=8):
    """Generate a 2D uint8 binary mask (0/255) directly from XML polygons.

    This avoids building an RGB overlay and avoids reading the entire WSI
    image. Only the WSI dimensions are required to size the mask.
    
    Args:
        xml_path: Path to XML annotation file
        wsi_path: Path to WSI file
        downsample_factor: Factor to downsample mask (default 8) to reduce memory.
                          Higher = less memory but slightly less precision.
    """
    # Open WSI and robustly determine dimensions
    if USE_CUCIM:
        wsi = CuImage(str(wsi_path))
        slide_w = slide_h = None
        if hasattr(wsi, 'get_width') and hasattr(wsi, 'get_height'):
            try:
                slide_w = int(wsi.get_width())
                slide_h = int(wsi.get_height())
            except Exception:
                slide_w = slide_h = None
        if slide_w is None and hasattr(wsi, 'width') and hasattr(wsi, 'height'):
            try:
                slide_w = int(wsi.width)
                slide_h = int(wsi.height)
            except Exception:
                slide_w = slide_h = None
        if slide_w is None and hasattr(wsi, 'shape'):
            try:
                sh = wsi.shape
                if len(sh) >= 2:
                    slide_h = int(sh[0])
                    slide_w = int(sh[1])
            except Exception:
                slide_w = slide_h = None
        if slide_w is None and hasattr(wsi, 'get_dimensions'):
            try:
                dims = wsi.get_dimensions()
                if isinstance(dims, (tuple, list)) and len(dims) >= 2:
                    slide_w = int(dims[0])
                    slide_h = int(dims[1])
            except Exception:
                slide_w = slide_h = None
        if slide_w is None and hasattr(wsi, 'size'):
            try:
                s = wsi.size
                if isinstance(s, (tuple, list)) and len(s) >= 2:
                    slide_w = int(s[0])
                    slide_h = int(s[1])
            except Exception:
                slide_w = slide_h = None
        if slide_w is None or slide_h is None:
            raise AttributeError(f"Could not determine CuImage dimensions for {wsi_path}")
    else:
        import openslide
        wsi = openslide.OpenSlide(str(wsi_path))
        slide_w, slide_h = wsi.dimensions

    # Build mask from XML polygons
    polygons = parse_xml_lxml(xml_path)
    if not polygons:
        # Close WSI reader before returning
        try:
            if hasattr(wsi, 'close'):
                wsi.close()
        except Exception:
            pass
        return None

    # Downsample mask to reduce memory usage
    # For 50000x50000 WSI with downsample=8: mask becomes 6250x6250 (only ~39MB instead of 2.5GB)
    mask_w = int(slide_w / downsample_factor)
    mask_h = int(slide_h / downsample_factor)
    
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    
    # Scale polygons to match downsampled mask
    for item in polygons:
        poly = item['polygon']
        poly_scaled = [(int(x / downsample_factor), int(y / downsample_factor)) for x, y in poly]
        poly_arr = np.array(poly_scaled, dtype=np.int32)
        cv2.fillPoly(mask, [poly_arr], color=255)
        del poly_arr, poly_scaled  # Free immediately after use

    # Close WSI reader
    try:
        if hasattr(wsi, 'close'):
            wsi.close()
    except Exception:
        pass

    return mask.astype(np.uint8)

