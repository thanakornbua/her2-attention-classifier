"""
SVS to Zarr Preprocessing Pipeline

Converts whole-slide images (SVS) with XML annotations into Zarr archives containing:
- patches/ : N x H x W x 3 uint8 array (Macenko-normalized patches)
- coords/ : N x 2 int32 array (x, y coordinates in level-0 SVS space)
- labels/ : N int8 array (patch-level labels, optional)
- mask/ : optional downsampled binary mask
- meta.json : slide metadata (MPP, magnification, label, paths)

Features:
- OpenSlide for SVS reading
- lxml for XML polygon parsing
- Binary mask generation at level-0 resolution
- Macenko stain normalization (GPU-accelerated with CuPy)
- Multiprocessing for parallel patch extraction
- Memory-efficient batch writes to Zarr
- Progress tracking with tqdm
- Resume capability (skips existing .zarr files)
"""

import os
import gc
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import zarr
from tqdm import tqdm

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    warnings.warn("OpenSlide not available. Install python-openslide.")

try:
    from lxml import etree as ET
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    HAS_LXML = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV not available. Mask generation will fail.")

# Import existing preprocessing modules
from src.preprocessing.stain_normalization import MacenkoNormalizer


# ======================================================================================
# Configuration & Data Structures
# ======================================================================================

@dataclass
class SlideMetadata:
    """Metadata for a single slide."""
    slide_id: str
    slide_name: str
    svs_path: str
    xml_path: str
    label: int  # 0=HER2-, 1=HER2+
    mpp: Optional[float] = None  # microns per pixel
    magnification: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    num_patches: int = 0
    

@dataclass
class PatchExtractionConfig:
    """Configuration for patch extraction."""
    patch_size: int = 256
    stride: int = 256
    level: int = 0  # pyramid level (0 = highest resolution)
    tissue_threshold: float = 0.2  # minimum tissue fraction in mask
    downsample_mask: int = 16  # downsample factor for mask storage
    num_workers: int = 8  # number of parallel workers
    batch_size: int = 100  # patches per batch write to Zarr
    use_gpu: bool = True  # GPU acceleration for Macenko
    skip_existing: bool = True  # skip existing .zarr files


# ======================================================================================
# XML Parsing & Mask Generation
# ======================================================================================

def parse_xml_polygons(xml_path: Path) -> List[Dict]:
    """
    Parse XML annotation file and extract polygons.
    
    Returns:
        List of dicts with keys: 'label' (str), 'polygon' (list of (x,y) tuples)
    """
    polygons = []
    
    if not xml_path.exists():
        logging.warning(f"XML file not found: {xml_path}")
        return polygons
    
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        
        # Try common XML structures
        for ann in root.findall('.//Annotation'):
            label = ann.get('Name') or ann.get('Value') or ann.get('PartOfGroup') or 'Unknown'
            
            for region in ann.findall('.//Region'):
                vertices = []
                for vertex in region.findall('.//Vertex'):
                    x = float(vertex.get('X'))
                    y = float(vertex.get('Y'))
                    vertices.append((int(x), int(y)))
                
                if len(vertices) >= 3:
                    polygons.append({'label': label, 'polygon': vertices})
        
        logging.info(f"Parsed {len(polygons)} polygons from {xml_path.name}")
        
    except Exception as e:
        logging.error(f"Failed to parse XML {xml_path}: {e}")
    
    return polygons


def create_binary_mask(polygons: List[Dict], width: int, height: int, 
                       downsample: int = 1) -> np.ndarray:
    """
    Create binary mask from polygons at specified resolution.
    
    Args:
        polygons: List of polygon dicts from parse_xml_polygons
        width: mask width (at full resolution)
        height: mask height (at full resolution)
        downsample: downsample factor (1 = full resolution)
        
    Returns:
        Binary mask (uint8, 0/255) at downsampled resolution
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV required for mask generation")
    
    # Create mask at downsampled resolution
    mask_w = width // downsample
    mask_h = height // downsample
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    
    if not polygons:
        return mask
    
    # Draw each polygon on mask
    for poly_dict in polygons:
        poly = poly_dict['polygon']
        # Downsample polygon coordinates
        poly_scaled = [(int(x // downsample), int(y // downsample)) for x, y in poly]
        poly_array = np.array(poly_scaled, dtype=np.int32)
        cv2.fillPoly(mask, [poly_array], 255)
    
    return mask


def point_in_polygons(x: int, y: int, polygons: List[Dict]) -> bool:
    """
    Check if point (x, y) is inside any polygon using ray casting.
    
    Args:
        x, y: coordinates to test
        polygons: list of polygon dicts
        
    Returns:
        True if point is inside any polygon
    """
    if not HAS_CV2:
        # Fallback: simple bounding box check
        for poly_dict in polygons:
            poly = poly_dict['polygon']
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            if min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
                return True
        return False
    
    # Use OpenCV for accurate point-in-polygon test
    for poly_dict in polygons:
        poly = poly_dict['polygon']
        poly_array = np.array(poly, dtype=np.int32)
        result = cv2.pointPolygonTest(poly_array, (float(x), float(y)), False)
        if result >= 0:  # inside or on edge
            return True
    
    return False


# ======================================================================================
# Patch Coordinate Generation
# ======================================================================================

def generate_patch_coords(width: int, height: int, patch_size: int, stride: int,
                          polygons: List[Dict], mask: Optional[np.ndarray] = None,
                          tissue_threshold: float = 0.2,
                          downsample_mask: int = 1) -> List[Tuple[int, int]]:
    """
    Generate valid patch coordinates based on polygon containment and tissue threshold.
    
    Args:
        width, height: slide dimensions at level 0
        patch_size: patch size in pixels
        stride: stride between patch centers
        polygons: list of polygon dicts
        mask: optional binary mask (downsampled)
        tissue_threshold: minimum fraction of tissue pixels in patch
        downsample_mask: downsample factor of mask
        
    Returns:
        List of (x, y) tuples representing valid patch top-left coordinates
    """
    valid_coords = []
    
    # Generate grid of candidate patch coordinates
    xs = range(0, width - patch_size + 1, stride)
    ys = range(0, height - patch_size + 1, stride)
    
    for x in xs:
        for y in ys:
            # Check if patch center is inside tumor polygons
            center_x = x + patch_size // 2
            center_y = y + patch_size // 2
            
            if polygons and not point_in_polygons(center_x, center_y, polygons):
                continue
            
            # Check tissue fraction if mask provided
            if mask is not None:
                mask_x1 = x // downsample_mask
                mask_y1 = y // downsample_mask
                mask_x2 = (x + patch_size) // downsample_mask
                mask_y2 = (y + patch_size) // downsample_mask
                
                # Ensure within mask bounds
                mask_x1 = max(0, min(mask_x1, mask.shape[1] - 1))
                mask_x2 = max(0, min(mask_x2, mask.shape[1]))
                mask_y1 = max(0, min(mask_y1, mask.shape[0] - 1))
                mask_y2 = max(0, min(mask_y2, mask.shape[0]))
                
                if mask_x2 <= mask_x1 or mask_y2 <= mask_y1:
                    continue
                
                patch_mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
                tissue_fraction = np.sum(patch_mask > 0) / patch_mask.size
                
                if tissue_fraction < tissue_threshold:
                    continue
            
            valid_coords.append((x, y))
    
    return valid_coords


# ======================================================================================
# Patch Extraction Worker
# ======================================================================================

def extract_patch_worker(args):
    """
    Worker function to extract and normalize a single patch.
    
    Args:
        args: tuple of (svs_path, x, y, patch_size, level, normalizer_params)
        
    Returns:
        dict with keys: 'patch' (np.ndarray), 'x', 'y', or None on failure
    """
    svs_path, x, y, patch_size, level, normalizer_params = args
    
    try:
        # Open slide (each worker gets its own handle)
        slide = openslide.open_slide(str(svs_path))
        
        # Read region
        region = slide.read_region((x, y), level, (patch_size, patch_size))
        patch = np.array(region)
        
        # Convert RGBA to RGB if needed
        if patch.shape[2] == 4:
            patch = patch[:, :, :3]
        
        # Close slide handle
        slide.close()
        
        # Apply Macenko normalization if parameters provided
        if normalizer_params is not None:
            try:
                normalizer = MacenkoNormalizer(
                    use_gpu=normalizer_params.get('use_gpu', False),
                    percentiles=normalizer_params.get('percentiles', (1, 99))
                )
                
                patch = normalizer.normalize(
                    patch,
                    mean_ref_stain_vectors=normalizer_params['stain_vectors'],
                    mean_ref_max_concentrations_tuple=normalizer_params['max_concentrations']
                )
            except Exception as e:
                logging.warning(f"Normalization failed for patch at ({x}, {y}): {e}")
        
        return {'patch': patch.astype(np.uint8), 'x': x, 'y': y}
        
    except Exception as e:
        logging.error(f"Failed to extract patch at ({x}, {y}): {e}")
        return None


# ======================================================================================
# Zarr Creation
# ======================================================================================

def create_zarr_for_slide(slide_meta: SlideMetadata, coords: List[Tuple[int, int]],
                          config: PatchExtractionConfig, 
                          normalizer_params: Optional[Dict],
                          output_dir: Path) -> bool:
    """
    Create Zarr archive for a single slide.
    
    Args:
        slide_meta: slide metadata
        coords: list of valid patch coordinates
        config: extraction configuration
        normalizer_params: Macenko normalization parameters (or None)
        output_dir: directory to write .zarr file
        
    Returns:
        True on success, False on failure
    """
    zarr_path = output_dir / f"{slide_meta.slide_id}.zarr"
    
    if config.skip_existing and zarr_path.exists():
        logging.info(f"Skipping existing Zarr: {zarr_path}")
        return True
    
    num_patches = len(coords)
    if num_patches == 0:
        logging.warning(f"No valid patches for slide {slide_meta.slide_id}, skipping")
        return False
    
    logging.info(f"Creating Zarr for {slide_meta.slide_id} with {num_patches} patches")
    
    try:
        # Create Zarr store
        store = zarr.DirectoryStore(str(zarr_path))
        root = zarr.group(store=store, overwrite=True)
        
        # Create datasets with chunking optimized for sequential read
        chunk_size = min(1000, num_patches)
        
        patches_array = root.create_dataset(
            'patches',
            shape=(num_patches, config.patch_size, config.patch_size, 3),
            chunks=(chunk_size, config.patch_size, config.patch_size, 3),
            dtype=np.uint8,
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        )
        
        coords_array = root.create_dataset(
            'coords',
            shape=(num_patches, 2),
            chunks=(chunk_size, 2),
            dtype=np.int32
        )
        
        labels_array = root.create_dataset(
            'labels',
            shape=(num_patches,),
            chunks=(chunk_size,),
            dtype=np.int8
        )
        
        # Prepare worker arguments
        worker_args = [
            (
                slide_meta.svs_path,
                x, y,
                config.patch_size,
                config.level,
                normalizer_params
            )
            for x, y in coords
        ]
        
        # Extract patches in parallel with batch writes
        patch_idx = 0
        batch_patches = []
        batch_coords = []
        
        with mp.Pool(processes=config.num_workers) as pool:
            with tqdm(total=num_patches, desc=f"Extracting {slide_meta.slide_id}", 
                     leave=False) as pbar:
                
                for result in pool.imap_unordered(extract_patch_worker, worker_args):
                    if result is None:
                        continue
                    
                    batch_patches.append(result['patch'])
                    batch_coords.append([result['x'], result['y']])
                    
                    # Write batch when full
                    if len(batch_patches) >= config.batch_size:
                        batch_size = len(batch_patches)
                        patches_array[patch_idx:patch_idx + batch_size] = np.array(batch_patches)
                        coords_array[patch_idx:patch_idx + batch_size] = np.array(batch_coords)
                        labels_array[patch_idx:patch_idx + batch_size] = slide_meta.label
                        
                        patch_idx += batch_size
                        batch_patches = []
                        batch_coords = []
                        
                        # Force garbage collection
                        gc.collect()
                    
                    pbar.update(1)
        
        # Write remaining patches
        if batch_patches:
            batch_size = len(batch_patches)
            patches_array[patch_idx:patch_idx + batch_size] = np.array(batch_patches)
            coords_array[patch_idx:patch_idx + batch_size] = np.array(batch_coords)
            labels_array[patch_idx:patch_idx + batch_size] = slide_meta.label
            patch_idx += batch_size
        
        # Resize arrays if some patches failed
        if patch_idx < num_patches:
            logging.warning(f"Only extracted {patch_idx}/{num_patches} patches for {slide_meta.slide_id}")
            patches_array.resize(patch_idx, config.patch_size, config.patch_size, 3)
            coords_array.resize(patch_idx, 2)
            labels_array.resize(patch_idx)
        
        # Save metadata
        slide_meta.num_patches = patch_idx
        meta_dict = asdict(slide_meta)
        root.attrs['metadata'] = json.dumps(meta_dict)
        
        # Write meta.json for easy access
        with open(zarr_path / 'meta.json', 'w') as f:
            json.dump(meta_dict, f, indent=2)
        
        logging.info(f"Successfully created {zarr_path} with {patch_idx} patches")
        return True
        
    except Exception as e:
        logging.error(f"Failed to create Zarr for {slide_meta.slide_id}: {e}")
        # Clean up partial zarr
        if zarr_path.exists():
            import shutil
            shutil.rmtree(zarr_path)
        return False


# ======================================================================================
# Main Pipeline
# ======================================================================================

def load_reference_stain_params(ref_stats_path: Path, use_gpu: bool = False) -> Optional[Dict]:
    """
    Load pre-computed Macenko reference parameters from .npz file.
    
    Args:
        ref_stats_path: path to ref_stain_stats.npz
        use_gpu: whether to use GPU for normalization
        
    Returns:
        Dict with 'stain_vectors', 'max_concentrations', 'use_gpu', 'percentiles'
        or None if file doesn't exist
    """
    if not ref_stats_path.exists():
        logging.warning(f"Reference stain stats not found at {ref_stats_path}")
        return None
    
    try:
        data = np.load(ref_stats_path)

        # Check what keys are available
        keys = list(data.keys())
        logging.info(f"Available keys in {ref_stats_path.name}: {keys}")

        # Handle different formats
        if 'stain_vectors' in keys and 'max_h' in keys and 'max_e' in keys:
            stain_vectors = data['stain_vectors']  # shape (3, 2)
            max_h = float(data['max_h'])
            max_e = float(data['max_e'])
        elif 'mean_stain_vectors' in keys and 'mean_max_h' in keys and 'mean_max_e' in keys:
            stain_vectors = data['mean_stain_vectors']
            max_h = float(data['mean_max_h'])
            max_e = float(data['mean_max_e'])
        else:
            logging.warning(f"Reference stats file has unexpected format. Keys: {keys}")
            logging.warning("Please regenerate reference stats using compute_reference_stain.py")
            return None

        params = {
            'stain_vectors': stain_vectors,
            'max_concentrations': (max_h, max_e),
            'use_gpu': use_gpu,
            'percentiles': (1, 99)
        }
        
        logging.info(f"Loaded reference stain parameters from {ref_stats_path}")
        logging.info(f"  Max H: {max_h:.4f}, Max E: {max_e:.4f}")
        return params
        
    except Exception as e:
        logging.error(f"Failed to load reference stain params: {e}")
        return None


def process_slide(slide_meta: SlideMetadata, config: PatchExtractionConfig,
                  normalizer_params: Optional[Dict], output_dir: Path) -> bool:
    """
    Process a single slide: parse XML, generate mask, extract patches, create Zarr.
    
    Args:
        slide_meta: slide metadata
        config: extraction configuration
        normalizer_params: Macenko parameters or None
        output_dir: output directory for Zarr files
        
    Returns:
        True on success, False on failure
    """
    try:
        # Check if already exists
        zarr_path = output_dir / f"{slide_meta.slide_id}.zarr"
        if config.skip_existing and zarr_path.exists():
            logging.info(f"Skipping existing slide: {slide_meta.slide_id}")
            return True
        
        # Open slide to get dimensions
        slide = openslide.open_slide(slide_meta.svs_path)
        width, height = slide.dimensions
        
        # Get MPP and magnification if available
        try:
            slide_meta.mpp = float(slide.properties.get('openslide.mpp-x', 0))
        except:
            pass
        
        try:
            slide_meta.magnification = float(slide.properties.get('openslide.objective-power', 0))
        except:
            pass
        
        slide_meta.width = width
        slide_meta.height = height
        
        slide.close()
        
        # Parse XML polygons
        polygons = parse_xml_polygons(Path(slide_meta.xml_path))
        
        if not polygons:
            logging.warning(f"No polygons found for {slide_meta.slide_id}, processing entire slide")
        
        # Generate binary mask (downsampled for memory efficiency)
        mask = create_binary_mask(polygons, width, height, config.downsample_mask)
        
        # Generate valid patch coordinates
        coords = generate_patch_coords(
            width, height,
            config.patch_size, config.stride,
            polygons, mask,
            config.tissue_threshold,
            config.downsample_mask
        )
        
        logging.info(f"Generated {len(coords)} valid patch coordinates for {slide_meta.slide_id}")
        
        # Create Zarr archive
        success = create_zarr_for_slide(
            slide_meta, coords, config, normalizer_params, output_dir
        )
        
        # Cleanup
        del mask
        gc.collect()
        
        return success
        
    except Exception as e:
        logging.error(f"Failed to process slide {slide_meta.slide_id}: {e}")
        return False


def main():
    """Main entry point for SVS to Zarr conversion."""
    # This will be called from notebook or CLI
    pass


if __name__ == "__main__":
    main()

