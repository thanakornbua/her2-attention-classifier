#!/usr/bin/env python3
"""
HER2 Slide Preprocessing CLI Script
Extracts patches from WSI slides (.svs), applies Macenko stain normalization,
and saves to Zarr format with aggressive memory management.
"""

import os
import sys
import logging
import random
import json
import math
import gc
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import openslide
except ImportError:
    print("ERROR: openslide-python not installed. Install with: pip install openslide-python")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("WARNING: scikit-learn not installed. Train/val split will be skipped.")
    train_test_split = None


class MacenkoNormalizer:
    """Macenko stain normalization with optional CuPy GPU acceleration."""
    
    def __init__(self, percentiles: Tuple[float, float] = (1, 99), use_gpu: bool = False):
        self.percentiles = percentiles
        self.use_gpu = use_gpu
        
        if use_gpu:
            try:
                import cupy as cp
                self.xp = cp
                self.gpu_available = True
            except ImportError:
                print("WARNING: CuPy not available, falling back to CPU")
                self.xp = np
                self.gpu_available = False
                self.use_gpu = False
        else:
            self.xp = np
            self.gpu_available = False

    @staticmethod
    def _rgb_to_od(image_rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to Optical Density space."""
        img = image_rgb.astype(np.float32) + 1.0  # avoid log(0)
        od = -np.log(img / 255.0)
        return od

    @staticmethod
    def _od_to_rgb(image_od: np.ndarray) -> np.ndarray:
        """Convert OD back to RGB."""
        rgb = (255.0 * np.exp(-image_od)).clip(0, 255).astype(np.uint8)
        return rgb

    @staticmethod
    def _to_cpu(arr):
        """Convert array to CPU numpy array if it's on GPU."""
        if hasattr(arr, 'get'):  # CuPy array
            return arr.get()
        return np.asarray(arr)

    def _cleanup_gpu(self):
        """Free GPU memory pools."""
        if self.use_gpu and self.gpu_available:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    def _get_stain_vectors_and_concentrations(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        """Extract stain vectors and concentrations from image."""
        # Convert to OD
        od = self._rgb_to_od(image_rgb)
        H, W, CH = od.shape
        od_reshaped = od.reshape(-1, 3)

        # Filter near-white background
        mask = np.sum(od_reshaped, axis=1) > 0.2
        od_filtered = od_reshaped[mask]
        if od_filtered.shape[0] < 100:
            od_filtered = od_reshaped

        # Move to xp for PCA
        xp = self.xp
        odf = xp.asarray(od_filtered, dtype=xp.float32)
        odf = odf - odf.mean(axis=0, keepdims=True)
        
        # SVD to get top 2 PCs
        try:
            U, S, VT = xp.linalg.svd(odf, full_matrices=False)
            v = VT.T[:, :2]  # (3,2)
            del U, S, VT  # Free memory immediately
        except Exception:
            # CPU fallback if GPU fails
            U, S, VT = np.linalg.svd(od_filtered - od_filtered.mean(axis=0, keepdims=True), full_matrices=False)
            v = VT.T[:, :2]
            del U, S, VT
            xp = np

        # Normalize columns
        v = v / xp.linalg.norm(v, axis=0, keepdims=True)

        # Ensure consistent direction
        for i in range(2):
            if float(v[:, i].sum()) < 0:
                v[:, i] = -v[:, i]

        # Macenko angle-based stain separation
        odf_2d = xp.asarray(od_filtered, dtype=xp.float32) @ v
        angles = xp.arctan2(odf_2d[:, 1], odf_2d[:, 0])

        # Find angle percentiles
        angles_cpu = self._to_cpu(angles)
        min_angle = float(np.percentile(angles_cpu, self.percentiles[0]))
        max_angle = float(np.percentile(angles_cpu, self.percentiles[1]))

        # Construct stain vectors at extreme angles
        stain_h = xp.cos(min_angle) * v[:, 0] + xp.sin(min_angle) * v[:, 1]
        stain_e = xp.cos(max_angle) * v[:, 0] + xp.sin(max_angle) * v[:, 1]

        # Normalize stain vectors
        stain_h = stain_h / xp.linalg.norm(stain_h)
        stain_e = stain_e / xp.linalg.norm(stain_e)

        # Stack into matrix [H, E] as columns
        stain_matrix = xp.column_stack([stain_h, stain_e])

        # Project ALL pixels onto stain vectors
        od_all = xp.asarray(od_reshaped.astype(np.float32))
        C = od_all @ stain_matrix
        C = xp.maximum(C, 0)

        # Back to CPU
        stain_vectors = self._to_cpu(stain_matrix)
        concentrations = self._to_cpu(C)

        # Cleanup intermediate arrays
        del odf_2d, angles, stain_h, stain_e, stain_matrix, odf, od_all, v, C

        return stain_vectors, concentrations, (H, W, CH)

    def get_mean_reference_stain_characteristics(self, list_of_reference_images_rgb: List[np.ndarray]):
        """Compute mean reference stain characteristics from list of images."""
        if not list_of_reference_images_rgb:
            raise ValueError("list_of_reference_images_rgb cannot be empty.")
        
        all_V = []
        max_h = []
        max_e = []
        
        for i, img in enumerate(tqdm(list_of_reference_images_rgb, desc="Computing reference stats")):
            V, C, _ = self._get_stain_vectors_and_concentrations(img)
            all_V.append(V)
            
            # Extract percentiles and delete C immediately
            h_val = float(np.percentile(C[:, 0], self.percentiles[1]))
            e_val = float(np.percentile(C[:, 1], self.percentiles[1]))
            max_h.append(h_val)
            max_e.append(e_val)
            del C, V
            
            # Free GPU memory every 10 images
            if self.use_gpu and (i + 1) % 10 == 0:
                self._cleanup_gpu()
            
            # Force garbage collection every 50 images
            if (i + 1) % 50 == 0:
                gc.collect()

        # Compute final statistics
        mean_V = np.mean(np.stack(all_V, axis=0), axis=0)
        mean_V = mean_V / np.linalg.norm(mean_V, axis=0, keepdims=True)
        mean_max_h = float(np.mean(max_h))
        mean_max_e = float(np.mean(max_e))

        # Cleanup
        del all_V, max_h, max_e
        gc.collect()
        self._cleanup_gpu()

        return mean_V, (mean_max_h, mean_max_e)

    def normalize(self, target_image_rgb: np.ndarray,
                  mean_ref_stain_vectors: np.ndarray,
                  mean_ref_max_concentrations_tuple: Tuple[float, float]) -> np.ndarray:
        """Normalize target image to match reference stain characteristics."""
        # Target characteristics
        V_t, C_t, shape = self._get_stain_vectors_and_concentrations(target_image_rgb)
        max_t_h = np.percentile(C_t[:, 0], self.percentiles[1])
        max_t_e = np.percentile(C_t[:, 1], self.percentiles[1])

        # Scale concentrations to reference
        ref_max_h, ref_max_e = mean_ref_max_concentrations_tuple
        scale_h = ref_max_h / (max_t_h + 1e-6)
        scale_e = ref_max_e / (max_t_e + 1e-6)
        
        Cn = C_t.copy()
        Cn[:, 0] *= scale_h
        Cn[:, 1] *= scale_e
        Cn = np.maximum(Cn, 0)

        # Reconstruct OD using reference stain vectors
        V_ref = mean_ref_stain_vectors.astype(np.float32)
        od_norm = (Cn @ V_ref.T).reshape(shape)
        rgb_norm = self._od_to_rgb(od_norm)
        
        # Cleanup
        del V_t, C_t, Cn, od_norm
        
        return rgb_norm


def load_reference_stain_params(npz_path: Path, use_gpu: bool = False) -> Optional[Dict]:
    """Load precomputed reference stain parameters from npz file."""
    try:
        if not npz_path.exists():
            return None
        data = np.load(str(npz_path))
        if 'stain_vectors' in data and ('max_h' in data or 'mean_max_h' in data):
            V = data['stain_vectors']
            max_h = float(data.get('max_h', data.get('mean_max_h')))
            max_e = float(data.get('max_e', data.get('mean_max_e')))
            return {
                'stain_vectors': V,
                'max_concentrations': (max_h, max_e),
                'use_gpu': use_gpu,
                'percentiles': (1, 99)
            }
    except Exception as e:
        logging.error(f"Failed to load reference stain parameters: {e}")
    return None


def load_images_from_folder(folder_path: Path, max_images: int = 200) -> List[np.ndarray]:
    """Load images from a folder."""
    images: List[np.ndarray] = []
    supported_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    if not folder_path.exists():
        return images
    
    image_files: List[Path] = []
    for ext in supported_ext:
        image_files.extend(list(folder_path.glob(f"*{ext}")))
        image_files.extend(list(folder_path.glob(f"*{ext.upper()}")))
    
    if len(image_files) == 0:
        return images
    
    if len(image_files) > max_images:
        image_files = random.sample(image_files, max_images)
    
    for p in image_files:
        try:
            with Image.open(p) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(np.array(img))
        except Exception as e:
            logging.debug(f"Skip {p}: {e}")
    
    return images


def compute_reference_stats(patches_root: Path, num_subfolders: int, images_per_folder: int,
                           use_gpu: bool, output_path: Path):
    """Compute reference stain statistics from sample patches."""
    print(f"Sampling from up to {num_subfolders} random folders...")
    subfolders = [d for d in patches_root.iterdir() if d.is_dir()]
    num_to_sample = min(num_subfolders, len(subfolders))
    random.seed(42)
    sampled_folders = random.sample(subfolders, num_to_sample) if num_to_sample > 0 else []

    all_reference_images: List[np.ndarray] = []
    print(f"Loading reference images from {len(sampled_folders)} folders (max {images_per_folder} per folder)...")
    
    for folder_idx, folder in enumerate(tqdm(sampled_folders, desc="Loading reference images")):
        imgs = load_images_from_folder(folder, max_images=images_per_folder)
        all_reference_images.extend(imgs)
        del imgs
        
        # Force garbage collection every 5 folders
        if (folder_idx + 1) % 5 == 0:
            gc.collect()

    print(f"✓ Loaded {len(all_reference_images)} reference images")
    
    if len(all_reference_images) > 0:
        print("Computing Macenko reference statistics...")
        normalizer = MacenkoNormalizer(use_gpu=use_gpu, percentiles=(1, 99))
        mean_V, (mean_max_h, mean_max_e) = normalizer.get_mean_reference_stain_characteristics(all_reference_images)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, stain_vectors=mean_V, max_h=mean_max_h, max_e=mean_max_e)

        print(f"✓ Reference statistics saved to {output_path}")
        print(f"  Max H: {mean_max_h:.4f}, Max E: {mean_max_e:.4f}")

        # Cleanup
        del all_reference_images, mean_V, normalizer
        gc.collect()


def discover_slides(data_root: str, cohorts: list) -> List[Dict]:
    """Discover SVS slides and corresponding XML annotations."""
    slides = []
    
    for cohort in cohorts:
        cohort_dir = Path(data_root) / cohort
        svs_dir = cohort_dir / "SVS"
        xml_dir = cohort_dir / "Annotations"
        
        if not svs_dir.exists():
            print(f"⚠️  SVS directory not found: {svs_dir}")
            continue
        if not xml_dir.exists():
            print(f"⚠️  Annotations directory not found: {xml_dir}")
            continue
        
        # Optional labels
        labels_dict = {}
        for label_file_name in ['labels.csv', 'HER2_TCGA_clean.csv']:
            label_file = cohort_dir / label_file_name
            if label_file.exists():
                try:
                    df_labels = pd.read_csv(label_file)
                    if 'slide_id' in df_labels.columns and 'label' in df_labels.columns:
                        labels_dict = dict(zip(df_labels['slide_id'], df_labels['label']))
                    elif 'case_id' in df_labels.columns and 'HER2_IHC_Status' in df_labels.columns:
                        def map_her2_status(status):
                            if isinstance(status, str) and (('Positive' in status) or ('3+' in status) or ('2+' in status)):
                                return 1
                            return 0
                        labels_dict = {row['case_id']: map_her2_status(row['HER2_IHC_Status']) 
                                     for _, row in df_labels.iterrows()}
                    print(f"✓ Loaded labels for {len(labels_dict)} slides from {cohort}/{label_file_name}")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load labels from {label_file}: {e}")
        
        # SVS files
        svs_files = list(svs_dir.glob("*.svs")) + list(svs_dir.glob("*.SVS"))
        
        for svs_path in svs_files:
            slide_name = svs_path.stem
            
            # For TCGA slides, use only part before first dot to match XML
            xml_base_name = slide_name
            if slide_name.startswith("TCGA-") and "." in slide_name:
                xml_base_name = slide_name.split(".")[0]
            
            xml_candidates = [
                xml_dir / f"{xml_base_name}.xml",
                xml_dir / f"{xml_base_name}.XML",
                xml_dir / f"{slide_name}.xml",
                xml_dir / f"{slide_name}.XML"
            ]
            
            xml_path = None
            for cand in xml_candidates:
                if cand.exists():
                    xml_path = cand
                    break
            
            if xml_path is None:
                print(f"⚠️  No XML found for {slide_name} (tried {xml_base_name}), skipping")
                continue
            
            label = int(labels_dict.get(slide_name, 0))
            slides.append({
                'slide_id': slide_name,
                'svs_path': str(svs_path),
                'xml_path': str(xml_path),
                'cohort': cohort,
                'label': label
            })
    
    return slides


def tissue_fraction_rgb(patch_rgb: np.ndarray) -> float:
    """Quick HSV-based tissue detection to filter whitespace patches."""
    # Simple tissue detection: low saturation or high value = background
    hsv = patch_rgb.astype(np.float32) / 255.0
    # Approximate HSV conversion (simpler than full conversion)
    v = hsv.max(axis=2)
    s = (hsv.max(axis=2) - hsv.min(axis=2)) / (v + 1e-8)
    tissue = (s > 0.1) & (v < 0.95)
    return float(tissue.mean())


def parse_xml_polygons(xml_path: str) -> List[np.ndarray]:
    """Parse Aperio-like XML annotations to extract polygons."""
    polys: List[np.ndarray] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for region in root.iter("Region"):
            pts = []
            vertices = list(region.iter("Vertex"))
            if not vertices:
                vertices = list(region.iter("Coordinate"))
            
            for v in vertices:
                x = v.get('X') or v.get('x') or v.get('XCoord')
                y = v.get('Y') or v.get('y') or v.get('YCoord')
                if x is None or y is None:
                    continue
                pts.append((float(x), float(y)))
            
            if len(pts) >= 3:
                polys.append(np.array(pts, dtype=np.float32))
    except Exception as e:
        logging.error(f"Failed to parse XML {xml_path}: {e}")
    
    return polys


def polygons_to_mask(polygons: List[np.ndarray], level0_size: Tuple[int, int], downsample: int = 16) -> np.ndarray:
    """Rasterize polygons into a downsampled binary mask."""
    W, H = level0_size
    w_small = max(1, W // downsample)
    h_small = max(1, H // downsample)
    
    mask_img = Image.new('L', (w_small, h_small), 0)
    draw = ImageDraw.Draw(mask_img)
    
    for poly in polygons:
        scaled = [(p[0] / downsample, p[1] / downsample) for p in poly]
        try:
            draw.polygon(scaled, outline=1, fill=1)
        except Exception:
            continue
    
    return (np.array(mask_img) > 0)


def generate_grid_centers(level0_size: Tuple[int, int], patch: int, stride: int) -> List[Tuple[int, int]]:
    """Generate regular grid of patch centers at level 0 coordinates."""
    W, H = level0_size
    xs = list(range(patch // 2, W - patch // 2 + 1, stride))
    ys = list(range(patch // 2, H - patch // 2 + 1, stride))
    centers = [(x, y) for y in ys for x in xs]
    return centers


def create_zarr_group(zarr_path: Path, num_patches: int, patch: int) -> zarr.hierarchy.Group:
    """Create Zarr group with datasets for patches, coords, and labels."""
    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=2)
    
    root.create_dataset('patches', shape=(num_patches, patch, patch, 3), maxshape=(None, patch, patch, 3),
                       chunks=(min(256, num_patches), patch, patch, 3), dtype='u1', compressor=compressor, overwrite=True)
    root.create_dataset('coords', shape=(num_patches, 2), maxshape=(None, 2),
                       chunks=(min(4096, num_patches), 2), dtype='i4', compressor=compressor, overwrite=True)
    root.create_dataset('labels', shape=(num_patches,), maxshape=(None,),
                       chunks=(min(4096, num_patches),), dtype='i1', compressor=compressor, overwrite=True)
    
    return root


def process_slide(slide_info: dict, patch_size: int, stride: int, level: int,
                 tissue_threshold: float, downsample_mask: int,
                 normalizer_params: Optional[Dict], out_dir: Path,
                 num_workers: int = 4, batch_size: int = 128, skip_existing: bool = True) -> bool:
    """Process a single slide: extract patches, normalize, and save to Zarr."""
    
    slide_id = slide_info['slide_id']
    svs_path = slide_info['svs_path']
    xml_path = slide_info['xml_path']
    label = int(slide_info['label'])

    zarr_path = out_dir / f"{slide_id}.zarr"
    
    if skip_existing and zarr_path.exists():
        meta_ok = (zarr_path / 'meta.json').exists()
        if meta_ok:
            logging.info(f"Skip existing: {slide_id}")
            return True
        else:
            logging.info(f"Existing zarr missing meta; rewriting: {slide_id}")

    slide = None
    try:
        # Open slide
        slide = openslide.OpenSlide(svs_path)
        W, H = slide.level_dimensions[0]
        
        # Parse XML polygons and create mask
        polygons = parse_xml_polygons(xml_path)
        if len(polygons) == 0:
            logging.warning(f"No polygons in XML for {slide_id}; skipping")
            slide.close()
            return False
        
        mask = polygons_to_mask(polygons, (W, H), downsample=downsample_mask)
        del polygons  # Free memory
        
        # Generate grid centers
        centers = generate_grid_centers((W, H), patch_size, stride)
        
        # Filter centers inside mask
        ds = downsample_mask
        valid_centers = [(x, y) for (x, y) in centers 
                        if mask[min(H // ds - 1, y // ds), min(W // ds - 1, x // ds)]]
        del centers, mask  # Free memory
        
        if len(valid_centers) == 0:
            logging.warning(f"No valid centers after masking for {slide_id}; skipping")
            slide.close()
            return False

        # Create normalizer if params available
        normalizer = None
        V_ref = None
        ref_max = None
        
        if normalizer_params is not None:
            normalizer = MacenkoNormalizer(
                percentiles=normalizer_params.get('percentiles', (1, 99)),
                use_gpu=normalizer_params.get('use_gpu', False)
            )
            V_ref = normalizer_params['stain_vectors']
            ref_max = normalizer_params['max_concentrations']

        # Pre-create zarr arrays
        z = create_zarr_group(zarr_path, len(valid_centers), patch_size)

        # Extract metadata
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        try:
            magnification = float(slide.properties.get('aperio.AppMag') or 
                                slide.properties.get('openslide.objective-power') or 0)
        except Exception:
            magnification = None

        # Worker function for parallel patch extraction
        def read_and_process(idx_center_pair):
            idx, (cx, cy) = idx_center_pair
            x0 = cx - patch_size // 2
            y0 = cy - patch_size // 2
            
            # Read region (returns RGBA)
            region = slide.read_region((x0, y0), level, (patch_size, patch_size))
            try:
                region = region.convert('RGB')
                patch = np.array(region)
            finally:
                region.close()
            
            # Quick tissue filter
            if tissue_threshold > 0:
                if tissue_fraction_rgb(patch) < tissue_threshold:
                    return idx, None, (cx, cy)
            
            # Normalize if available
            if normalizer is not None and V_ref is not None and ref_max is not None:
                try:
                    patch = normalizer.normalize(patch, mean_ref_stain_vectors=V_ref,
                                                mean_ref_max_concentrations_tuple=ref_max)
                except Exception as e:
                    logging.debug(f"Norm fail at idx {idx}: {e}")
            
            return idx, patch, (cx, cy)

        # Process patches in batches
        total = len(valid_centers)
        written = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for start in tqdm(range(0, total, batch_size), total=math.ceil(total / batch_size),
                            desc=f"{slide_id}", leave=False):
                end = min(start + batch_size, total)
                futures = [ex.submit(read_and_process, (i, valid_centers[i])) for i in range(start, end)]
                
                for fut in as_completed(futures):
                    idx, patch, coord = fut.result()
                    if patch is None:
                        continue
                    
                    # Write to zarr
                    z['patches'][written] = patch
                    z['coords'][written] = coord
                    z['labels'][written] = label
                    written += 1
                
                # Free GPU pools per batch
                if normalizer is not None and normalizer.use_gpu:
                    normalizer._cleanup_gpu()
                
                gc.collect()

        # Resize datasets to actual written size
        try:
            z['patches'].resize((written, patch_size, patch_size, 3))
            z['coords'].resize((written, 2))
            z['labels'].resize((written,))
        except Exception as e:
            logging.debug(f"Resize failed for {slide_id}: {e}")

        # Write metadata
        meta = {
            'slide_id': slide_id,
            'label': label,
            'num_patches': int(written),
            'mpp_x': float(mpp_x) if mpp_x else None,
            'mpp_y': float(mpp_y) if mpp_y else None,
            'magnification': magnification,
            'patch_size': patch_size,
            'stride': stride,
            'level': level
        }
        with open(zarr_path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        slide.close()
        
        # Final cleanup for this slide
        del valid_centers, z, normalizer
        gc.collect()
        
        return True
        
    except Exception as e:
        logging.error(f"Exception processing slide {slide_id}: {e}", exc_info=True)
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass
        return False


def create_train_val_split(zarr_output_dir: Path, outputs_root: Path):
    """Create train/val split manifest CSV files."""
    zarr_files = list(zarr_output_dir.glob("*.zarr"))
    print(f"Found {len(zarr_files)} Zarr files")

    if len(zarr_files) == 0:
        print("⚠️  No Zarr files found for train/val split")
        return

    zarr_manifest = []
    for zarr_path in tqdm(zarr_files, desc="Reading Zarr metadata"):
        meta_path = zarr_path / "meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            zarr_manifest.append({
                'zarr_path': str(zarr_path),
                'slide_id': meta.get('slide_id', zarr_path.stem),
                'label': meta.get('label', 0),
                'num_patches': meta.get('num_patches', 0)
            })

    df_zarr = pd.DataFrame(zarr_manifest)
    print(f"\n✓ Loaded metadata for {len(df_zarr)} Zarr files")
    print(f"Total patches: {df_zarr['num_patches'].sum():,}")

    if train_test_split is None:
        print("⚠️  scikit-learn not available, skipping train/val split")
        return

    if len(df_zarr) > 1 and df_zarr['label'].nunique() > 1:
        train_df, val_df = train_test_split(df_zarr, test_size=0.2, stratify=df_zarr['label'], random_state=42)
    else:
        train_df, val_df = train_test_split(df_zarr, test_size=0.2, random_state=42)

    train_manifest_path = outputs_root / "zarr_train_manifest.csv"
    val_manifest_path = outputs_root / "zarr_val_manifest.csv"
    
    train_df.to_csv(train_manifest_path, index=False)
    val_df.to_csv(val_manifest_path, index=False)
    
    print(f"\n✓ Train manifest saved: {train_manifest_path}")
    print(f"✓ Val manifest saved: {val_manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="HER2 Slide Preprocessing CLI")
    
    # Paths
    parser.add_argument('--data-root', type=str, required=True, help='Root directory containing cohort data')
    parser.add_argument('--outputs-root', type=str, required=True, help='Root directory for outputs')
    parser.add_argument('--zarr-output-dir', type=str, required=True, help='Output directory for Zarr files')
    parser.add_argument('--patches-root', type=str, default=None, help='Directory with existing patches for reference sampling')
    
    # Reference stats
    parser.add_argument('--ref-stats-path', type=str, default=None, help='Path to reference stain stats npz file')
    parser.add_argument('--num-ref-subfolders', type=int, default=100, help='Number of random subfolders to sample for reference')
    parser.add_argument('--images-per-folder', type=int, default=200, help='Max images per sampled subfolder')
    
    # Patch extraction
    parser.add_argument('--patch-size', type=int, default=512, help='Patch size')
    parser.add_argument('--stride', type=int, default=512, help='Patch stride (no overlap if equals patch size)')
    parser.add_argument('--level', type=int, default=0, help='OpenSlide level (0 = highest resolution)')
    parser.add_argument('--tissue-threshold', type=float, default=0.2, help='Minimum tissue fraction in patch')
    parser.add_argument('--downsample-mask', type=int, default=16, help='Downsample factor for mask')
    
    # Performance
    parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=128, help='Patches per batch')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU acceleration for Macenko normalization')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing Zarr files')
    
    # Dataset
    parser.add_argument('--cohorts', type=str, nargs='+', 
                       default=['TCGA_BRCA_Filtered', 'Yale_HER2_cohort', 'Yale_trastuzumab_response_cohort'],
                       help='Cohorts to process')
    
    # Actions
    parser.add_argument('--compute-ref-stats', action='store_true', help='Compute reference stain statistics')
    parser.add_argument('--process-slides', action='store_true', help='Process slides to Zarr')
    parser.add_argument('--create-split', action='store_true', help='Create train/val split manifest')
    
    args = parser.parse_args()
    
    # Setup logging
    outputs_root = Path(args.outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)
    log_dir = outputs_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "preprocess_cli.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*60)
    logging.info("HER2 Slide Preprocessing CLI Started")
    logging.info("="*60)
    logging.info(f"Data root: {args.data_root}")
    logging.info(f"Outputs root: {args.outputs_root}")
    logging.info(f"Zarr output: {args.zarr_output_dir}")
    logging.info(f"Patch size: {args.patch_size}x{args.patch_size}")
    logging.info(f"Workers: {args.num_workers}, GPU: {args.use_gpu}")
    
    # Create output directories
    zarr_output_dir = Path(args.zarr_output_dir)
    zarr_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine ref stats path
    if args.ref_stats_path is None:
        ref_stats_path = outputs_root / "ref_stain_stats.npz"
    else:
        ref_stats_path = Path(args.ref_stats_path)
    
    # Compute reference statistics if requested
    if args.compute_ref_stats:
        if args.patches_root is None:
            logging.error("--patches-root required for --compute-ref-stats")
            return
        
        patches_root = Path(args.patches_root)
        if not patches_root.exists():
            logging.error(f"Patches root not found: {patches_root}")
            return
        
        compute_reference_stats(patches_root, args.num_ref_subfolders, args.images_per_folder,
                              args.use_gpu, ref_stats_path)
    
    # Load reference parameters
    normalizer_params = load_reference_stain_params(ref_stats_path, use_gpu=args.use_gpu)
    if normalizer_params is None:
        logging.warning("⚠️  Reference stain parameters not loaded! Patches will NOT be stain normalized.")
    else:
        logging.info("✓ Loaded reference stain parameters")
        logging.info(f"  Max H: {normalizer_params['max_concentrations'][0]:.4f}")
        logging.info(f"  Max E: {normalizer_params['max_concentrations'][1]:.4f}")
    
    # Process slides if requested
    if args.process_slides:
        # Discover slides
        slides = discover_slides(args.data_root, args.cohorts)
        logging.info(f"\n✓ Discovered {len(slides)} slides across {len(args.cohorts)} cohorts")
        
        # Cohort breakdown
        for cohort in args.cohorts:
            count = sum(1 for s in slides if s['cohort'] == cohort)
            logging.info(f"  {cohort}: {count} slides")
        
        # Label distribution
        if slides:
            labels = [s['label'] for s in slides]
            logging.info(f"Label distribution: HER2- (0): {sum(1 for l in labels if l == 0)}, "
                       f"HER2+ (1): {sum(1 for l in labels if l == 1)}")
        
        # Process slides
        logging.info("\nStarting slide processing...")
        successful = 0
        failed = 0
        skipped = 0
        failed_slides = []
        
        for slide_info in tqdm(slides, desc="Processing slides"):
            zarr_path = zarr_output_dir / f"{slide_info['slide_id']}.zarr"
            if args.skip_existing and zarr_path.exists() and (zarr_path / 'meta.json').exists():
                skipped += 1
                continue
            
            ok = process_slide(
                slide_info=slide_info,
                patch_size=args.patch_size,
                stride=args.stride,
                level=args.level,
                tissue_threshold=args.tissue_threshold,
                downsample_mask=args.downsample_mask,
                normalizer_params=normalizer_params,
                out_dir=zarr_output_dir,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
            )
            
            if ok:
                successful += 1
            else:
                failed += 1
                failed_slides.append(slide_info['slide_id'])
            
            # Periodic cleanup every 3 slides
            if (successful + failed) % 3 == 0:
                gc.collect()
                if args.use_gpu:
                    try:
                        import cupy as cp
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:
                        pass
        
        logging.info(f"\n{'='*60}")
        logging.info("Processing complete!")
        logging.info(f"  Successful: {successful}")
        logging.info(f"  Failed: {failed}")
        logging.info(f"  Skipped (existing): {skipped}")
        logging.info(f"  Total: {len(slides)}")
        if failed_slides:
            logging.info(f"Failed slides: {', '.join(failed_slides[:10])}{'...' if len(failed_slides) > 10 else ''}")
        logging.info(f"{'='*60}")
    
    # Create train/val split if requested
    if args.create_split:
        create_train_val_split(zarr_output_dir, outputs_root)
    
    logging.info("\n✅ Preprocessing pipeline complete!")


if __name__ == "__main__":
    main()

