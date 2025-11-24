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
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm
import zarr
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

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

# Use the tested Macenko implementation from src/preprocessing
try:
    from src.preprocessing.stain_normalization import MacenkoNormalizer
except Exception as e:  # fallback: no stain normalization available
    MacenkoNormalizer = None
    logging.warning(f"Could not import MacenkoNormalizer from src.preprocessing.stain_normalization: {e}")


def load_reference_stain_params(npz_path: Path) -> Optional[Dict]:
    """Load precomputed reference stain parameters from npz file (CPU format)."""
    if not npz_path.exists():
        return None
    try:
        data = np.load(str(npz_path))
        V = data["stain_vectors"]
        max_h = float(data.get("max_h", data.get("mean_max_h")))
        max_e = float(data.get("max_e", data.get("mean_max_e")))
        return {
            "stain_vectors": V,
            "max_concentrations": (max_h, max_e),
        }
    except Exception as e:
        logging.error(f"Failed to load reference stain parameters from {npz_path}: {e}")
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


def compute_reference_stats_from_patches(patches_root: Path, num_subfolders: int, images_per_folder: int,
                           use_gpu: bool, output_path: Path):
    """Compute reference stain statistics from sample patches."""
    if MacenkoNormalizer is None:
        logging.error("MacenkoNormalizer unavailable; cannot compute reference stats")
        return

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
    """Discover SVS slides and corresponding XML annotations with correct labels.

    Labeling rules (per user spec):
    - If slide_id starts with 'Her2Pos_': label = 1
    - If slide_id starts with 'Her2Neg_': label = 0
    - If slide_id starts with 'TCGA-':
        * Take substring before first '.' (if any) as key
        * Look up in data/TCGA_BRCA_Filtered/HER2_TCGA_clean.csv
        * Use 'Slide' column to match key, and 'Clinical.HER2.status' as status column
        * Map status: Positive/3+/2+ → 1, everything else → 0
    - All other slides: label = 1 (positive)
    """
    slides: List[Dict] = []

    # Pre-load TCGA HER2 table once
    tcga_csv = Path(data_root) / "TCGA_BRCA_Filtered" / "HER2_TCGA_clean.csv"
    tcga_map: Dict[str, int] = {}
    if tcga_csv.exists():
        try:
            df_tcga = pd.read_csv(tcga_csv)
            # Expect columns 'Slide' and 'Clinical.HER2.status'
            if "Slide" in df_tcga.columns and "Clinical.HER2.status" in df_tcga.columns:
                def map_tcga_status(s):
                    if isinstance(s, str):
                        s_low = s.lower()
                        if "positive" in s_low or "3+" in s_low or "2+" in s_low:
                            return 1
                    return 0

                for _, row in df_tcga.iterrows():
                    slide_key = str(row["Slide"]).strip()
                    status = row["Clinical.HER2.status"]
                    tcga_map[slide_key] = map_tcga_status(status)
            else:
                logging.warning(
                    "TCGA HER2 file %s does not contain expected columns 'Slide' and 'Clinical.HER2.status'",
                    tcga_csv,
                )
        except Exception as e:
            logging.error(f"Failed to load TCGA HER2 table from {tcga_csv}: {e}")

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

        # SVS files
        svs_files = list(svs_dir.glob("*.svs")) + list(svs_dir.glob("*.SVS"))

        for svs_path in svs_files:
            slide_name = svs_path.stem

            # XML lookup: for TCGA use only part before first dot
            xml_base_name = slide_name
            if slide_name.startswith("TCGA-") and "." in slide_name:
                xml_base_name = slide_name.split(".")[0]

            xml_candidates = [
                xml_dir / f"{xml_base_name}.xml",
                xml_dir / f"{xml_base_name}.XML",
                xml_dir / f"{slide_name}.xml",
                xml_dir / f"{slide_name}.XML",
            ]

            xml_path = None
            for cand in xml_candidates:
                if cand.exists():
                    xml_path = cand
                    break

            if xml_path is None:
                print(f"⚠️  No XML found for {slide_name} (tried {xml_base_name}), skipping")
                continue

            # Labeling rules
            if slide_name.startswith("Her2Pos_"):
                label = 1
            elif slide_name.startswith("Her2Neg_"):
                label = 0
            elif slide_name.startswith("TCGA-"):
                # Use portion before first '.' as key
                key = slide_name.split(".")[0]
                label = tcga_map.get(key, 1)  # default 1 if not found per user: others positive
            else:
                # All other slides are positive cases
                label = 1

            slides.append(
                {
                    "slide_id": slide_name,
                    "svs_path": str(svs_path),
                    "xml_path": str(xml_path),
                    "cohort": cohort,
                    "label": int(label),
                }
            )

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


NORMALIZER_PARAMS_GLOBAL = None


def _init_process_pool(normalizer_params):
    global NORMALIZER_PARAMS_GLOBAL
    NORMALIZER_PARAMS_GLOBAL = normalizer_params


def _process_slide_task(payload):
    slide_info, shared_kwargs, env_gpu = payload
    if env_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = env_gpu
    kwargs = dict(shared_kwargs)
    kwargs["normalizer_params"] = NORMALIZER_PARAMS_GLOBAL
    ok = process_slide(slide_info=slide_info, **kwargs)
    return slide_info["slide_id"], ok


def process_slide(
    slide_info: dict,
    patch_size: int,
    stride: int,
    level: int,
    tissue_threshold: float,
    downsample_mask: int,
    normalizer_params: Optional[Dict],
    out_dir: Path,
    num_workers: int = 4,
    batch_size: int = 128,
    skip_existing: bool = True,
    use_gpu: bool = False,
) -> bool:
    """Process a single slide: extract patches, optionally stain-normalize, and save to Zarr.

    normalizer_params: dict with keys 'stain_vectors' and 'max_concentrations' or None.
    """

    slide_id = slide_info["slide_id"]
    svs_path = slide_info["svs_path"]
    xml_path = slide_info["xml_path"]
    label = int(slide_info["label"])

    zarr_path = out_dir / f"{slide_id}.zarr"

    if skip_existing and zarr_path.exists() and (zarr_path / "meta.json").exists():
        logging.info(f"Skip existing: {slide_id}")
        return True

    slide = None
    try:
        slide = openslide.OpenSlide(svs_path)
        W, H = slide.level_dimensions[0]

        # XML → polygons → mask
        polygons = parse_xml_polygons(xml_path)
        if not polygons:
            logging.warning(f"No polygons in XML for {slide_id}; skipping")
            slide.close()
            return False

        mask = polygons_to_mask(polygons, (W, H), downsample=downsample_mask)
        del polygons

        # Grid of centers at level 0
        centers = generate_grid_centers((W, H), patch_size, stride)
        ds = downsample_mask
        valid_centers = [
            (x, y)
            for (x, y) in centers
            if mask[min(H // ds - 1, y // ds), min(W // ds - 1, x // ds)]
        ]
        del centers, mask

        if not valid_centers:
            logging.warning(f"No valid centers after masking for {slide_id}; skipping")
            slide.close()
            return False

        # Prepare stain normalizer for this slide if params and implementation available
        macenko = None
        V_ref = None
        ref_max = None
        if normalizer_params is not None and MacenkoNormalizer is not None:
            V_ref = normalizer_params["stain_vectors"]
            ref_max = normalizer_params["max_concentrations"]
            macenko = MacenkoNormalizer(use_gpu=use_gpu)
            logging.info("Slide %s: initializing MacenkoNormalizer (GPU=%s)", slide_id, use_gpu)

        # Pre-create zarr arrays
        z = create_zarr_group(zarr_path, len(valid_centers), patch_size)

        # Metadata from slide
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        try:
            magnification = float(
                slide.properties.get("aperio.AppMag")
                or slide.properties.get("openslide.objective-power")
                or 0
            )
        except Exception:
            magnification = None

        def read_patch(idx_center_pair):
            """Read patch from slide, apply quick tissue filter only."""
            _, (cx, cy) = idx_center_pair
            x0 = cx - patch_size // 2
            y0 = cy - patch_size // 2

            region = slide.read_region((x0, y0), level, (patch_size, patch_size))
            try:
                patch = np.array(region.convert("RGB"))
            finally:
                region.close()

            if tissue_threshold > 0 and tissue_fraction_rgb(patch) < tissue_threshold:
                return None, None, None

            return patch, (cx, cy), label

        total = len(valid_centers)
        written = 0
        normalized_patches_count = 0

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for start in tqdm(
                range(0, total, batch_size),
                total=math.ceil(total / batch_size),
                desc=f"{slide_id}",
                leave=False,
            ):
                end = min(start + batch_size, total)
                futures = [
                    ex.submit(read_patch, (i, valid_centers[i])) for i in range(start, end)
                ]

                batch_patches: List[np.ndarray] = []
                batch_coords: List[Tuple[int, int]] = []
                batch_labels: List[int] = []

                for fut in as_completed(futures):
                    patch, coord, lbl = fut.result()
                    if patch is None:
                        continue
                    batch_patches.append(patch)
                    batch_coords.append(coord)
                    batch_labels.append(lbl)

                if not batch_patches:
                    continue

                # Stain normalization (CPU/GPU handled inside MacenkoNormalizer)
                if macenko is not None and V_ref is not None and ref_max is not None:
                    norm_patches = []
                    for p in batch_patches:
                        try:
                            p_norm = macenko.normalize(
                                p,
                                mean_ref_stain_vectors=V_ref,
                                mean_ref_max_concentrations_tuple=ref_max,
                            )
                            norm_patches.append(p_norm)
                            normalized_patches_count += 1
                        except Exception as e:
                            logging.debug(f"Normalization failed for slide {slide_id}: {e}")
                            norm_patches.append(p)
                    batch_patches = norm_patches

                # Write batch
                n = len(batch_patches)
                if n > 0:
                    z["patches"][written : written + n] = np.stack(batch_patches, axis=0)
                    z["coords"][written : written + n] = np.asarray(batch_coords, dtype=np.int32)
                    z["labels"][written : written + n] = np.asarray(batch_labels, dtype=np.int8)
                    written += n

                del batch_patches, batch_coords, batch_labels
                gc.collect()

        # Resize datasets down to actual number written
        try:
            z["patches"].resize((written, patch_size, patch_size, 3))
            z["coords"].resize((written, 2))
            z["labels"].resize((written,))
        except Exception as e:
            logging.debug(f"Resize failed for {slide_id}: {e}")

        meta = {
            "slide_id": slide_id,
            "label": label,
            "num_patches": int(written),
            "mpp_x": float(mpp_x) if mpp_x else None,
            "mpp_y": float(mpp_y) if mpp_y else None,
            "magnification": magnification,
            "patch_size": patch_size,
            "stride": stride,
            "level": level,
            "stain_normalized": bool(macenko is not None),
        }
        with open(zarr_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logging.info(
            "Slide %s: normalized %d/%d patches (GPU=%s)",
            slide_id,
            normalized_patches_count,
            int(written),
            use_gpu and (macenko is not None),
        )

        slide.close()
        del valid_centers, z, macenko, V_ref, ref_max
        gc.collect()
        return True

    except Exception as e:
        logging.error(f"Exception processing slide {slide_id}: {e}", exc_info=True)
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass
        gc.collect()
        return False


def sample_reference_patches_from_slide(
    slide_info: Dict,
    patch_size: int,
    patches_per_slide: int,
    downsample_mask: int,
    tissue_threshold: float = 0.0,
) -> List[np.ndarray]:
    """Sample a limited number of patches from a slide (entire-slide sampling)."""
    patches: List[np.ndarray] = []
    slide = None
    try:
        slide = openslide.OpenSlide(slide_info["svs_path"])
        W, H = slide.level_dimensions[0]
        polygons = parse_xml_polygons(slide_info["xml_path"])
        if not polygons:
            return patches
        mask = polygons_to_mask(polygons, (W, H), downsample=downsample_mask)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return patches
        rng = np.random.default_rng()
        count = min(patches_per_slide, coords.shape[0])
        replace = coords.shape[0] < patches_per_slide
        idx = rng.choice(coords.shape[0], size=count, replace=replace)
        selected = coords[idx]
        half = patch_size // 2
        for (y_ds, x_ds) in selected:
            xc = int((x_ds + 0.5) * downsample_mask)
            yc = int((y_ds + 0.5) * downsample_mask)
            xc = min(max(xc, half), W - half)
            yc = min(max(yc, half), H - half)
            region = slide.read_region((xc - half, yc - half), 0, (patch_size, patch_size))
            try:
                patch = np.array(region.convert("RGB"))
            finally:
                region.close()
            if tissue_threshold > 0 and tissue_fraction_rgb(patch) < tissue_threshold:
                continue
            patches.append(patch)
            if len(patches) >= patches_per_slide:
                break
    except Exception as exc:
        logging.debug(f"Reference sampling failed for {slide_info['slide_id']}: {exc}")
    finally:
        if slide is not None:
            slide.close()
    return patches


def compute_reference_stats_from_slides(
    data_root: Path,
    cohorts: List[str],
    num_slides: int,
    patches_per_slide: int,
    patch_size: int,
    downsample_mask: int,
    use_gpu: bool,
    output_path: Path,
):
    """Compute reference stats by sampling patches directly from whole slides."""
    if MacenkoNormalizer is None:
        logging.error("MacenkoNormalizer unavailable; cannot compute reference stats")
        return
    slides = discover_slides(str(data_root), cohorts)
    if not slides:
        logging.error("No slides discovered for reference stats.")
        return
    random.seed(42)
    random.shuffle(slides)
    sampled_slides = slides[: max(1, min(num_slides, len(slides)))]
    total_target = max(1, len(sampled_slides) * patches_per_slide)
    logging.info(
        "Sampling reference patches from %d slides (%d per slide, patch=%d)",
        len(sampled_slides),
        patches_per_slide,
        patch_size,
    )
    all_reference_images: List[np.ndarray] = []
    for slide_info in tqdm(sampled_slides, desc="Reference slides"):
        patches = sample_reference_patches_from_slide(
            slide_info,
            patch_size=patch_size,
            patches_per_slide=patches_per_slide,
            downsample_mask=downsample_mask,
        )
        if not patches:
            continue
        all_reference_images.extend(patches)
        del patches
        if len(all_reference_images) >= total_target:
            break
        if len(all_reference_images) % 100 == 0:
            gc.collect()
    logging.info("Collected %d reference patches", len(all_reference_images))
    if not all_reference_images:
        logging.error("No reference patches collected; cannot compute stats")
        return
    normalizer = MacenkoNormalizer(use_gpu=use_gpu, percentiles=(1, 99))
    mean_V, (mean_max_h, mean_max_e) = normalizer.get_mean_reference_stain_characteristics(all_reference_images)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, stain_vectors=mean_V, max_h=mean_max_h, max_e=mean_max_e)
    logging.info(
        "Reference stats saved to %s | Max H: %.4f Max E: %.4f",
        output_path,
        mean_max_h,
        mean_max_e,
    )
    del all_reference_images, mean_V, normalizer
    gc.collect()


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


def load_processed_slides_log(log_path: Path) -> set:
    """Load set of slide_ids that have been marked as processed in a CSV log.

    The CSV has at least a 'slide_id' column. If the file doesn't exist, returns an empty set.
    This log is independent of Macenko on/off; if a slide_id is present, it is considered processed.
    """
    processed = set()
    if not log_path.exists():
        return processed
    try:
        with log_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if "slide_id" not in reader.fieldnames:
                return processed
            for row in reader:
                sid = row.get("slide_id")
                if sid:
                    processed.add(str(sid))
    except Exception as e:
        logging.warning("Could not read processed slides log %s: %s", log_path, e)
    return processed


def append_processed_slide(log_path: Path, slide_info: Dict):
    """Append a single slide entry to the processed-slides CSV log.

    Columns: slide_id, cohort, label.
    Creates the file with header if it does not exist.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    try:
        with log_path.open("a", newline="") as f:
            fieldnames = ["slide_id", "cohort", "label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "slide_id": slide_info.get("slide_id"),
                    "cohort": slide_info.get("cohort"),
                    "label": slide_info.get("label"),
                }
            )
    except Exception as e:
        logging.warning("Failed to append to processed slides log %s: %s", log_path, e)


def main():
    parser = argparse.ArgumentParser(description="HER2 Slide Preprocessing CLI")
    
    # Paths
    parser.add_argument('--data-root', type=str, required=True, help='Root directory containing cohort data')
    parser.add_argument('--outputs-root', type=str, required=True, help='Root directory for outputs')
    parser.add_argument('--zarr-output-dir', type=str, required=True, help='Output directory for Zarr files')
    parser.add_argument('--patches-root', type=str, default=None, help='Directory with existing patches for reference sampling')
    
    # Reference stats
    parser.add_argument('--ref-stats-path', type=str, default=None, help='Path to reference stain stats npz file')
    parser.add_argument('--num-ref-subfolders', type=int, default=100, help='Number of random subfolders to sample for reference (patch mode)')
    parser.add_argument('--images-per-folder', type=int, default=200, help='Max images per sampled subfolder (patch mode)')
    parser.add_argument('--ref-mode', type=str, choices=['patches', 'slides'], default='patches', help='How to compute reference stats: patches (legacy) or slides (sample patches directly from SVS)')
    parser.add_argument('--ref-slides', type=int, default=40, help='Number of slides to sample for reference stats when --ref-mode=slides')
    parser.add_argument('--ref-patches-per-slide', type=int, default=50, help='Number of patches sampled per slide for reference stats when --ref-mode=slides')

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
    parser.add_argument('--disable-macenko', action='store_true', help='Force-disable Macenko stain normalization even if reference stats exist')
    parser.add_argument('--slide-fraction', type=float, default=1.0, help='Fraction of discovered slides to process (0-1], default 1.0')
    parser.add_argument('--slide-fraction-seed', type=int, default=42, help='Random seed used when subsampling slides via --slide-fraction')
    parser.add_argument('--slide-workers', type=int, default=1, help='Number of processes for slide-level parallelism (>=1)')
    parser.add_argument('--slide-worker-gpus', type=str, default=None, help='Comma-separated GPU ids to assign per slide worker (e.g., "0,1,2"). Only used when --slide-workers > 1.')

    # Dataset
    parser.add_argument('--cohorts', type=str, nargs='+', 
                       default=['TCGA_BRCA_Filtered', 'Yale_HER2_cohort', 'Yale_trastuzumab_response_cohort'],
                       help='Cohorts to process')
    
    # Actions
    parser.add_argument('--compute-ref-stats', action='store_true', help='Compute reference stain statistics')
    parser.add_argument('--process-slides', action='store_true', help='Process slides to Zarr')
    parser.add_argument('--create-split', action='store_true', help='Create train/val split manifest')
    parser.add_argument('--ref-data-root', type=str, default=None, help='Alternative data root for reference stat computation when --ref-mode=slides (defaults to --data-root)')

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
        if args.ref_mode == 'patches':
            if args.patches_root is None:
                logging.error("--patches-root required for ref_mode=patches")
                return
            patches_root = Path(args.patches_root)
            if not patches_root.exists():
                logging.error(f"Patches root not found: {patches_root}")
                return
            compute_reference_stats_from_patches(
                patches_root,
                args.num_ref_subfolders,
                args.images_per_folder,
                args.use_gpu,
                ref_stats_path,
            )
        else:
            ref_data_root = Path(args.ref_data_root) if args.ref_data_root else Path(args.data_root)
            if not ref_data_root.exists():
                logging.error(f"Reference data root not found: {ref_data_root}")
                return
            compute_reference_stats_from_slides(
                data_root=ref_data_root,
                cohorts=args.cohorts,
                num_slides=args.ref_slides,
                patches_per_slide=args.ref_patches_per_slide,
                patch_size=args.patch_size,
                downsample_mask=args.downsample_mask,
                use_gpu=args.use_gpu,
                output_path=ref_stats_path,
            )

    if args.disable_macenko:
        logging.info("Macenko stain normalization DISABLED via --disable-macenko")
        normalizer_params = None
    else:
        normalizer_params = load_reference_stain_params(ref_stats_path)
        if normalizer_params is None:
            logging.warning(
                "⚠️  Reference stain parameters not loaded! Patches will NOT be stain normalized."
            )
        else:
            logging.info("✓ Loaded reference stain parameters")
            logging.info(
                f"  Max H: {normalizer_params['max_concentrations'][0]:.4f}, "
                f"Max E: {normalizer_params['max_concentrations'][1]:.4f}"
            )
            logging.info("Macenko stain normalization ENABLED (GPU=%s)", args.use_gpu)

    # Process slides if requested
    if args.process_slides:
        slides = discover_slides(args.data_root, args.cohorts)
        logging.info(
            f"\n✓ Discovered {len(slides)} slides across {len(args.cohorts)} cohorts"
        )
        for cohort in args.cohorts:
            count = sum(1 for s in slides if s["cohort"] == cohort)
            logging.info(f"  {cohort}: {count} slides")

        if slides:
            labels = [s["label"] for s in slides]
            logging.info(
                "Label distribution: HER2- (0): %d, HER2+ (1): %d",
                sum(1 for l in labels if l == 0),
                sum(1 for l in labels if l == 1),
            )

        # Load processed-slides CSV log (independent of Macenko flag)
        processed_log_path = Path(args.outputs_root) / "preprocessing" / "processed_slides.csv"
        processed_slides = load_processed_slides_log(processed_log_path)
        if processed_slides:
            logging.info("Loaded %d previously processed slides from %s", len(processed_slides), processed_log_path)

        # Subsample slides if fraction < 1.0
        slide_fraction = max(0.0, min(1.0, args.slide_fraction))
        if slide_fraction <= 0:
            logging.warning("Slide fraction <= 0; nothing to process.")
            slides = []
        elif slide_fraction < 1.0 and slides:
            random.seed(args.slide_fraction_seed)
            sample_size = max(1, int(round(len(slides) * slide_fraction)))
            sample_size = min(sample_size, len(slides))
            slides = random.sample(slides, sample_size)
            logging.info(f"✓ Subsampled to {len(slides)} slides ({slide_fraction*100:.1f}%)")

        logging.info("\nStarting slide processing...")
        successful = failed = skipped = 0
        failed_slides: List[str] = []

        slide_worker_count = max(1, args.slide_workers)
        gpu_ids = []
        if args.slide_worker_gpus:
            gpu_ids = [g.strip() for g in args.slide_worker_gpus.split(',') if g.strip()]
        gpu_assign = []
        if gpu_ids:
            for idx in range(slide_worker_count):
                gpu_assign.append(gpu_ids[idx % len(gpu_ids)])

        shared_kwargs = dict(
            patch_size=args.patch_size,
            stride=args.stride,
            level=args.level,
            tissue_threshold=args.tissue_threshold,
            downsample_mask=args.downsample_mask,
            out_dir=Path(args.zarr_output_dir),
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
            use_gpu=args.use_gpu,
        )

        if slide_worker_count == 1:
            for slide_info in tqdm(slides, desc="Processing slides"):
                slide_id = slide_info["slide_id"]
                if slide_id in processed_slides:
                    skipped += 1
                    continue

                zarr_path = Path(args.zarr_output_dir) / f"{slide_id}.zarr"
                if args.skip_existing and zarr_path.exists() and (zarr_path / "meta.json").exists():
                    skipped += 1
                    # Also record in CSV log so future runs know it's processed, regardless of Macenko flag
                    append_processed_slide(processed_log_path, slide_info)
                    processed_slides.add(slide_id)
                    continue

                ok = process_slide(
                    slide_info=slide_info,
                    normalizer_params=normalizer_params,
                    **shared_kwargs,
                )

                if ok:
                    successful += 1
                    append_processed_slide(processed_log_path, slide_info)
                    processed_slides.add(slide_id)
                else:
                    failed += 1
                    failed_slides.append(slide_id)

                if (successful + failed) % 3 == 0:
                    gc.collect()
        else:
            payloads = []
            for idx, slide_info in enumerate(slides):
                slide_id = slide_info["slide_id"]
                if slide_id in processed_slides:
                    skipped += 1
                    continue

                zarr_path = Path(args.zarr_output_dir) / f"{slide_id}.zarr"
                if args.skip_existing and zarr_path.exists() and (zarr_path / "meta.json").exists():
                    skipped += 1
                    append_processed_slide(processed_log_path, slide_info)
                    processed_slides.add(slide_id)
                    continue

                env_gpu = gpu_assign[idx % len(gpu_assign)] if gpu_assign else None
                payloads.append((slide_info, shared_kwargs, env_gpu))

            with ProcessPoolExecutor(max_workers=slide_worker_count, initializer=_init_process_pool, initargs=(normalizer_params,)) as pool:
                futures = [pool.submit(_process_slide_task, payload) for payload in payloads]

                for fut in tqdm(as_completed(futures), total=len(futures), desc="Slides (mp)"):
                    slide_id, ok = fut.result()
                    if ok:
                        successful += 1
                        # In mp mode we only know slide_id, so synthesize a minimal info dict
                        append_processed_slide(processed_log_path, {"slide_id": slide_id, "cohort": "", "label": ""})
                        processed_slides.add(slide_id)
                    else:
                        failed += 1
                        failed_slides.append(slide_id)

        logging.info("\n" + "=" * 60)
        logging.info("Processing complete!")
        logging.info(f"  Successful: {successful}")
        logging.info(f"  Failed: {failed}")
        logging.info(f"  Skipped (existing or logged): {skipped}")
        logging.info(f"  Total considered: {len(slides)}")
        if failed_slides:
            logging.info(
                "Failed slides: %s",
                ", ".join(failed_slides[:10])
                + ("..." if len(failed_slides) > 10 else ""),
            )
        logging.info("=" * 60)

    if args.create_split:
        create_train_val_split(Path(args.zarr_output_dir), Path(args.outputs_root))

    logging.info("\n✅ Preprocessing pipeline complete!")


if __name__ == "__main__":
    main()
