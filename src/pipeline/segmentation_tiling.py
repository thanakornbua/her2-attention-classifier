import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import cv2
import gc
import os

# Try importing openslide for metadata
try:
    import openslide
except ImportError:
    openslide = None

from src.dataloader.wsi_reader import WSIReader
from src.preprocessing.xml_to_mask import get_mask
from src.pipeline.logging_utils import setup_logging
import src.pipeline.config as config

logger = setup_logging("segmentation_tiling")

def extract_segmentation_patches(
    slides_df: pd.DataFrame,
    output_path: Path,
    level: int = 1,
    patch_size: int = 256,
    stride: int = 256
):
    """
    Extracts patches and corresponding masks from WSIs at a specific level for segmentation training.
    Only extracts patches that contain tumor annotation (positive masks).
    
    Args:
        slides_df: DataFrame containing 'full_path' and 'full_path_annotation'.
        output_path: Path to the output Zarr file (e.g. 'data/unet_patches.zarr').
        level: WSI pyramid level to extract from (default 1).
        patch_size: Size of the patch in pixels at the target level.
        stride: Stride for extraction in pixels at the target level.
    """
    if openslide is None:
        logger.error("OpenSlide is required for level metadata reading.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize Zarr
    store = zarr.open(str(output_path), mode='w')
    
    # Compressor
    compressor = config.ZARR_COMPRESSOR
    
    # Create arrays (resizable)
    # Images: (N, H, W, 3) uint8
    if 'images' not in store:
        images_ds = store.create_dataset(
            'images', shape=(0, patch_size, patch_size, 3), chunks=(32, patch_size, patch_size, 3),
            dtype='uint8', compressor=compressor
        )
    else:
        images_ds = store['images']

    # Masks: (N, H, W) uint8
    if 'masks' not in store:
        masks_ds = store.create_dataset(
            'masks', shape=(0, patch_size, patch_size), chunks=(32, patch_size, patch_size),
            dtype='uint8', compressor=compressor
        )
    else:
        masks_ds = store['masks']
    
    buffer_images = []
    buffer_masks = []
    BUFFER_SIZE = 64
    
    total_patches = 0
    
    for idx, row in tqdm(slides_df.iterrows(), total=len(slides_df), desc=f"Extracting Level {level} Patches"):
        wsi_path = str(row['full_path'])
        xml_path = row.get('full_path_annotation')
        
        # Skip if no annotation
        if pd.isna(xml_path) or not os.path.exists(str(xml_path)):
            continue
            
        try:
            # Get Level Metadata
            slide = openslide.OpenSlide(wsi_path)
            try:
                if level >= slide.level_count:
                    logger.warning(f"Level {level} not available for {Path(wsi_path).name} (max {slide.level_count-1}). Skipping.")
                    continue
                
                ds_rate = slide.level_downsamples[level]
                w_level, h_level = slide.level_dimensions[level]
            finally:
                slide.close()
            
            # Generate mask at the target level resolution
            # get_mask uses downsample_factor relative to level 0
            mask_level = get_mask(xml_path, wsi_path, downsample_factor=ds_rate)
            
            if mask_level is None:
                continue
                
            # Ensure mask matches level dimensions (get_mask might be slightly off due to rounding)
            # Resize if necessary (nearest neighbor for masks)
            if mask_level.shape != (h_level, w_level):
                mask_level = cv2.resize(mask_level, (w_level, h_level), interpolation=cv2.INTER_NEAREST)
            
            # Grid generation
            ys = range(0, h_level - patch_size + 1, stride)
            xs = range(0, w_level - patch_size + 1, stride)
            
            with WSIReader(wsi_path) as reader:
                for y in ys:
                    for x in xs:
                        # Check mask content
                        mask_patch = mask_level[y:y+patch_size, x:x+patch_size]
                        
                        # Keep if it contains annotation (tumor)
                        # Adjust threshold as needed. Here > 0 pixels.
                        if np.any(mask_patch > 0):
                            # Extract Image
                            # WSIReader expects x, y in Level 0
                            x0 = int(x * ds_rate)
                            y0 = int(y * ds_rate)
                            
                            img_patch = reader.read_region(x0, y0, level, patch_size)
                            
                            # Check if image read was successful and shape is correct
                            if img_patch.shape != (patch_size, patch_size, 3):
                                continue

                            buffer_images.append(img_patch)
                            buffer_masks.append(mask_patch)
                            
                            if len(buffer_images) >= BUFFER_SIZE:
                                images_ds.append(np.stack(buffer_images))
                                masks_ds.append(np.stack(buffer_masks))
                                total_patches += len(buffer_images)
                                buffer_images = []
                                buffer_masks = []
            
            del mask_level
            gc.collect()
                                
        except Exception as e:
            logger.error(f"Error processing {Path(wsi_path).name}: {e}")
            continue
            
    # Final flush
    if buffer_images:
        images_ds.append(np.stack(buffer_images))
        masks_ds.append(np.stack(buffer_masks))
        total_patches += len(buffer_images)
        
    logger.info(f"Extraction complete. Total patches: {total_patches}")
    logger.info(f"Saved to {output_path}")
