import numpy as np
import gc
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from src.dataloader.wsi_reader import WSIReader
from src.preprocessing.macenko import estimate_reference_stain_vectors
from src.preprocessing.xml_to_mask import get_mask
from src.pipeline.logging_utils import setup_logging
import src.pipeline.config as config

logger = setup_logging("normalization")

def compute_reference_vectors(paths_df, output_path: Path = None):
    """
    Collects random patches from slides and computes global reference stain vectors.
    """
    all_patches_for_reference = []
    successful_slides = 0
    
    logger.info(f"Collecting reference patches from {len(paths_df)} slides...")
    
    for idx, row in tqdm(paths_df.iterrows(), total=len(paths_df), desc="Collecting reference patches"):
        path = row['full_path']
        xml_path = row.get('full_path_annotation')
        
        try:
            # Generate mask if ROI filtering is enabled and annotation exists
            mask = None
            if config.FILTER_BY_ROI and pd.notna(xml_path):
                try:
                    mask = get_mask(xml_path, path, downsample_factor=config.MASK_DOWNSAMPLE)
                except Exception as e:
                    logger.warning(f"Failed to generate mask for {path}: {e}")
                    mask = None

            with WSIReader(path, prefer_cucim=config.USE_CUCIM) as reader:
                width, height = reader.dimensions
                np.random.seed(config.SEED + idx)
                
                patches_from_slide = 0
                
                # Determine sampling locations
                if mask is not None:
                    # Find valid coordinates in the mask
                    ys, xs = np.where(mask > 0)
                    if len(xs) > 0:
                        # Sample indices
                        n_samples = min(len(xs), config.PATCHES_PER_SLIDE)
                        indices = np.random.choice(len(xs), n_samples, replace=False)
                        sample_xs = xs[indices] * config.MASK_DOWNSAMPLE
                        sample_ys = ys[indices] * config.MASK_DOWNSAMPLE
                    else:
                        logger.warning(f"Mask for {path} is empty. Falling back to random sampling.")
                        sample_xs = []
                        sample_ys = []
                else:
                    # Random sampling
                    sample_xs = np.random.randint(0, max(1, width - config.PATCH_SIZE), config.PATCHES_PER_SLIDE)
                    sample_ys = np.random.randint(0, max(1, height - config.PATCH_SIZE), config.PATCHES_PER_SLIDE)

                # Extract patches
                for x, y in zip(sample_xs, sample_ys):
                    try:
                        # Ensure coordinates are within bounds
                        x = int(np.clip(x, 0, width - config.PATCH_SIZE))
                        y = int(np.clip(y, 0, height - config.PATCH_SIZE))
                        
                        patch = reader.read_region(x=x, y=y, level=config.LEVEL_REFERENCE, size=config.PATCH_SIZE)
                        
                        # Filter out low-quality patches
                        if patch.std() > 5:
                            all_patches_for_reference.append(patch)
                            patches_from_slide += 1
                    except Exception:
                        continue
                
                if patches_from_slide > 0:
                    successful_slides += 1
            
            del mask
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to read slide {path}: {e}")
            pass

    logger.info(f"Reference collection complete. Slides: {successful_slides}/{len(paths_df)}, Patches: {len(all_patches_for_reference)}")

    reference_stain_vectors = None
    if len(all_patches_for_reference) > 0:
        try:
            reference_stain_vectors = estimate_reference_stain_vectors(all_patches_for_reference)
            logger.info(f"Reference stain vectors computed: {reference_stain_vectors.shape}")
        except IndexError as e:
            logger.warning(f"Could not compute reference vectors: {e}")
    else:
        logger.error("No patches collected for reference!")

    # Cleanup
    all_patches_for_reference.clear()
    gc.collect()

    # Save
    if output_path is None:
        output_path = config.OUTPUT_BASE / 'reference_stain_vectors.npy'
        
    if reference_stain_vectors is not None:
        np.save(output_path, reference_stain_vectors)
        logger.info(f"Reference stain vectors saved to: {output_path}")
    
    return reference_stain_vectors
