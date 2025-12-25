import numpy as np
import pandas as pd
import zarr
import gc
from pathlib import Path
from tqdm.auto import tqdm
from src.dataloader.wsi_reader import WSIReader
from src.preprocessing.xml_to_mask import get_mask
from src.preprocessing.macenko import macenko_stain_normalize
from src.pipeline.logging_utils import setup_logging
import src.pipeline.config as config

logger = setup_logging("tiling")

# Try to import cupy for GPU acceleration
try:
    import cupy as xp
except ImportError:
    import numpy as xp

class TilingPipeline:
    def __init__(self, ref_vectors=None):
        self.ref_vectors = ref_vectors
        self.zarr_state = {}
        self.buffers = {'norm': [], 'raw': []}
        self.metadata = {'norm': self._init_metadata(), 'raw': self._init_metadata()}
        self.counters = {'norm': 0, 'raw': 0}
        
        # Initialize Zarr stores
        self._init_zarr_stores()

    def _init_metadata(self):
        return {
            "slide_name": [], "case_name": [], "label": [],
            "patch_global_index": [], "patch_in_slide": [],
            "x": [], "y": []
        }

    def _open_zarr_store(self, path_primary: Path, path_fallback: Path):
        try:
            store = zarr.open(str(path_primary), mode="w", shape=(0, config.PATCH_SIZE_TRAIN, config.PATCH_SIZE_TRAIN, 3), 
                              chunks=config.PATCH_CHUNK, dtype="uint8", compressor=config.ZARR_COMPRESSOR, zarr_version=2)
            return store, path_primary
        except Exception as e:
            logger.warning(f"Primary Zarr open failed at {path_primary}: {e}. Using fallback {path_fallback}")
            store = zarr.open(str(path_fallback), mode="w", shape=(0, config.PATCH_SIZE_TRAIN, config.PATCH_SIZE_TRAIN, 3), 
                              chunks=config.PATCH_CHUNK, dtype="uint8", compressor=config.ZARR_COMPRESSOR, zarr_version=2)
            return store, path_fallback

    def _init_zarr_stores(self):
        store_norm, path_norm = self._open_zarr_store(config.PATCH_ZARR_OUTPUT_NORM, config.PATCH_ZARR_OUTPUT_NORM_FALLBACK)
        store_raw, path_raw = self._open_zarr_store(config.PATCH_ZARR_OUTPUT_RAW, config.PATCH_ZARR_OUTPUT_RAW_FALLBACK)
        
        self.zarr_state = {
            'patch_store_norm': store_norm,
            'store_norm_path': path_norm,
            'patch_store_raw': store_raw,
            'store_raw_path': path_raw,
        }

    def _normalize_patch(self, patch_uint8):
        if self.ref_vectors is None:
            return macenko_stain_normalize(patch_uint8, reference_stain_vectors=None)
        
        ref = xp.asarray(self.ref_vectors, dtype=xp.float32)
        arr = xp.asarray(patch_uint8, dtype=xp.float32)
        od = -xp.log((arr + 1.0) / 256.0)
        od_flat = od.reshape(-1, 3)
        conc = xp.linalg.lstsq(ref.T, od_flat.T, rcond=None)[0].T
        od_norm = conc @ ref
        rgb = xp.exp(-od_norm) * 256.0
        rgb = xp.clip(rgb, 0, 255).reshape(patch_uint8.shape).astype(xp.uint8)
        return rgb.get() if hasattr(rgb, "get") else rgb

    def _flush_buffer(self, label, force=False):
        buf = self.buffers[label]
        if not buf:
            return
            
        if len(buf) >= config.WRITE_BUFFER_SIZE or force:
            store_key = f'patch_store_{label}'
            path_key = f'store_{label}_path'
            fallback_path = config.PATCH_ZARR_OUTPUT_NORM_FALLBACK if label == 'norm' else config.PATCH_ZARR_OUTPUT_RAW_FALLBACK
            
            try:
                store = self.zarr_state[store_key]
                new_size = store.shape[0] + len(buf)
                store.resize((new_size, config.PATCH_SIZE_TRAIN, config.PATCH_SIZE_TRAIN, 3))
                store[store.shape[0] - len(buf): store.shape[0]] = np.stack(buf, axis=0)
                buf.clear()
            except Exception as e:
                logger.warning(f"Write failed: {e}. Switching to fallback store.")
                fb = zarr.open(str(fallback_path), mode="a", shape=(0, config.PATCH_SIZE_TRAIN, config.PATCH_SIZE_TRAIN, 3), 
                               chunks=config.PATCH_CHUNK, dtype="uint8", compressor=config.ZARR_COMPRESSOR, zarr_version=2)
                new_size = fb.shape[0] + len(buf)
                fb.resize((new_size, config.PATCH_SIZE_TRAIN, config.PATCH_SIZE_TRAIN, 3))
                fb[fb.shape[0] - len(buf): fb.shape[0]] = np.stack(buf, axis=0)
                buf.clear()
                self.zarr_state[store_key] = fb
                self.zarr_state[path_key] = fallback_path

    def process_slide(self, path, xml_path, case_name, label, do_normalize):
        subset_name = "normalized" if do_normalize else "raw"
        target_label = "norm" if do_normalize else "raw"
        
        try:
            mask = None
            if config.FILTER_BY_ROI and pd.notna(xml_path):
                try:
                    mask = get_mask(xml_path, path, downsample_factor=config.MASK_DOWNSAMPLE)
                except Exception as mask_err:
                    logger.error(f"Mask generation failed for {path}: {mask_err}")
                    mask = None
                
                if mask is None:
                    logger.warning(f"Skipping slide {path} due to missing mask")
                    return

            with WSIReader(path, prefer_cucim=config.USE_CUCIM) as reader:
                width, height = reader.dimensions
                xs = list(range(0, max(0, width - config.PATCH_SIZE_TRAIN + 1), config.PATCH_STRIDE))
                ys = list(range(0, max(0, height - config.PATCH_SIZE_TRAIN + 1), config.PATCH_STRIDE))
                patches_this_slide = 0
                
                for y in ys:
                    if config.PATCH_LIMIT_PER_SLIDE and patches_this_slide >= config.PATCH_LIMIT_PER_SLIDE:
                        break
                    for x in xs:
                        if config.PATCH_LIMIT_PER_SLIDE and patches_this_slide >= config.PATCH_LIMIT_PER_SLIDE:
                            break
                        
                        if config.FILTER_BY_ROI and mask is not None:
                            center_x_ds = int((x + config.PATCH_SIZE_TRAIN / 2) / config.MASK_DOWNSAMPLE)
                            center_y_ds = int((y + config.PATCH_SIZE_TRAIN / 2) / config.MASK_DOWNSAMPLE)
                            if (center_y_ds >= mask.shape[0] or center_x_ds >= mask.shape[1] or mask[center_y_ds, center_x_ds] == 0):
                                continue
                        
                        try:
                            patch = reader.read_region(x=x, y=y, level=config.LEVEL_ANALYSIS, size=config.PATCH_SIZE_TRAIN)
                            out_patch = self._normalize_patch(patch) if do_normalize else patch
                            
                            self.buffers[target_label].append(out_patch)
                            
                            meta = self.metadata[target_label]
                            meta["slide_name"].append(Path(path).name)
                            meta["case_name"].append(case_name)
                            meta["label"].append(label)
                            meta["patch_global_index"].append(self.counters[target_label])
                            meta["patch_in_slide"].append(patches_this_slide)
                            meta["x"].append(x)
                            meta["y"].append(y)
                            
                            self.counters[target_label] += 1
                            patches_this_slide += 1
                            
                            self._flush_buffer(target_label)
                            
                        except Exception as e:
                            continue
            
            del mask
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to process slide {path}: {e}")

    def finalize(self):
        self._flush_buffer('norm', force=True)
        self._flush_buffer('raw', force=True)
        
        patch_meta_norm_df = pd.DataFrame(self.metadata['norm'])
        patch_meta_raw_df = pd.DataFrame(self.metadata['raw'])
        
        patch_meta_norm_path = config.OUTPUT_BASE / "patch_metadata_512_norm.csv"
        patch_meta_raw_path = config.OUTPUT_BASE / "patch_metadata_512_raw.csv"
        
        patch_meta_norm_df.to_csv(patch_meta_norm_path, index=False)
        patch_meta_raw_df.to_csv(patch_meta_raw_path, index=False)
        
        logger.info(f"Pipeline finished. Metadata saved to {patch_meta_norm_path} and {patch_meta_raw_path}")

def run_tiling(slides_norm, slides_raw, ref_vectors):
    pipeline = TilingPipeline(ref_vectors)
    
    for idx, row in tqdm(slides_norm.iterrows(), total=len(slides_norm), desc="Tiling Normalized"):
        pipeline.process_slide(row['full_path'], row['full_path_annotation'], row['case_name'], row['label'], do_normalize=True)
        
    for idx, row in tqdm(slides_raw.iterrows(), total=len(slides_raw), desc="Tiling Raw"):
        pipeline.process_slide(row['full_path'], row['full_path_annotation'], row['case_name'], row['label'], do_normalize=False)
        
    pipeline.finalize()
