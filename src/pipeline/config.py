"""
Configuration module for the HER2 Attention Classifier pipeline.
"""

from pathlib import Path
import os

# ============================================================================
# PATCH EXTRACTION CONFIGURATION
# ============================================================================

# Reference calculation parameters
PATCHES_PER_SLIDE = 100       # Random patches per slide for reference vectors
PATCH_SIZE = 256              # Patch size for reference calculation
LEVEL_REFERENCE = 2           # WSI pyramid level for reference (lower res, faster)
SEED = 42                     # Random seed for reproducibility

# Training patch parameters
PATCH_SIZE_TRAIN = 512        # High-res patch size for training
PATCH_STRIDE = 512            # Non-overlapping patches (stride = patch size)
PATCH_LIMIT_PER_SLIDE = None  # Optional: cap patches per slide (None = unlimited)
LEVEL_ANALYSIS = 0            # WSI pyramid level for training patches (full res)

# Backend and processing
USE_CUCIM = False             # Use CuCIM for faster WSI reading (False = OpenSlide)
PATCH_USE_GPU = True          # Use GPU for Macenko normalization if available

# ROI filtering
FILTER_BY_ROI = True          # Only extract patches within annotated ROIs
MASK_DOWNSAMPLE = 4           # Downsample factor for mask generation (memory/speed)

# ============================================================================
# DATASET SPLIT CONFIGURATION
# ============================================================================

# Patch normalization split
MAX_SLIDES_TO_PROCESS = None    # Number of slides to randomly sample
SAMPLE_SEED = 42              # Seed for slide sampling
SPLIT_NORMALIZED_RATIO = 0.20 # 20% normalized patches, 80% raw patches

# Train/Val/Test split for model training
TRAIN_RATIO = 0.70            # 70% training samples
VAL_RATIO = 0.10              # 10% validation samples
TEST_RATIO = 0.20             # 20% test samples
SPLIT_SEED = 42               # Seed for reproducible splits

# ============================================================================
# ZARR STORAGE CONFIGURATION
# ============================================================================

# Primary storage location
# Use environment variable or default to a local path
_DEFAULT_ZARR_PATH = str(Path("data/zarr_output").resolve())

PATCH_ZARR_OUTPUT = Path(os.getenv("PATCH_ZARR_OUTPUT", _DEFAULT_ZARR_PATH))

# Zarr optimization settings
try:
    from numcodecs import Blosc
    ZARR_COMPRESSOR = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
except ImportError:
    ZARR_COMPRESSOR = None  # Fallback if numcodecs is not installed

PATCH_CHUNK = (128, PATCH_SIZE_TRAIN, PATCH_SIZE_TRAIN, 3)  # Chunk size for I/O
WRITE_BUFFER_SIZE = 64        # Number of patches to buffer before writing

# Computed paths (normalized and raw outputs)
PATCH_ZARR_OUTPUT_NORM = PATCH_ZARR_OUTPUT / "patches_norm.zarr"
PATCH_ZARR_OUTPUT_RAW = PATCH_ZARR_OUTPUT / "patches_raw.zarr"

def update_zarr_path(new_path: str):
    """Updates the Zarr output path and derived paths."""
    global PATCH_ZARR_OUTPUT, PATCH_ZARR_OUTPUT_NORM, PATCH_ZARR_OUTPUT_RAW
    PATCH_ZARR_OUTPUT = Path(new_path)
    PATCH_ZARR_OUTPUT_NORM = PATCH_ZARR_OUTPUT / "patches_norm.zarr"
    PATCH_ZARR_OUTPUT_RAW = PATCH_ZARR_OUTPUT / "patches_raw.zarr"


# Fallback storage (if primary drive is full)
_DEFAULT_FALLBACK_PATH = str(Path("data/zarr_fallback").resolve())

PATCH_ZARR_FALLBACK = Path(os.getenv("PATCH_ZARR_FALLBACK", _DEFAULT_FALLBACK_PATH))
PATCH_ZARR_OUTPUT_NORM_FALLBACK = PATCH_ZARR_FALLBACK / "patches_norm.zarr"
PATCH_ZARR_OUTPUT_RAW_FALLBACK = PATCH_ZARR_FALLBACK / "patches_raw.zarr"

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUTPUT_BASE = Path('outputs')
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
