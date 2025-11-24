# HER2 Slide Preprocessing CLI Usage Guide

## Overview

The `preprocess_slides_cli.py` script is an optimized CLI tool for extracting patches from whole slide images (WSI), applying Macenko stain normalization, and saving to Zarr format. It includes aggressive memory management to prevent memory leaks.

## Features

- ✅ **Memory leak prevention**: Aggressive garbage collection and GPU memory cleanup after each slide
- ✅ **GPU acceleration**: Optional CuPy-based acceleration for Macenko normalization
- ✅ **Parallel processing**: Multi-threaded patch extraction per slide
- ✅ **Progress tracking**: TQDM progress bars
- ✅ **Skip existing**: Avoid reprocessing already completed slides
- ✅ **Zarr format**: Efficient storage with compression

## Installation

### Required packages:
```bash
pip install numpy pandas scipy pillow zarr tqdm scikit-learn openslide-python
```

### Optional (for GPU acceleration):
```bash
pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version
```

## Usage

### 1. Compute Reference Stain Statistics

Before processing slides, compute reference stain characteristics from sample patches:

```bash
python preprocess_slides_cli.py \
    --data-root "/media/thanakornbuath/data SSD/her2-attention-classifier/data" \
    --outputs-root "/media/thanakornbuath/data SSD/her2-attention-classifier/outputs" \
    --zarr-output-dir "/media/thanakornbuath/patch/zarr_norm" \
    --patches-root "/media/thanakornbuath/data SSD/her2-attention-classifier/outputs/patches" \
    --compute-ref-stats \
    --num-ref-subfolders 100 \
    --images-per-folder 200 \
    --use-gpu
```

This will:
- Sample images from 100 random patch subfolders
- Load up to 200 images per folder
- Compute mean Macenko stain vectors and max concentrations
- Save to `outputs/ref_stain_stats.npz`

### 2. Process Slides to Zarr

Extract patches from WSI slides with stain normalization:

```bash
python preprocess_slides_cli.py \
    --data-root "/media/thanakornbuath/data SSD/her2-attention-classifier/data" \
    --outputs-root "/media/thanakornbuath/data SSD/her2-attention-classifier/outputs" \
    --zarr-output-dir "/media/thanakornbuath/patch/zarr_norm" \
    --process-slides \
    --patch-size 512 \
    --stride 512 \
    --num-workers 8 \
    --batch-size 128 \
    --tissue-threshold 0.2 \
    --use-gpu \
    --skip-existing \
    --cohorts TCGA_BRCA_Filtered Yale_HER2_cohort Yale_trastuzumab_response_cohort
```

This will:
- Load reference stain stats from `outputs/ref_stain_stats.npz`
- Process all slides in specified cohorts
- Extract 512×512 patches with stride 512 (no overlap)
- Filter patches with <20% tissue content
- Apply Macenko stain normalization
- Save to Zarr format with compression
- Skip existing Zarr files

### 3. Create Train/Val Split

Generate train/val split CSV manifests:

```bash
python preprocess_slides_cli.py \
    --data-root "/media/thanakornbuath/data SSD/her2-attention-classifier/data" \
    --outputs-root "/media/thanakornbuath/data SSD/her2-attention-classifier/outputs" \
    --zarr-output-dir "/media/thanakornbuath/patch/zarr_norm" \
    --create-split
```

This will:
- Read all Zarr files in output directory
- Create 80/20 train/val split (stratified by label)
- Save manifests to `outputs/zarr_train_manifest.csv` and `outputs/zarr_val_manifest.csv`

### 4. All-in-One Processing

Run all steps in sequence:

```bash
python preprocess_slides_cli.py \
    --data-root "/media/thanakornbuath/data SSD/her2-attention-classifier/data" \
    --outputs-root "/media/thanakornbuath/data SSD/her2-attention-classifier/outputs" \
    --zarr-output-dir "/media/thanakornbuath/patch/zarr_norm" \
    --patches-root "/media/thanakornbuath/data SSD/her2-attention-classifier/outputs/patches" \
    --compute-ref-stats \
    --process-slides \
    --create-split \
    --patch-size 512 \
    --stride 512 \
    --num-workers 8 \
    --batch-size 128 \
    --use-gpu \
    --skip-existing \
    --cohorts TCGA_BRCA_Filtered Yale_HER2_cohort Yale_trastuzumab_response_cohort
```

## Command-Line Arguments

### Paths
- `--data-root`: Root directory containing cohort data (required)
- `--outputs-root`: Root directory for outputs (required)
- `--zarr-output-dir`: Output directory for Zarr files (required)
- `--patches-root`: Directory with existing patches for reference sampling (optional)
- `--ref-stats-path`: Path to reference stain stats npz file (default: `outputs/ref_stain_stats.npz`)

### Reference Statistics
- `--num-ref-subfolders`: Number of random subfolders to sample (default: 100)
- `--images-per-folder`: Max images per sampled subfolder (default: 200)

### Patch Extraction
- `--patch-size`: Patch size in pixels (default: 512)
- `--stride`: Patch stride (default: 512, no overlap if equals patch size)
- `--level`: OpenSlide level, 0 = highest resolution (default: 0)
- `--tissue-threshold`: Minimum tissue fraction in patch 0-1 (default: 0.2)
- `--downsample-mask`: Downsample factor for annotation mask (default: 16)

### Performance
- `--num-workers`: Number of parallel workers for patch extraction (default: 8)
- `--batch-size`: Patches per batch (default: 128)
- `--use-gpu`: Use GPU acceleration for Macenko normalization (flag)
- `--skip-existing`: Skip existing Zarr files (flag)

### Dataset
- `--cohorts`: List of cohorts to process (default: TCGA_BRCA_Filtered Yale_HER2_cohort Yale_trastuzumab_response_cohort)

### Actions
- `--compute-ref-stats`: Compute reference stain statistics (flag)
- `--process-slides`: Process slides to Zarr (flag)
- `--create-split`: Create train/val split manifest (flag)

## Expected Directory Structure

```
data/
├── TCGA_BRCA_Filtered/
│   ├── SVS/
│   │   ├── TCGA-XX-YYYY.01234.svs
│   │   └── ...
│   ├── Annotations/
│   │   ├── TCGA-XX-YYYY.xml
│   │   └── ...
│   └── HER2_TCGA_clean.csv
├── Yale_HER2_cohort/
│   ├── SVS/
│   ├── Annotations/
│   └── labels.csv (optional)
└── Yale_trastuzumab_response_cohort/
    ├── SVS/
    ├── Annotations/
    └── labels.csv (optional)

outputs/
├── patches/          # For reference sampling
├── ref_stain_stats.npz
├── zarr_train_manifest.csv
├── zarr_val_manifest.csv
└── logs/
    └── preprocess_cli.log

zarr_output_dir/
├── slide_001.zarr/
│   ├── patches/      # N × 512 × 512 × 3 array
│   ├── coords/       # N × 2 (x,y) coordinates
│   ├── labels/       # N labels
│   └── meta.json
└── slide_002.zarr/
    └── ...
```

## Zarr File Structure

Each slide produces a `.zarr` directory containing:

- **patches**: Dataset of shape `(N, 512, 512, 3)` with normalized RGB patches
- **coords**: Dataset of shape `(N, 2)` with (x, y) coordinates in SVS level 0
- **labels**: Dataset of shape `(N,)` with slide-level labels (0=HER2-, 1=HER2+)
- **meta.json**: Slide metadata including MPP, magnification, patch count

## Memory Optimization Features

The CLI script includes several memory leak prevention strategies:

1. **Explicit deletion**: Large arrays are deleted immediately after use with `del`
2. **Garbage collection**: `gc.collect()` called periodically (every 3 slides, every batch)
3. **GPU memory cleanup**: CuPy memory pools freed after each batch and slide
4. **Batch processing**: Patches processed in batches rather than all at once
5. **Context managers**: `with` statements ensure proper resource cleanup
6. **Array resizing**: Zarr arrays resized to actual size to save memory
7. **Reference cleanup**: Reference images freed immediately after computing stats

## Performance Tips

1. **GPU acceleration**: Use `--use-gpu` if you have a CUDA-compatible GPU with CuPy installed (2-3x faster)
2. **Worker threads**: Adjust `--num-workers` based on your CPU cores (typically 4-16)
3. **Batch size**: Increase `--batch-size` if you have more RAM available (64-256)
4. **Skip existing**: Always use `--skip-existing` to avoid reprocessing
5. **Tissue threshold**: Increase `--tissue-threshold` to filter more background patches
6. **Monitor logs**: Check `outputs/logs/preprocess_cli.log` for errors and warnings

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size` (e.g., 64 or 32)
- Reduce `--images-per-folder` (e.g., 100)
- Reduce `--num-ref-subfolders` (e.g., 50)
- Increase `--downsample-mask` (e.g., 32)
- Don't use `--use-gpu` if GPU memory is limited

### Slow Processing
- Increase `--num-workers` (e.g., 12 or 16)
- Use `--use-gpu` if available
- Increase `--tissue-threshold` to skip more patches
- Use SSD for input/output if possible

### Missing Slides
- Check that SVS and XML files are in correct directories
- For TCGA slides, XML filename should match part before first dot
- Check `outputs/logs/preprocess_cli.log` for skip messages

### Normalization Issues
- Ensure `ref_stain_stats.npz` exists before processing
- Verify reference stats with sufficient images (aim for 5000+ total)
- Check Max H and Max E values are reasonable (typically 0.5-2.0)

## Example Full Workflow

```bash
# Step 1: Compute reference statistics
python preprocess_slides_cli.py \
    --data-root "/path/to/data" \
    --outputs-root "/path/to/outputs" \
    --zarr-output-dir "/path/to/zarr" \
    --patches-root "/path/to/outputs/patches" \
    --compute-ref-stats \
    --num-ref-subfolders 100 \
    --images-per-folder 200 \
    --use-gpu

# Step 2: Process all slides
python preprocess_slides_cli.py \
    --data-root "/path/to/data" \
    --outputs-root "/path/to/outputs" \
    --zarr-output-dir "/path/to/zarr" \
    --process-slides \
    --patch-size 512 \
    --stride 512 \
    --num-workers 8 \
    --batch-size 128 \
    --use-gpu \
    --skip-existing

# Step 3: Create train/val split
python preprocess_slides_cli.py \
    --data-root "/path/to/data" \
    --outputs-root "/path/to/outputs" \
    --zarr-output-dir "/path/to/zarr" \
    --create-split

# Done! Use zarr_train_manifest.csv and zarr_val_manifest.csv for training
```

## Notes

- Processing time depends on slide size and hardware (typically 5-30 min per slide)
- Zarr files are compressed with Blosc (zstd) for efficient storage
- Each slide's patches are stored in a single Zarr group for efficient loading
- Use `--skip-existing` to resume interrupted processing
- GPU acceleration only helps for Macenko normalization, not I/O operations

