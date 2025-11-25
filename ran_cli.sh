#!/bin/bash
# Example script for running HER2 slide preprocessing pipeline
# Edit the paths below to match your setup

# Configuration
DATA_ROOT="/home/thanakornbuath/hdd/her2-attention-classifier/data"
OUTPUTS_ROOT="/media/thanakornbuath/data SSD/her2-attention-classifier/outputs"
ZARR_OUTPUT_DIR="/media/thanakornbuath/data SSD/her2-attention-classifier/zarr_raw"
ZARR_NORM_OUTPUT_DIR="/media/thanakornbuath/data SSD/her2-attention-classifier/zarr_norm"
PATCHES_ROOT="/home/thanakornbuath/hdd/her2-attention-classifier/outputs/patches"

# Parameters
PATCH_SIZE=512
STRIDE=512
NUM_WORKERS=8
BATCH_SIZE=128
TISSUE_THRESHOLD=0.2
NUM_REF_SUBFOLDERS=100
IMAGES_PER_FOLDER=200
REF_MODE="slides"            # or "patches"
REF_SLIDES=200                 # used when REF_MODE=slides
REF_PATCHES_PER_SLIDE=100      # used when REF_MODE=slides
REF_DATA_ROOT="$DATA_ROOT"   # override if reference slides live elsewhere
SLIDE_FRACTION=0.8            # fraction of slides to process (0-1]
SLIDE_FRACTION_SEED=42
SLIDE_WORKERS=1
SLIDE_WORKER_GPUS=""        # e.g. "0,1" when SLIDE_WORKERS>1
DISABLE_MACENKO=0             # set to 1 to skip stain normalization entirely

# GPU flag (comment out if no GPU)
GPU_FLAG="--use-gpu"

GPU_ARGS=()
if [ -n "$GPU_FLAG" ]; then
    GPU_ARGS+=("$GPU_FLAG")
fi
MACENKO_ARG=()
if [ "$DISABLE_MACENKO" -eq 1 ]; then
    MACENKO_ARG+=(--disable-macenko)
fi
SLIDE_GPU_ARG=()
if [ -n "$SLIDE_WORKER_GPUS" ]; then
    SLIDE_GPU_ARG+=(--slide-worker-gpus "$SLIDE_WORKER_GPUS")
fi
SLIDE_WORKER_ARG=(--slide-workers $SLIDE_WORKERS)
SLIDE_FRACTION_ARGS=(--slide-fraction $SLIDE_FRACTION --slide-fraction-seed $SLIDE_FRACTION_SEED)

# Cohorts to process
COHORTS="TCGA_BRCA_Filtered Yale_HER2_cohort Yale_trastuzumab_response_cohort"

echo "=============================================="
echo "HER2 Slide Preprocessing Pipeline"
echo "=============================================="
echo ""

# Step 1: Compute reference stain statistics (if not already done)
REF_STATS="$OUTPUTS_ROOT/ref_stain_stats.npz"
if [ ! -f "$REF_STATS" ]; then
    echo "Step 1: Computing reference stain statistics..."
    if [ "$REF_MODE" = "patches" ]; then
        python preprocess_slides_cli.py \
            --data-root "$DATA_ROOT" \
            --outputs-root "$OUTPUTS_ROOT" \
            --zarr-output-dir "$ZARR_OUTPUT_DIR" \
            --patches-root "$PATCHES_ROOT" \
            --compute-ref-stats \
            --ref-mode patches \
            --num-ref-subfolders $NUM_REF_SUBFOLDERS \
            --images-per-folder $IMAGES_PER_FOLDER \
            ${GPU_ARGS[@]}
    else
        python preprocess_slides_cli.py \
            --data-root "$REF_DATA_ROOT" \
            --outputs-root "$OUTPUTS_ROOT" \
            --zarr-output-dir "$ZARR_OUTPUT_DIR" \
            --compute-ref-stats \
            --ref-mode slides \
            --ref-slides $REF_SLIDES \
            --ref-patches-per-slide $REF_PATCHES_PER_SLIDE \
            --patch-size $PATCH_SIZE \
            --downsample-mask $((STRIDE / (STRIDE / PATCH_SIZE))) \
            --cohorts $COHORTS \
            ${GPU_ARGS[@]}
    fi

    if [ $? -ne 0 ]; then
        echo "ERROR: Reference stats computation failed!"
        exit 1
    fi
    echo "✓ Reference stats computed successfully"
else
    echo "Step 1: Reference stats already exist, skipping..."
fi

echo ""

# Step 2: Process slides to Zarr
echo "Step 2: Processing slides to Zarr format..."
python preprocess_slides_cli.py \
    --data-root "$DATA_ROOT" \
    --outputs-root "$OUTPUTS_ROOT" \
    --zarr-output-dir "$ZARR_OUTPUT_DIR" \
    --mixed-raw-output-dir "$ZARR_OUTPUT_DIR" \
    --mixed-norm-output-dir "$ZARR_NORM_OUTPUT_DIR" \
    --mixed-norm-fraction 0.2 \
    --mixed-split-seed 123 \
    --process-slides \
    --patch-size $PATCH_SIZE \
    --stride $STRIDE \
    --num-workers $NUM_WORKERS \
    --batch-size $BATCH_SIZE \
    --tissue-threshold $TISSUE_THRESHOLD \
    --skip-existing \
    --cohorts $COHORTS \
    ${MACENKO_ARG[@]} \
    ${GPU_ARGS[@]} \
    ${SLIDE_WORKER_ARG[@]} \
    ${SLIDE_GPU_ARG[@]} \
    ${SLIDE_FRACTION_ARGS[@]}

if [ $? -ne 0 ]; then
    echo "ERROR: Slide processing failed!"
    exit 1
fi
echo "✓ Slides processed successfully"

echo ""

# Step 3: Create train/val split
echo "Step 3: Creating train/val split..."
python preprocess_slides_cli.py \
    --data-root "$DATA_ROOT" \
    --outputs-root "$OUTPUTS_ROOT" \
    --zarr-output-dir "$ZARR_OUTPUT_DIR" \
    --create-split

if [ $? -ne 0 ]; then
    echo "ERROR: Train/val split creation failed!"
    exit 1
fi
echo "✓ Train/val split created successfully"

echo ""
echo "=============================================="
echo "✅ Preprocessing pipeline completed!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  - Raw Zarr files: $ZARR_OUTPUT_DIR/*.zarr"
echo "  - Normalized Zarr files: $ZARR_NORM_OUTPUT_DIR/*.zarr"
echo "  - Train manifest: $OUTPUTS_ROOT/zarr_train_manifest.csv"
echo "  - Val manifest: $OUTPUTS_ROOT/zarr_val_manifest.csv"
echo "  - Logs: $OUTPUTS_ROOT/logs/preprocess_cli.log"
echo ""
