#!/bin/bash
# Example script for running HER2 slide preprocessing pipeline
# Edit the paths below to match your setup

# Configuration
DATA_ROOT="/media/thanakornbuath/data SSD/her2-attention-classifier/data"
OUTPUTS_ROOT="/media/thanakornbuath/data SSD/her2-attention-classifier/outputs"
ZARR_OUTPUT_DIR="/media/thanakornbuath/patch/zarr_norm"
PATCHES_ROOT="/media/thanakornbuath/data SSD/her2-attention-classifier/outputs/patches"

# Parameters
PATCH_SIZE=512
STRIDE=512
NUM_WORKERS=8
BATCH_SIZE=128
TISSUE_THRESHOLD=0.2
NUM_REF_SUBFOLDERS=100
IMAGES_PER_FOLDER=200

# Cohorts to process
COHORTS="TCGA_BRCA_Filtered Yale_HER2_cohort Yale_trastuzumab_response_cohort"

# GPU flag (comment out if no GPU)
GPU_FLAG="--use-gpu"

echo "=============================================="
echo "HER2 Slide Preprocessing Pipeline"
echo "=============================================="
echo ""

# Step 1: Compute reference stain statistics (if not already done)
REF_STATS="$OUTPUTS_ROOT/ref_stain_stats.npz"
if [ ! -f "$REF_STATS" ]; then
    echo "Step 1: Computing reference stain statistics..."
    python preprocess_slides_cli.py \
        --data-root "$DATA_ROOT" \
        --outputs-root "$OUTPUTS_ROOT" \
        --zarr-output-dir "$ZARR_OUTPUT_DIR" \
        --patches-root "$PATCHES_ROOT" \
        --compute-ref-stats \
        --num-ref-subfolders $NUM_REF_SUBFOLDERS \
        --images-per-folder $IMAGES_PER_FOLDER \
        $GPU_FLAG

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
    --process-slides \
    --patch-size $PATCH_SIZE \
    --stride $STRIDE \
    --num-workers $NUM_WORKERS \
    --batch-size $BATCH_SIZE \
    --tissue-threshold $TISSUE_THRESHOLD \
    --skip-existing \
    --cohorts $COHORTS \
    $GPU_FLAG

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
echo "  - Zarr files: $ZARR_OUTPUT_DIR/*.zarr"
echo "  - Train manifest: $OUTPUTS_ROOT/zarr_train_manifest.csv"
echo "  - Val manifest: $OUTPUTS_ROOT/zarr_val_manifest.csv"
echo "  - Logs: $OUTPUTS_ROOT/logs/preprocess_cli.log"
echo ""

