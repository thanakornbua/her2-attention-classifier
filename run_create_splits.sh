#!/bin/bash
# Example usage of create_zarr_splits.py

# Configuration
ZARR_NORM="/media/thanakornbuath/data SSD/her2-attention-classifier/zarr_norm"
ZARR_RAW="/media/thanakornbuath/data SSD/her2-attention-classifier/zarr_raw"
OUTPUT_DIR="/media/thanakornbuath/data SSD/her2-attention-classifier/outputs"
TCGA_CSV="/media/thanakornbuath/data SSD/her2-attention-classifier/data/TCGA_BRCA_Filtered/HER2_TCGA_clean.csv"

# Split ratios (must sum to 1.0)
TRAIN_RATIO=0.7   # 70% for training
VAL_RATIO=0.15    # 15% for validation
TEST_RATIO=0.15   # 15% for test

# Random seed for reproducibility
SEED=42

echo "=============================================="
echo "Creating Stratified Train/Val/Test Splits"
echo "=============================================="
echo ""
echo "This will scan zarr_norm and zarr_raw directories"
echo "and create stratified splits:"
echo "  - zarr_train_manifest.csv (70% of slides)"
echo "  - zarr_val_manifest.csv   (15% of slides)"
echo "  - zarr_test_manifest.csv  (15% of slides)"
echo ""
echo "Label determination:"
echo "  - Her2Pos* → Positive (class 1)"
echo "  - Her2Neg* → Negative (class 0)"
echo "  - TCGA-*   → Lookup in $TCGA_CSV"
echo ""

# Activate conda environment
echo "Activating her2-class environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate her2-class

# Run split creation
python create_zarr_splits.py \
    --zarr-dirs "$ZARR_NORM" "$ZARR_RAW" \
    --tcga-csv "$TCGA_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --train $TRAIN_RATIO \
    --val $VAL_RATIO \
    --test $TEST_RATIO \
    --seed $SEED

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✓ Split Creation Complete"
    echo "=============================================="
    echo ""
    echo "Manifests saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Review the statistics above"
    echo "  2. Check manifests:"
    echo "     head $OUTPUT_DIR/zarr_train_manifest.csv"
    echo "  3. Start training:"
    echo "     ./train.sh"
    echo ""
else
    echo ""
    echo "✗ Error creating splits!"
    exit 1
fi
