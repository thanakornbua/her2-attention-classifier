#!/bin/bash
# Wrapper script to evaluate trained models on zarr test set
# Usage: ./evaluate_model.sh MODEL_PATH [OUTPUT_DIR] [ARCH]

set -e

# Default values
MODEL_PATH="${1:-/home/thanakornbuath/Desktop/best_resnet50.pth}"
OUTPUT_DIR="${2:-/media/thanakornbuath/data SSD/her2-attention-classifier/outputs/eval}"
ARCH="${3:-resnet50}"
LIMIT_SLIDES="${4:-}"
LIMIT_PATCHES="${5:-}"
TEST_MANIFEST="/media/thanakornbuath/data SSD/her2-attention-classifier/outputs/zarr_test_manifest.csv"

# Check if test manifest exists
if [ ! -f "$TEST_MANIFEST" ]; then
    echo "Error: Test manifest not found: $TEST_MANIFEST"
    echo ""
    echo "Please run split creation first:"
    echo "  ./run_create_splits.sh"
    exit 1
fi

# Validate model path
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found: $MODEL_PATH"
    exit 1
fi

# Activate conda environment
echo "Activating her2-class environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate her2-class

# Run evaluation
echo ""
echo "=============================================="
echo "Model Evaluation (Zarr Test Set)"
echo "=============================================="
echo "Model:         $MODEL_PATH"
echo "Architecture:  $ARCH"
echo "Test manifest: $TEST_MANIFEST"
echo "Output:        $OUTPUT_DIR"
echo ""

EXTRA_ARGS=""
if [ -n "$LIMIT_SLIDES" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --limit-slides $LIMIT_SLIDES"
fi
if [ -n "$LIMIT_PATCHES" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --limit-patches $LIMIT_PATCHES"
fi

python evaluate_zarr_model.py \
        --model-path "$MODEL_PATH" \
        --test-manifest "$TEST_MANIFEST" \
        --arch "$ARCH" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 32 \
        --num-workers 8 \
        --amp $EXTRA_ARGS

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✓ Evaluation Complete"
    echo "=============================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files:"
    echo "  - metrics.json          (accuracy, precision, recall, F1, AUROC)"
    echo "  - confusion_matrix.csv  (predicted vs actual)"
    echo "  - tp_fn_fp_tn.csv       (per-class counts)"
    echo ""
    
    # Show metrics summary
    if [ -f "$OUTPUT_DIR/metrics.json" ]; then
        echo "Quick Summary:"
        python -c "import json; m=json.load(open('$OUTPUT_DIR/metrics.json')); print(f\"  Accuracy: {m['accuracy']:.4f}\"); print(f\"  AUROC:    {m['auroc_macro']:.4f}\")"
    fi
else
    echo ""
    echo "✗ Evaluation failed!"
    exit 1
fi
