
#!/bin/bash
# Enhanced training script for HER2 Phase 1 classifiers
# Trains ResNet-50 and EfficientNet-B0 with comprehensive overfitting prevention:
#   - Patch rotation augmentation
#   - Elastic deformation
#   - Dropout (p=0.3-0.5)
#   - Early stopping based on AUC (patience=5 epochs)
#   - Test set evaluation

set -e  # Exit on error

# Configuration
OUTPUTS_ROOT="/media/thanakornbuath/data SSD/her2-attention-classifier/outputs"
ZARR_NORM_DIR="/media/thanakornbuath/data SSD/her2-attention-classifier/zarr_norm"

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=1e-5
TRAIN_STEPS_PER_EPOCH=500
VAL_STEPS=100
MAX_PATCHES_PER_SLIDE=2048
NUM_WORKERS=8

# Overfitting prevention
ROTATION_DEGREES=15
ELASTIC_ALPHA=2.5
ELASTIC_SIGMA=5.5
ELASTIC_PROB=0.7
ELASTIC_KERNEL=15
DROPOUT_P=0.4
EARLY_STOP_PATIENCE=5

# Check if manifests exist
TRAIN_MANIFEST="$OUTPUTS_ROOT/zarr_train_manifest.csv"
VAL_MANIFEST="$OUTPUTS_ROOT/zarr_val_manifest.csv"

if [ ! -f "$TRAIN_MANIFEST" ] || [ ! -f "$VAL_MANIFEST" ]; then
    echo "ERROR: Train/val manifests not found!"
    echo "  Expected: $TRAIN_MANIFEST"
    echo "            $VAL_MANIFEST"
    echo ""
    echo "Run preprocessing first: ./run_preprocessing.sh"
    exit 1
fi

# Create test manifest if it doesn't exist (10% holdout from val set)
TEST_MANIFEST="$OUTPUTS_ROOT/zarr_test_manifest.csv"
if [ ! -f "$TEST_MANIFEST" ]; then
    echo "Creating test manifest (10% holdout from validation)..."
    python3 << EOF
import pandas as pd
from pathlib import Path

val_df = pd.read_csv("$VAL_MANIFEST")
n_test = max(1, int(len(val_df) * 0.1))
test_df = val_df.sample(n=n_test, random_state=42)
val_df_new = val_df.drop(test_df.index)

test_df.to_csv("$TEST_MANIFEST", index=False)
val_df_new.to_csv("$VAL_MANIFEST", index=False)

print(f"✓ Created test manifest with {len(test_df)} slides")
print(f"✓ Updated val manifest with {len(val_df_new)} slides")
EOF
fi

echo "=============================================="
echo "HER2 Phase 1 Training Pipeline (Enhanced)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Epochs: $EPOCHS (with early stopping)"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Dropout: $DROPOUT_P"
echo "  Early stop patience: $EARLY_STOP_PATIENCE epochs"
echo "  Rotation: ±${ROTATION_DEGREES}°"
echo "  Elastic deformation: enabled"
echo "  Data source: zarr_norm (Macenko-normalized)"
echo ""

# Function to train a model
train_model() {
    local MODEL_NAME=$1
    local OUTPUT_DIR="$OUTPUTS_ROOT/phase1_training/${MODEL_NAME}_$(date +%Y%m%d-%H%M%S)"
    
    echo "Training $MODEL_NAME..."
    echo "Output: $OUTPUT_DIR"
    
    mkdir -p "$OUTPUT_DIR"
    
    python src/train/train_phase1_zarr.py \
        --train-manifest "$TRAIN_MANIFEST" \
        --val-manifest "$VAL_MANIFEST" \
        --test-manifest "$TEST_MANIFEST" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL_NAME" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --weight-decay $WEIGHT_DECAY \
        --train-steps-per-epoch $TRAIN_STEPS_PER_EPOCH \
        --val-steps $VAL_STEPS \
        --max-patches-per-slide $MAX_PATCHES_PER_SLIDE \
        --num-workers $NUM_WORKERS \
        --rotation-degrees $ROTATION_DEGREES \
        --elastic-alpha $ELASTIC_ALPHA \
        --elastic-sigma $ELASTIC_SIGMA \
        --elastic-prob $ELASTIC_PROB \
        --elastic-kernel-size $ELASTIC_KERNEL \
        --dropout-p $DROPOUT_P \
        --early-stop-patience $EARLY_STOP_PATIENCE \
        --enable-tensorboard \
        --enable-wandb
    
    if [ $? -eq 0 ]; then
        echo "✓ $MODEL_NAME training completed successfully"
        echo "  Model saved to: $OUTPUT_DIR/best_${MODEL_NAME}.pth"
        return 0
    else
        echo "✗ $MODEL_NAME training failed!"
        return 1
    fi
}

# Train ResNet-50
echo "================================================"
echo "Step 1/2: Training ResNet-50"
echo "================================================"
echo ""
train_model "resnet50"
RESNET_STATUS=$?
echo ""

# Train EfficientNet-B0
echo "================================================"
echo "Step 2/2: Training EfficientNet-B0"
echo "================================================"
echo ""
train_model "efficientnet_b0"
EFFICIENTNET_STATUS=$?
echo ""

# Summary
echo "=============================================="
echo "✅ Training Pipeline Complete"
echo "=============================================="
echo ""
echo "Results:"
if [ $RESNET_STATUS -eq 0 ]; then
    echo "  ✓ ResNet-50: SUCCESS"
else
    echo "  ✗ ResNet-50: FAILED"
fi

if [ $EFFICIENTNET_STATUS -eq 0 ]; then
    echo "  ✓ EfficientNet-B0: SUCCESS"
else
    echo "  ✗ EfficientNet-B0: FAILED"
fi
echo ""

echo "All model checkpoints saved to:"
echo "  $OUTPUTS_ROOT/phase1_training/"
echo ""
echo "View training metrics:"
echo "  tensorboard --logdir=$OUTPUTS_ROOT/phase1_training"
echo ""

# Exit with failure if any model failed
if [ $RESNET_STATUS -ne 0 ] || [ $EFFICIENTNET_STATUS -ne 0 ]; then
    exit 1
fi