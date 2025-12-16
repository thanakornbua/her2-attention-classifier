# Quick Start: Dual Dataset Training

## What Changed?

The training now uses **both normalized AND raw patches** automatically:
- **1.1M total patches** (was 230K with normalized only)
- **Better model robustness** from diverse stain variations
- **Same code structure** - just uncomment to run

## Dataset Summary

```
Total Patches: 1,144,818
├── Normalized:  230,120 (20.1%)  ← Stain-normalized
└── Raw:         914,698 (79.9%)  ← Original colors

Training:  801,372 patches (70%)
Validation: 114,482 patches (10%)
Test:       228,964 patches (20%)

HER2 Distribution (Combined):
├── Positive: 606,379 (53.0%)
└── Negative: 538,439 (47.0%)
```

## How to Train

### Step 1: Prepare Data (Already Done)

Section 4.2 created:
- `outputs/preprocessing/patch_metadata_512_norm.csv` (230K patches)
- `outputs/preprocessing/patch_metadata_512_raw.csv` (914K patches)
- `E:\zarr\patches_norm.zarr` (230K)
- `E:\zarr\patches_raw.zarr` (914K)

### Step 2: Load Combined Dataset

Run **Section 5.1.1** (Phase 1 Config):
```python
phase1_config = {
    'seed': 42,
    'batch_size': 16,
    'num_epochs': 50,
    'lr': 1e-4,
    'backbone': 'resnet50',
    'num_classes': 2,
    'dropout': 0.5,
    'amp_enabled': True,
    'early_stop_patience': 10,
}
```

### Step 3: Load Datasets

Run **Section 5.1.2** (Dataset Loading):
- Loads both normalized and raw metadata CSVs
- Opens both Zarr archives
- Combines into single dataset
- Creates train/val/test splits
- Shows: 1,144,818 total patches loaded ✓

### Step 4: Start Training

**Section 5.1.3** - Uncomment and run:
```python
from src.training.train_phase1 import run_training

run_training(
    zarr_path=zarr_path,
    train_indices=train_indices,
    val_indices=val_indices,
    output_dir='outputs/phase1',
    config=phase1_config,
    zarr_path_secondary=zarr_path_secondary,  # ← Dual Zarr
    patch_metadata=patch_metadata              # ← Routing info
)
```

## Monitor Training

### TensorBoard (Real-time)

Open new terminal:
```bash
cd c:\Users\tanth\Desktop\her2-attention-classifier
tensorboard --logdir=outputs/phase1/tensorboard_logs
```

Visit: http://localhost:6006

### Metrics Tracked

- Training loss
- Validation loss
- Validation AUC (main metric)
- Accuracy, Precision, Recall, F1

## Expected Performance

### Training Time
- **Per epoch**: ~2-3 minutes (ResNet-50, batch 16, 800K samples)
- **50 epochs**: ~2-3 hours total
- **GPU**: A100 recommended, RTX 3090 also works

### Memory Usage
- **Batch size 16**: ~14GB GPU memory
- **Peak RAM**: ~50GB during data loading

## Output Files

After training completes:
```
outputs/phase1/
├── best_model.pth          ← Best checkpoint (highest AUC)
├── last_model.pth          ← Final epoch
├── metrics.json            ← Training history
├── config_used.yaml        ← Configuration used
└── tensorboard_logs/       ← TensorBoard events
    ├── train/
    └── validation/
```

## Key Improvements Over Single Dataset

| Aspect | Single (Normalized) | Dual (Norm+Raw) |
|--------|-------------------|-----------------|
| Training patches | 161K | 801K |
| Data diversity | Limited | High |
| Stain variation | Normalized | Both variants |
| Robustness | Lower | Higher |
| Training time | ~30 min | ~2.5 hours |

## Troubleshooting

### Memory Issues
Reduce batch size:
```python
phase1_config['batch_size'] = 8  # Instead of 16
```

### Slow Loading
- Ensure both Zarr archives are on SSD/NVMe
- Check disk I/O with Task Manager

### Training Not Starting
- Verify zarr_path_secondary is not None
- Confirm patch_metadata has 'source' column
- Check 'patch_global_index' column exists

## What Happens During Training

For each batch:
1. Sample random indices from combined dataset
2. Look up 'source' (normalized or raw)
3. Load patch from correct Zarr archive
4. Pass through ResNet-50 backbone
5. Compute loss and backpropagate
6. Update weights

The model learns features robust to both stain normalization and original variations.

## Next Steps After Training

1. **Evaluate**: Check `metrics.json` and TensorBoard
2. **Phase 2**: Feature extraction using trained model
3. **MIL Training**: Slide-level aggregation
4. **Optional U-Net**: Tumor segmentation

See `DUAL_DATASET_TRAINING.md` for detailed documentation.
