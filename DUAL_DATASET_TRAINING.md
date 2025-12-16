# Dual Dataset Training with Normalized + Raw Patches

## Overview

The training pipeline has been updated to use **both normalized and raw patches** simultaneously during Phase 1 training. This provides the model with diverse stain variations and improves robustness.

## Dataset Composition

### Combined Dataset Statistics
- **Total patches**: 1,144,818
- **Normalized patches**: 230,120 (20.1%)
  - Positive: 110,108
  - Negative: 120,012
- **Raw patches**: 914,698 (79.9%)
  - Positive: 496,271
  - Negative: 418,427

### Train/Val/Test Split (70/10/20)
- **Training samples**: 801,372 patches
  - From normalized: ~46,024 patches
  - From raw: ~755,348 patches
- **Validation samples**: 114,482 patches
- **Test samples**: 228,964 patches

## Data Sources

### Normalized Patches (Stain-Normalized)
- **CSV**: `outputs/preprocessing/patch_metadata_512_norm.csv`
- **Zarr Archive**: `E:\zarr\patches_norm.zarr`
- **Format**: RGB uint8 (512×512×3)
- **Preprocessing**: Macenko stain normalization applied

### Raw Patches (Original)
- **CSV**: `outputs/preprocessing/patch_metadata_512_raw.csv`
- **Zarr Archive**: `E:\zarr\patches_raw.zarr`
- **Format**: RGB uint8 (512×512×3)
- **Preprocessing**: None (original slide colors)

## Implementation Details

### Section 5.1.2: Dataset Loading

The notebook cell automatically:
1. **Loads both CSVs**: normalized and raw metadata
2. **Opens both Zarr archives**: primary (normalized) and secondary (raw)
3. **Combines datasets**: Concatenates metadata with 'source' column tracking
4. **Creates indices**: Shuffles and creates train/val/test splits

```python
# Combined dataset
patch_metadata = pd.concat([patch_metadata_norm, patch_metadata_raw], ignore_index=True)
# Added 'source' column: either 'normalized' or 'raw'

# Split indices
train_indices, val_indices, test_indices = create_splits(patch_metadata)
```

### ZarrPatchDataset Class (Dual Zarr Support)

Updated `src/dataloader/zarr_patch_dataset.py` to:
- Accept **two Zarr archives** simultaneously
- Route requests based on **'source' column** in metadata
- Use **patch_global_index** to index within correct archive

```python
dataset = ZarrPatchDataset(
    zarr_root=zarr_path_norm,              # Primary (normalized)
    indices=train_indices,
    zarr_root_secondary=zarr_path_raw,     # Secondary (raw)
    patch_metadata=patch_metadata           # Combined with 'source' column
)
```

### Training Loop

`src/training/train_phase1.py` updated to:
- Accept **zarr_path_secondary** parameter
- Accept **patch_metadata** parameter
- Initialize dataset in dual mode
- Pass metadata through training pipeline

```python
run_training(
    zarr_path=zarr_path_norm,
    train_indices=train_indices,
    val_indices=val_indices,
    output_dir='outputs/phase1',
    config=phase1_config,
    zarr_path_secondary=zarr_path_raw,     # Enable dual mode
    patch_metadata=patch_metadata           # Enable routing
)
```

## Benefits of Dual-Dataset Training

1. **Stain Invariance**: Model learns from both stain-normalized and original variants
2. **Robustness**: Better generalization to unseen slides with varying stain distributions
3. **Data Diversity**: 4x more training data compared to single dataset
4. **Real-World Performance**: Handles actual histopathology stain variations

## How It Works

### Data Loading Flow

```
patch_metadata (combined) → split indices → Training
    ↓
    Contains 'source' column: 'normalized' or 'raw'
    
During training, for each index:
    ↓
    Metadata lookup: metadata.iloc[idx]['source'] → 'normalized' or 'raw'
    ↓
    If 'normalized': Load from E:\zarr\patches_norm.zarr
    If 'raw': Load from E:\zarr\patches_raw.zarr
    ↓
    Return patch tensor + label to model
```

### Index Mapping

Each index in combined dataset maps to:
1. A position in the combined metadata DataFrame
2. A 'source' field ('normalized' or 'raw')
3. A 'patch_global_index' within that source's Zarr

Example:
```
Combined index 0: source='normalized', patch_global_index=0
Combined index 230120: source='raw', patch_global_index=0
Combined index 230121: source='raw', patch_global_index=1
```

## Training Configuration

### Phase 1 Config (Example)
```python
phase1_config = {
    'seed': 42,
    'batch_size': 16,           # Adjust based on GPU memory
    'num_epochs': 50,
    'num_workers': 4,
    'lr': 1e-4,
    'backbone': 'resnet50',
    'num_classes': 2,           # Binary: HER2+/-
    'dropout': 0.5,
    'amp_enabled': True,        # Automatic Mixed Precision
    'early_stop_patience': 10,
}
```

### GPU Memory Considerations

- **Batch size 16**: ~14GB for ResNet-50 with normalized patches
- **800K training samples**: ~50,000 iterations per epoch (16 batch size)
- **Training time**: ~2-3 hours per epoch on A100 GPU

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir=outputs/phase1/tensorboard_logs
```

Metrics logged:
- Training loss
- Validation loss
- AUC (Area Under Curve)
- Accuracy, Precision, Recall, F1

### Output Files

After training completes:
```
outputs/phase1/
├── best_model.pth              # Best checkpoint (by AUC)
├── last_model.pth              # Final epoch checkpoint
├── metrics.json                # Training history
├── config_used.yaml            # Configuration snapshot
└── tensorboard_logs/           # TensorBoard events
```

## Switching Back to Single Dataset

If needed to use only one dataset:

**Normalized only** (20% of data):
```python
# In Section 5.1.2, load only normalized metadata
patch_metadata = patch_metadata_norm
zarr_path = zarr_path_norm
# Pass to training WITHOUT zarr_path_secondary parameter
```

**Raw only** (80% of data):
```python
# In Section 5.1.2, load only raw metadata
patch_metadata = patch_metadata_raw
zarr_path = zarr_path_raw
# Pass to training WITHOUT zarr_path_secondary parameter
```

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in config (e.g., 8 instead of 16)
- Reduce `num_workers` (e.g., 2 instead of 4)

### Slow Data Loading
- Ensure Zarr files are on fast storage (SSD/NVMe)
- Both `E:\zarr\patches_norm.zarr` and `E:\zarr\patches_raw.zarr` should be accessible
- Check disk I/O usage during first epoch

### Index Out of Range Errors
- Verify both Zarr archives match metadata:
  - Normalized: 230,120 patches
  - Raw: 914,698 patches
- Check that 'source' and 'patch_global_index' columns exist in metadata

## Next Steps

1. **Run Phase 1 Training**: Uncomment cell in Section 5.1.3
2. **Monitor Progress**: Use TensorBoard during training
3. **Evaluate Results**: Check metrics.json for performance
4. **Phase 2 Training**: Use trained Phase 1 checkpoint for feature extraction
5. **Optional: U-Net Training**: Train segmentation model on same data

## References

- Patch Extraction: Section 4.2 in `main.ipynb`
- Dataset Loading: `src/dataloader/zarr_patch_dataset.py`
- Training Module: `src/training/train_phase1.py`
- Models: `src/models/patch_classifier.py`
