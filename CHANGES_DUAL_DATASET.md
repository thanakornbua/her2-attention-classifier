# Changes Made for Dual Dataset Training

## Summary

Updated the training pipeline to use **both normalized and raw patches** simultaneously during Phase 1 training, increasing training data from 230K to 1.1M patches.

## Files Modified

### 1. **main.ipynb** (Notebook)

#### Section 5.1 Header (Updated)
- Changed from "choose which dataset to use" to "uses both datasets combined"
- Added explicit note: "Training: Uses both datasets combined for robust learning"
- Highlighted benefit: "improves model robustness to stain variations"

#### Section 5.1.2 - Dataset Loading Cell (Rewritten)
**Before**: Single dataset selection with `USE_NORMALIZED` flag
```python
USE_NORMALIZED = True  # Toggle between normalized and raw
```

**After**: Combined dataset loading
```python
# Load both CSVs
patch_metadata_norm = pd.read_csv(metadata_csv_norm)  # 230,120 patches
patch_metadata_raw = pd.read_csv(metadata_csv_raw)    # 914,698 patches

# Combine
patch_metadata = pd.concat([patch_metadata_norm, patch_metadata_raw])  # 1,144,818
patch_metadata['source'] = ['normalized' if i < len(patch_metadata_norm) else 'raw' for i in range(...)]

# Split indices from combined dataset
train_indices, val_indices, test_indices = create_splits(patch_metadata)
```

#### Section 5.1.3 - Training Call (Updated)
**Before**: 
```python
run_training(
    zarr_path=zarr_path,
    train_indices=train_indices,
    val_indices=val_indices,
    output_dir='outputs/phase1',
    config=phase1_config
)
```

**After**:
```python
run_training(
    zarr_path=zarr_path,
    train_indices=train_indices,
    val_indices=val_indices,
    output_dir='outputs/phase1',
    config=phase1_config,
    zarr_path_secondary=zarr_path_secondary,  # ← New: Raw Zarr
    patch_metadata=patch_metadata              # ← New: Routing info
)
```

---

### 2. **src/dataloader/zarr_patch_dataset.py** (Dataset Class)

#### Changes
- Added support for **dual Zarr archives** (primary + secondary)
- Added **patch_metadata parameter** for routing between archives
- Added **'source' column awareness** to determine which Zarr to read from
- Updated to read **'her2_status' from metadata** instead of separate labels array
- Maintains backward compatibility with single Zarr mode

#### Key Methods
```python
class ZarrPatchDataset(Dataset):
    def __init__(self, zarr_root, indices, zarr_root_secondary=None, patch_metadata=None):
        # Now supports dual mode if secondary and metadata provided
    
    def __getitem__(self, i):
        idx = int(self.indices[i])
        
        if self.use_dual_zarr:
            source = self.metadata.iloc[idx]['source']
            patch_idx = int(self.metadata.iloc[idx]['patch_global_index'])
            
            if source == 'normalized':
                patch = self.patches_primary[patch_idx]
            else:
                patch = self.patches_secondary[patch_idx]
        else:
            patch = self.patches_primary[idx]
        
        label = int(self.metadata.iloc[idx]['her2_status'])
        # ... preprocessing ...
        return img_t, cls_t, loc_t
```

#### Backward Compatibility
- Single Zarr mode still works: `ZarrPatchDataset(zarr_path, indices)`
- Dual mode requires: `ZarrPatchDataset(zarr_path, indices, zarr_path_secondary, patch_metadata)`

---

### 3. **src/training/train_phase1.py** (Training Module)

#### Changes
- Added **zarr_path_secondary** parameter to `run_training()` function
- Added **patch_metadata** parameter to `run_training()` function
- Updated dataset initialization to use dual mode when parameters provided
- Maintains backward compatibility with single Zarr training

#### Function Signature (Before)
```python
def run_training(
    zarr_path: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    output_dir: str = 'outputs/phase1',
    config: Optional[Dict] = None
):
```

#### Function Signature (After)
```python
def run_training(
    zarr_path: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    output_dir: str = 'outputs/phase1',
    config: Optional[Dict] = None,
    zarr_path_secondary: Optional[str] = None,
    patch_metadata: Optional['pd.DataFrame'] = None
):
```

#### Dataset Initialization Logic
```python
if zarr_path_secondary is not None and patch_metadata is not None:
    print("Using DUAL Zarr archives (normalized + raw)")
    train_ds = ZarrPatchDataset(
        zarr_path, 
        indices=train_indices,
        zarr_root_secondary=zarr_path_secondary,
        patch_metadata=patch_metadata
    )
else:
    print("Using single Zarr archive")
    train_ds = ZarrPatchDataset(zarr_path, indices=train_indices)
```

---

## Data Flow Diagram

```
Section 5.1.2 (Dataset Loading)
├── Load patch_metadata_512_norm.csv (230K patches)
├── Load patch_metadata_512_raw.csv (914K patches)
└── Combine with 'source' column
    ├── patch_metadata_norm['source'] = 'normalized'
    └── patch_metadata_raw['source'] = 'raw'

Combined patch_metadata (1.1M rows)
├── Columns: slide_name, case_name, her2_status, patch_global_index, x, y, source
└── Split into train/val/test indices

Section 5.1.3 (Training Call)
└── run_training(
    zarr_path=E:\zarr\patches_norm.zarr,
    zarr_path_secondary=E:\zarr\patches_raw.zarr,
    patch_metadata=combined_metadata,
    train_indices=[indices with source info]
)

During Training
├── For each batch:
│   ├── Sample random training indices
│   ├── Look up metadata for each index
│   ├── Check 'source' column
│   ├── Route to correct Zarr (normalized or raw)
│   ├── Use 'patch_global_index' to retrieve patch
│   └── Return patch + label
```

---

## Key Improvements

### Dataset Size
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total patches | 230K | 1.1M | +4.97x |
| Training patches | 161K | 801K | +4.97x |
| Validation patches | 23K | 114K | +4.97x |
| Stain variants | 1 (normalized) | 2 (both) | +1 |

### Model Robustness
- **Stain invariance**: Learns from both normalized and original variations
- **Better generalization**: More diverse training data
- **Real-world performance**: Handles actual histopathology stain distributions

---

## Testing & Validation

### Tests Run
1. ✅ **Syntax validation**: Both Python files compile without errors
2. ✅ **Notebook execution**: Section 5.1.2 runs successfully
   - Both CSVs loaded: True, True
   - Both Zarr archives opened: shapes (230120, 512, 512, 3) and (914698, 512, 512, 3)
   - Combined dataset: 1,144,818 patches
   - Train/val/test split: 801,372 / 114,482 / 228,964
3. ✅ **Config validation**: Phase 1 config loaded correctly

### Expected Behavior
- Training will sample from both Zarr archives automatically
- Batch will contain mixture of normalized and raw patches
- Model learns features robust to both stain conditions
- No code changes needed - just uncomment and run

---

## Backward Compatibility

All changes are **backward compatible**:

1. **ZarrPatchDataset**: Single Zarr mode still works
   ```python
   # Old code still works
   dataset = ZarrPatchDataset(zarr_path, indices)
   ```

2. **train_phase1.py**: Single Zarr training still works
   ```python
   # Old code still works
   run_training(zarr_path, train_indices, val_indices)
   ```

3. **Notebook**: Can switch to single dataset if needed by not passing secondary params

---

## Documentation Created

1. **DUAL_DATASET_TRAINING.md**
   - Comprehensive documentation of dual-dataset setup
   - Implementation details
   - Data flow explanations
   - Troubleshooting guide

2. **QUICK_START_DUAL_TRAINING.md**
   - Quick reference for running training
   - Step-by-step instructions
   - Expected performance metrics
   - Common issues and solutions

---

## Next Steps

1. **Run Section 5.1.2** to load combined dataset
2. **Uncomment Section 5.1.3** and run training
3. **Monitor with TensorBoard**: `tensorboard --logdir=outputs/phase1/tensorboard_logs`
4. **After training**: Use best checkpoint for Phase 2 feature extraction

---

## Version Info

- **Updated**: December 15, 2025
- **Python Version**: 3.10+
- **PyTorch**: 2.0+
- **Zarr**: Latest (installed in conda env)
- **NumPy**: Latest
- **Pandas**: Latest
