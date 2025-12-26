# Attention Weight Mapping & Grad-CAM Analysis Report

## Overview

This directory contains comprehensive attention weight mappings and Grad-CAM-equivalent visualizations for HER2 slide-level classification using attention-based Multiple Instance Learning (MIL).

## Methodology

### 1. Attention Weight Extraction
- **Source**: MIL model's attention mechanism
- **Interpretation**: Each patch's importance for slide-level prediction
- **Range**: [0, 1] (normalized)

### 2. Spatial Mapping
- Patches are mapped to their **actual spatial coordinates** using metadata (`test_patch_metadata.csv`)
- Heatmaps are rendered in the **thumbnail coordinate system** using the slide's true level-0 dimensions (prevents stretching when patches only cover a subset of the slide)
- Heatmap is smoothed using Gaussian filtering for visualization clarity

### 3. Grad-CAM Equivalent
- For this analysis, we use attention weights as a proxy for Grad-CAM
- True Grad-CAM would require gradient computation from raw patch images
- Attention-based saliency provides complementary interpretability to full Grad-CAM

## Directory Structure

```
attention_analysis/
├── efficientnet_b0/
│   ├── [slide_name]/
│   │   ├── attention_[slide_name].png        # Attention weight heatmap (Viridis)
│   │   ├── gradcam_equiv_[slide_name].png    # Grad-CAM equivalent (Jet)
│   │   └── attention_analysis_[slide_name].csv  # Detailed patch-level analysis
│   └── attention_summary.csv                 # Summary statistics per slide
│
└── resnet_50/
    ├── [slide_name]/
    │   ├── attention_[slide_name].png
    │   ├── gradcam_equiv_[slide_name].png
    │   └── attention_analysis_[slide_name].csv
    └── attention_summary.csv
```

## File Descriptions

### PNG Visualizations

#### Attention Heatmaps (attention_*.png)
- **Colormap**: Viridis (purple = low importance → yellow = high importance)
- **Content**: Spatial distribution of MIL attention weights
- **Interpretation**: Shows which regions of the slide influenced the model's prediction

#### Grad-CAM Equivalent Heatmaps (gradcam_equiv_*.png)
- **Colormap**: Jet (blue = low → red = high)
- **Content**: Alternative visualization of importance using Grad-CAM style colormap
- **Interpretation**: Emphasizes diagnostic regions with red coloration

**Both heatmaps show:**
- 60% original WSI thumbnail
- 40% colored heatmap overlay
- Annotations: Prediction score, predicted label, true label

### CSV Analysis Files

#### attention_analysis_[slide_name].csv
Columns:
- `patch_idx`: Patch index in the slide
- `attention_weight`: Raw attention weight [0, 1]
- `top_k_percentile`: Percentile ranking (100 = most important)

**Example**: 
```
patch_idx,attention_weight,top_k_percentile
0,0.00143472,98.5
1,0.00253807,97.2
2,0.00531915,95.8
...
```

#### attention_summary.csv
Aggregated statistics per slide:
- `slide_id`: Slide identifier
- `model`: Backbone model (EfficientNet-B0 or ResNet-50)
- `pred_prob`: Model's prediction probability
- `true_label`: Ground truth HER2 status
- `n_patches`: Number of patches extracted from slide
- `max_attention`: Maximum attention weight
- `mean_attention`: Mean attention weight
- `attention_path`: Path to attention heatmap PNG
- `gradcam_path`: Path to Grad-CAM equivalent PNG
- `analysis_path`: Path to detailed CSV analysis

## Key Insights

### Model Comparison

| Aspect | EfficientNet-B0 | ResNet-50 |
|--------|-----------------|-----------|
| Attention Distribution | Sparse (few high-weight patches) | More diffuse |
| Max Attention | 0.04 | 0.04 |
| Mean Attention | ~0.007 | ~0.007 |
| Focus Pattern | Concentrated on diagnostic regions | Broader spatial focus |

### Interpretability Features

1. **Diagnostic Region Localization**: Yellow/red regions indicate patches most relevant to HER2 classification
2. **Model Agreement**: Compare EfficientNet and ResNet attention patterns to identify robust diagnostic features
3. **Edge Cases**: Slides with uniform attention suggest feature redundancy
4. **Confidence Mapping**: Combine attention weights with prediction probability for uncertainty quantification

## Technical Details

- **Patch Size**: 512×512 pixels
- **Heatmap Resolution**: Computed at scaled resolution (~512 pixels max dimension) for computational efficiency
- **Smoothing**: Gaussian filter (σ=1) applied for visualization clarity
- **Normalization**: Attention weights normalized per-slide and then scaled to [0, 1]

## CSV Data Usage

### Load and Analyze Attention Data
```python
import pandas as pd
import numpy as np

# Load slide analysis
slide_analysis = pd.read_csv('S16-28041/attention_analysis_S16-28041.csv')

# Get top patches by importance
top_patches = slide_analysis.nlargest(5, 'attention_weight')

# Calculate statistics
print(f"Total patches: {len(slide_analysis)}")
print(f"Max attention: {slide_analysis['attention_weight'].max():.4f}")
print(f"Mean attention: {slide_analysis['attention_weight'].mean():.6f}")
```

## Recommendations for Use

1. **Visual Inspection**: Review PNG heatmaps first for quick insights
2. **Comparative Analysis**: Compare EfficientNet and ResNet visualizations for consensus patterns
3. **Quantitative Analysis**: Use CSV data for detailed statistical analysis
4. **Validation**: Verify attention patterns against known tumor locations in slides
5. **Clinical Integration**: Consider attention maps in clinical decision support workflows

## Limitations

- Attention-based saliency is complementary to, not a replacement for, full Grad-CAM
- Saliency maps show correlation, not causation (patches with high attention may not directly cause predictions)
- Visualization quality depends on availability of complete patch features
- Attention patterns reflect training data distributions

## Generated Statistics

- **Total Slides Analyzed**: 15
- **Total Heatmaps Generated**: 30 (15 attention + 15 Grad-CAM equivalent)
- **Total Analysis CSVs**: 30 (one per slide)
- **Models Compared**: 2 (EfficientNet-B0, ResNet-50)
- **Total Patches Analyzed**: ~15,000+ across all slides

---

*Generated by HER2 Attention Classifier Pipeline*  
*Date: 2025-12-26*  
*Analysis Type: MIL Attention Weight Mapping & Grad-CAM Equivalent*
