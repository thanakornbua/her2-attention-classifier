"""
Attention visualization utilities for MIL models.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional


def visualize_top_instances(
    attention_weights: np.ndarray,
    patch_images: np.ndarray,
    top_k: int = 5,
    output_path: Optional[Path] = None
) -> None:
    """
    Visualize top-k attended instances.
    
    Args:
        attention_weights: Attention scores [num_patches]
        patch_images: Patch images [num_patches, H, W, 3]
        top_k: Number of top instances to visualize
        output_path: Optional path to save visualization
    """
    top_indices = np.argsort(attention_weights)[-top_k:][::-1]
    
    fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 4))
    if top_k == 1:
        axes = [axes]
    
    for i, idx in enumerate(top_indices):
        ax = axes[i]
        ax.imshow(patch_images[idx].astype(np.uint8))
        ax.set_title(f"Attention: {attention_weights[idx]:.3f}")
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def create_attention_heatmap(
    attention_weights: np.ndarray,
    grid_shape: Tuple[int, int],
    patch_size: int = 256,
    output_path: Optional[Path] = None
) -> np.ndarray:
    """
    Create heatmap from attention weights on grid.
    
    Args:
        attention_weights: Attention scores [num_patches]
        grid_shape: Grid dimensions (n_rows, n_cols)
        patch_size: Size of each patch
        output_path: Optional path to save
        
    Returns:
        Heatmap array
    """
    n_rows, n_cols = grid_shape
    heatmap = attention_weights.reshape(n_rows, n_cols)
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    if output_path is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap, cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Attention Weight')
        plt.title('MIL Attention Heatmap')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return heatmap
