"""
MIL pooling for WSI/bag-level classification: mean, max, attention, top-k, quantile.
Pure numpy, normalized attention, robust to varying bag sizes.
"""
from __future__ import annotations
from typing import Optional

import numpy as np


def mean_pool(instance_probs: np.ndarray) -> np.ndarray:
    """Mean pooling: avg(probs). Uniform importance. Shape (K, C) -> (C,)."""
    return instance_probs.mean(axis=0)


def max_pool(instance_probs: np.ndarray) -> np.ndarray:
    """Max pooling: max(probs). Most suspicious region. Shape (K, C) -> (C,). Outlier-sensitive."""
    return instance_probs.max(axis=0)


def attention_pool(instance_probs: np.ndarray, attn_weights: np.ndarray) -> np.ndarray:
    """Attention pooling: weighted sum (normalized). Shape (K, C) + attn(K,) -> (C,). Ilse et al. 2018."""
    attn = np.asarray(attn_weights).reshape(-1, 1)
    attn = attn / (attn.sum() + 1e-8)
    return (instance_probs * attn).sum(axis=0)


def top_k_mean_pool(instance_probs: np.ndarray, k: int = 5) -> np.ndarray:
    """Top-K mean: avg of top-k per class. Shape (K, C) -> (C,). Robust to outliers, focuses on suspicious regions."""
    k_actual = min(k, instance_probs.shape[0])
    # Vectorized: get top-k indices for all classes at once
    top_k_idx = np.argpartition(instance_probs, -k_actual, axis=0)[-k_actual:]
    cols = np.arange(instance_probs.shape[1])
    return instance_probs[top_k_idx, cols].mean(axis=0)


def quantile_pool(instance_probs: np.ndarray, q: float = 0.9) -> np.ndarray:
    """Quantile pooling: q-th quantile (default 0.9=90%). Shape (K, C) -> (C,). q=0.5=median, q=1.0=max."""
    return np.quantile(instance_probs, q, axis=0)


def compute_attention_from_features(features: np.ndarray, attn_weights: np.ndarray, attn_bias: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute attention scores: features @ weights + bias. Shape (K, D) @ (D,) -> (K,). Not normalized."""
    attn_weights = np.asarray(attn_weights).reshape(-1, 1)
    scores = features @ attn_weights
    if attn_bias is not None:
        scores = scores + attn_bias
    return scores.squeeze()


def get_top_attention_indices(attn_weights: np.ndarray, k: int = 10) -> np.ndarray:
    """Get top-k indices by attention (highest first). For heatmap visualization."""
    k_actual = min(k, len(attn_weights))
    return np.argsort(attn_weights)[-k_actual:][::-1]
