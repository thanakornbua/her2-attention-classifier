"""
Reproducibility utilities for deterministic training.

Provides functions to set random seeds across all libraries (Python, NumPy, PyTorch)
for reproducible experiments.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Additional CUDA determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def enable_deterministic_mode():
    """
    Enable fully deterministic training (slower but reproducible).
    Call this in addition to set_seed() for maximum reproducibility.
    """
    torch.use_deterministic_algorithms(True)
