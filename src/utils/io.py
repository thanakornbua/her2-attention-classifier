"""
I/O utilities for saving/loading checkpoints, configs, and metrics.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: Path,
    config: Dict[str, Any] = None
):
    """
    Save a training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        metrics: Dictionary of metrics (loss, accuracy, etc.)
        save_path: Path to save checkpoint
        config: Optional training configuration
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
    weights_only: bool = False
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to map tensors to
        
    Returns:
        Dictionary containing checkpoint data
    """
    # Allow legacy numpy scalar in checkpoints when weights_only=False (PyTorch 2.6 safety change)
    if not weights_only:
        try:
            import numpy as _np
            torch.serialization.add_safe_globals([_np.core.multiarray.scalar])
        except Exception:
            pass

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_metrics(metrics: Dict[str, Any], save_path: Path):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save JSON file
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary of configuration values
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: Path):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
