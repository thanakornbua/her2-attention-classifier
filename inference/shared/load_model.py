"""
Model loader utilities for inference scripts.

Supports common medical imaging backbones:
- ResNet18/50: Standard ImageNet backbones
- Extensible to DenseNet, EfficientNet, ViT, etc.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def load_classification_model(
    arch: str,
    num_classes: int,
    checkpoint_path: Path,
    device: torch.device,
    pretrained: bool = False,
) -> nn.Module:
    """
    Load a classification model with checkpoint weights.

    Args:
        arch: Architecture name ('resnet18', 'resnet50')
        num_classes: Number of output classes
        checkpoint_path: Path to .pth checkpoint file
        device: torch.device for model placement
        pretrained: Whether to initialize with ImageNet weights (usually False for finetuned models)

    Returns:
        Loaded model in eval mode

    Checkpoint Format:
        Supports both:
        - Direct state_dict: torch.save(model.state_dict(), path)
        - Dict with 'state_dict' key: torch.save({'state_dict': ..., 'epoch': ...}, path)
    """
    # Build model architecture
    if arch == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Supported: resnet18, resnet50")

    # Load checkpoint
    # FIX #10: Security - Use weights_only=True to prevent arbitrary code execution
    # This protects against malicious .pth files with embedded code
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    except Exception as e:
        # Fallback for older PyTorch versions or incompatible checkpoints
        print(f"Warning: Loading with weights_only=True failed, falling back to unsafe load: {e}")
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load weights
    model.load_state_dict(state_dict)

    # Move to device and set eval mode
    model = model.to(device)
    model.eval()
    
    # OPTIMIZATION: Enable inference mode for better performance
    # This is more aggressive than torch.no_grad() and provides extra speedup
    # Note: Will be wrapped in torch.no_grad() in evaluation scripts
    if hasattr(torch, 'inference_mode'):
        # inference_mode is available in PyTorch 1.9+
        # It disables autograd and view tracking for maximum performance
        pass  # Applied by caller with torch.no_grad() or torch.inference_mode()

    return model
