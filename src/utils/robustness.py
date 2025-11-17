"""
Robustness evaluation: TTA and corruption testing for medical AI.

TTA: Flip-4 (H/V/Both) | Corruptions: Noise, Blur, JPEG, Brightness
Deltas: clean_acc - corrupted_acc (positive = degradation)
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as TF


@torch.no_grad()
def tta_flip4_predict(model, images: torch.Tensor) -> torch.Tensor:
    """4-way flip TTA: average logits from original + H/V/Both flips."""
    logits = torch.stack([
        model(images),
        model(TF.hflip(images)),
        model(TF.vflip(images)),
        model(TF.vflip(TF.hflip(images))),
    ]).mean(0)
    return torch.softmax(logits, dim=1)


# Corruption functions (vectorized where possible)
def _gaussian_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    return torch.clamp(x + std * torch.randn_like(x), 0, 1)


def _gaussian_blur(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    return TF.gaussian_blur(x, kernel_size) if hasattr(TF, 'gaussian_blur') else x


def _jpeg_compression(x: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """Simplified JPEG simulation via color quantization."""
    return torch.stack([
        TF.to_tensor(TF.to_pil_image(x[i]).quantize(colors=quality, method=2).convert("RGB"))
        for i in range(x.shape[0])
    ])


def _brightness_contrast(x: torch.Tensor, brightness: float = 0.2, contrast: float = 0.2) -> torch.Tensor:
    """Combined brightness/contrast adjustment."""
    return torch.stack([
        TF.to_tensor(TF.adjust_contrast(TF.adjust_brightness(TF.to_pil_image(x[i]), 1 + brightness), 1 + contrast))
        for i in range(x.shape[0])
    ])


CORRUPTIONS: Dict[str, Callable] = {
    "gaussian_noise": _gaussian_noise,
    "gaussian_blur": _gaussian_blur,
    "jpeg_compression": _jpeg_compression,
    "brightness_contrast": _brightness_contrast,
}


def _compute_accuracy(model, dataloader, device, corruption_fn: Callable = None) -> float:
    """Helper: compute accuracy with optional corruption applied."""
    y_true, y_pred = [], []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        if corruption_fn:
            images = corruption_fn(images)
        preds = torch.softmax(model(images), dim=1).argmax(dim=1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
    
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y_true, y_pred))


@torch.no_grad()
def evaluate_corruptions(model, dataloader, device, corruption_fns: Dict[str, Callable] = None) -> Dict[str, float]:
    """
    Evaluate robustness under corruptions.
    
    Returns: {"clean_acc": 0.92, "gaussian_noise": 0.03, ...}
    Delta > 0 = degradation | delta < 5% = robust | delta > 10% = sensitive
    """
    model.eval()
    corruption_fns = corruption_fns or CORRUPTIONS
    
    # Clean baseline
    clean_acc = _compute_accuracy(model, dataloader, device)
    
    # Evaluate each corruption
    deltas = {"clean_acc": clean_acc}
    for name, fn in corruption_fns.items():
        corrupted_acc = _compute_accuracy(model, dataloader, device, corruption_fn=fn)
        deltas[name] = clean_acc - corrupted_acc
    
    return deltas
