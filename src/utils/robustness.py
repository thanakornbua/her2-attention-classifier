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
    """
    FIX #2: Memory-efficient 4-way flip TTA with sequential processing.
    Old: torch.stack() keeps 4 full batches in GPU memory (4x VRAM usage)
    New: Sequential accumulation with running sum (1x VRAM usage)
    """
    # Accumulate logits sequentially instead of stacking all at once
    logits = model(images)
    logits = logits + model(TF.hflip(images))
    logits = logits + model(TF.vflip(images))
    logits = logits + model(TF.vflip(TF.hflip(images)))
    logits = logits / 4.0  # Average
    return torch.softmax(logits, dim=1)


# Corruption functions (vectorized where possible)
def _gaussian_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    return torch.clamp(x + std * torch.randn_like(x), 0, 1)


def _gaussian_blur(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    return TF.gaussian_blur(x, kernel_size) if hasattr(TF, 'gaussian_blur') else x


def _jpeg_compression(x: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """
    FIX #5: Optimized JPEG simulation with parallel batch processing.
    Old: Python loop with per-image GPU→CPU→PIL→GPU transfers (100x slower)
    New: Parallel batch conversion with ThreadPoolExecutor for CPU-bound ops
    """
    from PIL import Image
    import io
    from concurrent.futures import ThreadPoolExecutor
    
    # Convert entire batch to CPU numpy once
    x_np = x.cpu().numpy().transpose(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]
    x_np = (x_np * 255).astype(np.uint8)
    
    def compress_single(img_np):
        """Compress single image (CPU-bound, can parallelize)."""
        img_pil = Image.fromarray(img_np)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_compressed = Image.open(buffer).convert("RGB")
        return np.array(img_compressed)
    
    # OPTIMIZATION: Parallel JPEG compression for high-load datasets
    # Use ThreadPoolExecutor since PIL is CPU-bound and releases GIL
    batch_size = x_np.shape[0]
    max_workers = min(batch_size, 4)  # Limit workers to avoid overhead
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        compressed = list(executor.map(compress_single, x_np))
    
    # Convert back to tensor once
    result = np.stack(compressed).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return torch.from_numpy(result).to(x.device)


def _brightness_contrast(x: torch.Tensor, brightness: float = 0.2, contrast: float = 0.2) -> torch.Tensor:
    """
    FIX #5: Vectorized brightness/contrast without Python loops.
    Old: Per-image PIL conversions (GPU→CPU→PIL→GPU for each image)
    New: Torch-native vectorized operations on entire batch
    """
    # Vectorized brightness adjustment: scale all pixel values
    x_bright = torch.clamp(x * (1 + brightness), 0, 1)
    
    # Vectorized contrast adjustment: scale around mean
    mean = x_bright.mean(dim=[2, 3], keepdim=True)  # Per-channel mean [B, C, 1, 1]
    x_contrast = torch.clamp((x_bright - mean) * (1 + contrast) + mean, 0, 1)
    
    return x_contrast


CORRUPTIONS: Dict[str, Callable] = {
    "gaussian_noise": _gaussian_noise,
    "gaussian_blur": _gaussian_blur,
    "jpeg_compression": _jpeg_compression,
    "brightness_contrast": _brightness_contrast,
}


def _compute_accuracy(model, dataloader, device, corruption_fn: Callable = None) -> float:
    """
    FIX #1: Pre-allocated arrays to prevent memory leak.
    Helper: compute accuracy with optional corruption applied.
    """
    num_samples = len(dataloader.dataset)
    y_true = np.zeros(num_samples, dtype=np.int64)
    y_pred = np.zeros(num_samples, dtype=np.int64)
    
    idx = 0
    for images, labels in dataloader:
        batch_size = len(labels)
        images = images.to(device, non_blocking=True)
        if corruption_fn:
            images = corruption_fn(images)
        preds = torch.softmax(model(images), dim=1).argmax(dim=1).cpu().numpy()
        
        y_pred[idx:idx+batch_size] = preds
        y_true[idx:idx+batch_size] = labels.cpu().numpy()
        idx += batch_size
    
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y_true, y_pred))


@torch.no_grad()
def evaluate_corruptions(model, dataloader, device, corruption_fns: Dict[str, Callable] = None) -> Dict[str, float]:
    """
    FIX #3 + #8: Memory-efficient streaming corruption evaluation.
    Old: Store 5 full prediction arrays (clean + 4 corruptions) = 400 MB for 10M samples
    New: Streaming accuracy computation with constant memory (only counters)
    
    Returns: {"clean_acc": 0.92, "gaussian_noise": 0.03, ...}
    Delta > 0 = degradation | delta < 5% = robust | delta > 10% = sensitive
    """
    model.eval()
    corruption_fns = corruption_fns or CORRUPTIONS
    
    # FIX #8: Use streaming counters instead of storing all predictions
    # Memory: O(1) instead of O(n_samples * n_corruptions)
    total_samples = 0
    correct_counts = {
        "clean": 0
    }
    for name in corruption_fns.keys():
        correct_counts[name] = 0
    
    # Single pass: evaluate clean + all corruptions per batch
    for images, labels in dataloader:
        batch_size = len(labels)
        images = images.to(device, non_blocking=True)
        labels_np = labels.cpu().numpy()
        
        total_samples += batch_size
        
        # Clean predictions
        preds_clean = torch.softmax(model(images), dim=1).argmax(dim=1).cpu().numpy()
        correct_counts["clean"] += (preds_clean == labels_np).sum()
        
        # Corrupted predictions (apply each corruption sequentially to avoid GPU memory spike)
        for name, fn in corruption_fns.items():
            try:
                images_corrupted = fn(images)
                preds_corrupted = torch.softmax(model(images_corrupted), dim=1).argmax(dim=1).cpu().numpy()
                correct_counts[name] += (preds_corrupted == labels_np).sum()
                
                # FIX #11: Clear corrupted images immediately to free GPU memory
                del images_corrupted
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"WARNING: OOM during {name} corruption, skipping this batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    # Compute accuracies from counters
    clean_acc = float(correct_counts["clean"]) / total_samples if total_samples > 0 else 0.0
    
    deltas = {"clean_acc": clean_acc}
    for name in corruption_fns.keys():
        corrupted_acc = float(correct_counts[name]) / total_samples if total_samples > 0 else 0.0
        deltas[name] = clean_acc - corrupted_acc
    
    return deltas
