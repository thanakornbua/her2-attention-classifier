"""Utility helpers for testing single patch PNGs against PyTorch .pth models.

Both the CLI and GUI front-ends rely on these helpers so preprocessing,
normalization, model loading, and result formatting remain consistent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from inference.shared.load_model import load_classification_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NORMALIZATION_CHOICES = ("none", "imagenet", "tf")
ACTIVATION_CHOICES = ("auto", "softmax", "sigmoid", "none")

def load_pth_model(
    arch: str,
    num_classes: int,
    model_path: Path,
    *,
    device: Optional[str] = None,
    pretrained: bool = False,
):
    """Load a PyTorch classification model from a .pth checkpoint."""
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_classification_model(arch, num_classes, model_path, target, pretrained=pretrained)
    model.eval()
    return model


def load_class_names_file(path: Path) -> List[str]:
    """Read class names from .json/.txt/.csv file."""
    normalized = path.suffix.lower()
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Class-name file {path} is empty")

    if normalized == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return [str(v) for v in data.values()]
        if isinstance(data, list):
            return [str(v) for v in data]
        raise ValueError("JSON class names must be a list or dict")

    # Plain-text fallback (txt/csv)
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Could not parse any names from {path}")
    return names


def resolve_class_names(raw_names: Optional[Sequence[str]], num_outputs: int) -> List[str]:
    """Pad/trim/raw class names so they match the model outputs."""
    if raw_names is None:
        return [f"class_{idx}" for idx in range(num_outputs)]
    names = [str(n) for n in raw_names]
    if len(names) < num_outputs:
        names.extend(f"class_{idx}" for idx in range(len(names), num_outputs))
    return names[:num_outputs]


def _load_patch_array(patch_path: Path, image_size: int) -> np.ndarray:
    image = Image.open(patch_path).convert("RGB")
    if image_size > 0:
        image = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def _normalize_patch(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "imagenet":
        return (arr - IMAGENET_MEAN) / IMAGENET_STD
    if mode == "tf":
        return (arr - 0.5) * 2.0
    return arr


def prepare_patch_batch(
    patch_path: Path,
    *,
    image_size: int,
    normalization: str,
) -> torch.Tensor:
    """Load + normalize a patch, returning a (1, 3, H, W) float32 tensor."""
    if normalization not in NORMALIZATION_CHOICES:
        raise ValueError(f"Unsupported normalization '{normalization}'")
    arr = _load_patch_array(patch_path, image_size)
    arr = _normalize_patch(arr, normalization)
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def apply_activation(logits: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Convert raw model outputs into probabilities."""
    if mode not in ACTIVATION_CHOICES:
        raise ValueError(f"Unsupported activation '{mode}'")
    preds = np.array(logits).squeeze()
    if preds.ndim == 0:
        preds = np.expand_dims(preds, axis=0)

    if mode == "none":
        return preds.astype(np.float32)

    if mode == "sigmoid" or (mode == "auto" and preds.size == 1):
        prob = 1.0 / (1.0 + np.exp(-preds))
        return prob.astype(np.float32)

    if mode == "softmax" or mode == "auto":
        max_val = np.max(preds)
        exp = np.exp(preds - max_val)
        denom = np.clip(exp.sum(), 1e-12, None)
        return (exp / denom).astype(np.float32)

    raise ValueError(f"Unhandled activation mode '{mode}'")


@torch.inference_mode()
def predict_patch(
    model: torch.nn.Module,
    patch_path: Path,
    *,
    image_size: int,
    normalization: str,
    activation: str,
    device: Optional[str] = None,
) -> np.ndarray:
    batch = prepare_patch_batch(patch_path, image_size=image_size, normalization=normalization)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch = batch.to(target)
    model = model.to(target)
    logits = model(batch).detach().cpu().numpy()
    probs = apply_activation(logits, activation)
    return probs


def format_topk(probabilities: np.ndarray, class_names: Sequence[str], top_k: int) -> List[dict]:
    probs = np.array(probabilities).flatten()
    if probs.ndim != 1:
        raise ValueError("Expected 1D probability vector")
    if len(class_names) != probs.size:
        raise ValueError("class_names length must match probabilities")
    k = max(1, min(top_k, probs.size))
    order = np.argsort(probs)[::-1][:k]
    return [
        {"rank": idx + 1, "class": class_names[label], "score": float(probs[label])}
        for idx, label in enumerate(order)
    ]


@dataclass
class PatchTestResult:
    probabilities: List[float]
    class_names: List[str]
    top_k: List[dict]


def build_result(probabilities: np.ndarray, class_names: List[str], top_k: int) -> PatchTestResult:
    probs_list = [float(p) for p in np.array(probabilities).flatten()]
    topk_rows = format_topk(np.array(probs_list), class_names, top_k)
    return PatchTestResult(probabilities=probs_list, class_names=class_names, top_k=topk_rows)


__all__ = [
    "ACTIVATION_CHOICES",
    "NORMALIZATION_CHOICES",
    "PatchTestResult",
    "apply_activation",
    "build_result",
    "format_topk",
    "load_class_names_file",
    "load_pth_model",
    "predict_patch",
    "prepare_patch_batch",
    "resolve_class_names",
]
