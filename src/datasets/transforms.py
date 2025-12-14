"""
Data augmentation and preprocessing transforms.
"""

import torch
import torchvision.transforms as T
from typing import Optional


def get_train_transforms(patch_size: int = 512, augment: bool = True):
    """
    Get training transforms with optional augmentation.
    
    Args:
        patch_size: Size of input patches
        augment: Whether to apply augmentation
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if augment:
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ])
    else:
        return T.Compose([])


def get_val_transforms():
    """
    Get validation transforms (no augmentation).
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    return T.Compose([])


def normalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    """
    Apply ImageNet normalization to image tensor.
    
    Args:
        image: Image tensor in range [0, 1]
        
    Returns:
        Normalized image tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    return (image - mean) / std
