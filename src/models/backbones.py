"""
Backbone networks for feature extraction.
"""

import torch.nn as nn
import torchvision.models as tvm
from typing import Tuple


def get_resnet50_backbone(pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Get ResNet-50 backbone (without classification head).
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        Tuple of (backbone module, feature dimension)
    """
    if pretrained:
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    else:
        backbone = tvm.resnet50(weights=None)
    
    # Remove classification head (avgpool + fc)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
    feat_dim = 2048
    
    return feature_extractor, feat_dim


def get_efficientnet_b0_backbone(pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Get EfficientNet-B0 backbone (without classification head).
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        Tuple of (backbone module, feature dimension)
    """
    if pretrained:
        backbone = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        backbone = tvm.efficientnet_b0(weights=None)
    
    # Get features without classifier
    feature_extractor = nn.Sequential(*list(backbone.features.children()))
    feat_dim = 1280
    
    return feature_extractor, feat_dim


def get_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Get backbone by name.
    
    Args:
        backbone_name: Name of backbone ('resnet50' or 'efficientnet_b0')
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        Tuple of (backbone module, feature dimension)
    """
    if backbone_name == 'resnet50':
        return get_resnet50_backbone(pretrained)
    elif backbone_name == 'efficientnet_b0':
        return get_efficientnet_b0_backbone(pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
