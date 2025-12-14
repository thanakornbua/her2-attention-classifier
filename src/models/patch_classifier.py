"""
Patch-level classification model.
"""

import torch
import torch.nn as nn
from .backbones import get_backbone


class PatchClassifier(nn.Module):
    """
    Patch-level binary classifier for HER2 status prediction.
    
    Architecture:
        - Backbone: ResNet-50 or EfficientNet-B0
        - Global Average Pooling
        - Classification head with dropout
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        num_classes: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Args:
            backbone_name: Name of backbone ('resnet50' or 'efficientnet_b0')
            num_classes: Number of output classes
            dropout: Dropout probability in classification head
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Logits tensor [B, num_classes]
        """
        features = self.backbone(x)  # [B, feat_dim, H', W']
        pooled = self.gap(features)  # [B, feat_dim, 1, 1]
        pooled = pooled.flatten(1)   # [B, feat_dim]
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
