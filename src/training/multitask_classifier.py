"""
Multi-Task Lesion Classifier Module

This module implements a multi-task deep learning model for HER2 lesion classification
and localization in histopathology images.

Architecture:
    - Backbone: ResNet-50 or EfficientNet-B0 (pretrained on ImageNet)
    - Classification Head: 2-layer MLP with dropout (outputs class logits)
    - Localization Head: 2-layer MLP with sigmoid (outputs bounding box coordinates)
    - Feature Extraction: Global Average Pooling (GAP) on backbone features

Key Components:
    - MultiTaskLesionClassifier: Main model class with dual-task heads
    - Loss Functions: Weighted categorical cross-entropy + L1/IoU localization loss
    - Training Utilities: train_epoch() with AMP support and phase-based freezing
    
Loss Functions:
    - Classification: Weighted CE loss for handling class imbalance
    - Localization: Combined L1 + (1 - IoU) loss for bounding box regression
    
Training Strategy:
    - Phase 1: Freeze backbone, train heads only
    - Phase 2: Unfreeze all layers, fine-tune end-to-end
    - Mixed Precision: Configurable AMP support for faster training

Usage Example:
    ```python
    model = MultiTaskLesionClassifier(
        num_classes=3, 
        use_efficientnet_b0=False,
        AMP_ENABLED=True
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Phase 1: Train with frozen backbone
    for epoch in range(15):
        loss = train_epoch(model, train_dl, optimizer, class_weights, 
                          phase_1=True, device=device)
    
    # Phase 2: Fine-tune entire model
    for epoch in range(30):
        loss = train_epoch(model, train_dl, optimizer, class_weights, 
                          phase_1=False, device=device)
    ```

Author: HER2 Classification Pipeline
Date: 2025
"""

# Model initialization and losses/utilities
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm

class MultiTaskLesionClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, use_efficientnet_b0: bool = False, AMP_ENABLED: bool = True):
        super().__init__()
        self.AMP_ENABLED = AMP_ENABLED
        if use_efficientnet_b0:
            backbone = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.feature_extractor = nn.Sequential(*(list(backbone.features.children())))
            self.gap = nn.AdaptiveAvgPool2d((1,1))
            feat_dim = 1280
        else:
            backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            self.feature_extractor = nn.Sequential(*(list(backbone.children())[:-2]))
            self.gap = nn.AdaptiveAvgPool2d((1,1))
            feat_dim = 2048

        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
        self.loc_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        pooled = self.gap(feats).flatten(1)
        cls_logits = self.cls_head(pooled)
        loc = self.loc_head(pooled)
        return cls_logits, loc

def weighted_categorical_cross_entropy(logits, targets, weights):
    return nn.CrossEntropyLoss(weight=weights)(logits, targets)

def intersection_over_union(box_p, box_gt):
    px, py, pw, ph = box_p.unbind(-1)
    gx, gy, gw, gh = box_gt.unbind(-1)
    p_x1 = px - pw/2; p_y1 = py - ph/2; p_x2 = px + pw/2; p_y2 = py + ph/2
    g_x1 = gx - gw/2; g_y1 = gy - gh/2; g_x2 = gx + gw/2; g_y2 = gy + gh/2
    ix1 = torch.maximum(p_x1, g_x1); iy1 = torch.maximum(p_y1, g_y1)
    ix2 = torch.minimum(p_x2, g_x2); iy2 = torch.minimum(p_y2, g_y2)
    iw = torch.clamp(ix2 - ix1, min=0); ih = torch.clamp(iy2 - iy1, min=0)
    inter = iw * ih
    area_p = (p_x2 - p_x1) * (p_y2 - p_y1)
    area_g = (g_x2 - g_x1) * (g_y2 - g_y1)
    union = area_p + area_g - inter + 1e-6
    return inter / union

def l1_iou_localization_loss(pred_bboxes, gt_bboxes):
    l1 = nn.L1Loss()(pred_bboxes, gt_bboxes)
    iou = intersection_over_union(pred_bboxes, gt_bboxes).mean()
    return l1 + (1 - iou)

def compute_multi_loss(cls_outputs, loc_outputs, cls_targets, loc_targets, class_weights, alpha: float = 1.0):
    l_cls = weighted_categorical_cross_entropy(cls_outputs, cls_targets, class_weights)
    l_loc = l1_iou_localization_loss(loc_outputs, loc_targets)
    return l_cls + alpha * l_loc

def set_model_phase(model, phase_1: bool):
    for p in model.feature_extractor.parameters():
        p.requires_grad = not (phase_1)

from torch.amp import autocast, GradScaler

def train_epoch(model, dataloader, optimizer, class_weights, phase_1, device):
    model.train()
    set_model_phase(model, phase_1)
    
    # Create scaler based on model's AMP setting
    scaler = GradScaler('cuda', enabled=model.AMP_ENABLED)
    
    total_loss = 0.0
    for imgs, labels, boxes in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        boxes = boxes.to(device)
        optimizer.zero_grad()
        with autocast('cuda', enabled=model.AMP_ENABLED):
            cls_logits, loc_out = model(imgs)
            loss = compute_multi_loss(cls_logits, loc_out, labels, boxes, class_weights.to(device))
        if model.AMP_ENABLED:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(dataloader.dataset)