import torch
import torch.nn as nn


def weighted_categorical_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor):
    """
    Weighted cross entropy for multi-class classification.
    logits: (B, C), targets: (B,), weights: (C,)
    """
    ce = nn.CrossEntropyLoss(weight=weights)
    return ce(logits, targets)


def iou_loss(pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
    """Simple IoU loss = 1 - IoU for bbox in normalized [0,1] coords."""
    # pred_bboxes, gt_bboxes: (B, 4) [x, y, w, h]
    px, py, pw, ph = pred_bboxes.unbind(dim=1)
    gx, gy, gw, gh = gt_bboxes.unbind(dim=1)

    # convert to (x1,y1,x2,y2)
    p_x1 = px - pw / 2
    p_y1 = py - ph / 2
    p_x2 = px + pw / 2
    p_y2 = py + ph / 2

    g_x1 = gx - gw / 2
    g_y1 = gy - gh / 2
    g_x2 = gx + gw / 2
    g_y2 = gy + gh / 2

    inter_x1 = torch.maximum(p_x1, g_x1)
    inter_y1 = torch.maximum(p_y1, g_y1)
    inter_x2 = torch.minimum(p_x2, g_x2)
    inter_y2 = torch.minimum(p_y2, g_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_g = (g_x2 - g_x1).clamp(min=0) * (g_y2 - g_y1).clamp(min=0)

    union = area_p + area_g - inter_area + 1e-6
    iou = inter_area / union
    return 1.0 - iou.mean()


def l1_iou_localization_loss(pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, alpha: float = 0.5):
    """Combined L1 + IoU loss."""
    l1 = torch.nn.functional.l1_loss(pred_bboxes, gt_bboxes)
    iou = iou_loss(pred_bboxes, gt_bboxes)
    return l1 + alpha * iou


def compute_multi_loss(cls_outputs: torch.Tensor,
                       loc_outputs: torch.Tensor,
                       cls_targets: torch.Tensor,
                       loc_targets: torch.Tensor,
                       class_weights: torch.Tensor,
                       alpha: float = 1.0) -> torch.Tensor:
    """
    L_total = L_cls + alpha * L_loc
    """
    l_cls = weighted_categorical_cross_entropy(cls_outputs, cls_targets, class_weights)
    l_loc = l1_iou_localization_loss(loc_outputs, loc_targets)
    return l_cls + alpha * l_loc
