import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


def calculate_classification_metrics(targets: np.ndarray,
                                     predictions: np.ndarray,
                                     probabilities: np.ndarray):
    """Compute Accuracy, Precision, Recall, F1, AUC per class, Confusion Matrix."""
    acc = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average=None)

    # one-vs-rest AUC per class
    num_classes = probabilities.shape[1]
    auc_per_class = []
    for c in range(num_classes):
        y_true = (targets == c).astype(int)
        y_score = probabilities[:, c]
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = np.nan
        auc_per_class.append(auc)

    cm = confusion_matrix(targets, predictions)

    return {
        "accuracy": acc,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "auc_per_class": np.array(auc_per_class),
        "confusion_matrix": cm,
    }


def intersection_over_union(box_p: np.ndarray, box_gt: np.ndarray) -> float:
    """IoU for boxes [x,y,w,h] normalized in [0,1]."""
    px, py, pw, ph = box_p
    gx, gy, gw, gh = box_gt

    p_x1 = px - pw / 2
    p_y1 = py - ph / 2
    p_x2 = px + pw / 2
    p_y2 = py + ph / 2

    g_x1 = gx - gw / 2
    g_y1 = gy - gh / 2
    g_x2 = gx + gw / 2
    g_y2 = gy + gh / 2

    inter_x1 = max(p_x1, g_x1)
    inter_y1 = max(p_y1, g_y1)
    inter_x2 = min(p_x2, g_x2)
    inter_y2 = min(p_y2, g_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_p = max(0.0, p_x2 - p_x1) * max(0.0, p_y2 - p_y1)
    area_g = max(0.0, g_x2 - g_x1) * max(0.0, g_y2 - g_y1)

    union = area_p + area_g - inter_area + 1e-6
    return inter_area / union


def calculate_localization_metrics(pred_bboxes: np.ndarray, gt_bboxes: np.ndarray):
    """
    Compute IoU, MAE_center, MAE_size, MAE_bbox.
    """
    assert pred_bboxes.shape == gt_bboxes.shape
    n = pred_bboxes.shape[0]

    ious = []
    mae_center = []
    mae_size = []
    mae_bbox = []

    for i in range(n):
        p = pred_bboxes[i]
        g = gt_bboxes[i]
        ious.append(intersection_over_union(p, g))
        mae_center.append(np.abs(p[:2] - g[:2]).mean())
        mae_size.append(np.abs(p[2:] - g[2:]).mean())
        mae_bbox.append(np.abs(p - g).mean())

    return {
        "mean_iou": float(np.mean(ious)),
        "mae_center": float(np.mean(mae_center)),
        "mae_size": float(np.mean(mae_size)),
        "mae_bbox": float(np.mean(mae_bbox)),
    }
