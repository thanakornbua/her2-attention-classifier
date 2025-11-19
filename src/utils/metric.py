"""
Medical metrics: Classification (accuracy/P/R/F1/AUROC/confusion), Segmentation (Dice/IoU/HD95/ASSD), Fairness.
Graceful degradation for edge cases (single class, empty masks, missing scipy).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# ======================== Classification ========================

def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], average: str = "macro") -> Dict[str, float]:
    """Accuracy, precision, recall, F1 (macro/micro/weighted)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def multiclass_auroc(y_true: Iterable[int], y_pred_probs: np.ndarray) -> float:
    """Macro AUROC (one-vs-rest). Returns NaN if single class present."""
    y_true = np.asarray(y_true)
    if y_pred_probs.ndim != 2:
        raise ValueError("y_pred_probs must be 2D (N, num_classes)")
    try:
        return float(roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def confusion_matrix_and_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (cm, tp, fn, fp, tn) where cm is (C, C) and others are (C,)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn, fp = cm.sum(axis=1) - tp, cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fn + fp)
    return cm, tp, fn, fp, tn


def group_metrics(y_true: Iterable[int], y_pred: Iterable[int], groups: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Per-cohort accuracy + fairness gap. Returns {group: {count, accuracy, gap_vs_overall}, 'overall', 'gap_max'}."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    overall_acc = float(accuracy_score(y_true, y_pred))
    
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum():
            acc_g = float(accuracy_score(y_true[mask], y_pred[mask]))
            result[g] = {"count": int(mask.sum()), "accuracy": acc_g, "gap_vs_overall": overall_acc - acc_g}
    
    gap_max = max((abs(v["gap_vs_overall"]) for v in result.values()), default=0.0) if result else 0.0
    result["overall"] = {"count": len(y_true), "accuracy": overall_acc, "gap_vs_overall": 0.0}
    result["gap_max"] = {"value": gap_max}
    return result


def write_confusion_matrix_csv(cm: np.ndarray, class_names: List[str], out_path: Path | str) -> Path:
    """Save confusion matrix to CSV with labels."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("," + ",".join(class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{class_names[i]},{','.join(map(str, row.tolist()))}\n")
    return out_path


def write_counts_csv(tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, tn: np.ndarray, class_names: List[str], out_path: Path | str) -> Path:
    """Save TP/FN/FP/TN per class to CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("Class,TP,FN,FP,TN\n")
        for i, name in enumerate(class_names):
            f.write(f"{name},{int(tp[i])},{int(fn[i])},{int(fp[i])},{int(tn[i])}\n")
    return out_path


# ======================== Segmentation ========================

def dice_coefficient(pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-6) -> float:
    """Dice = 2|A∩B|/(|A|+|B|). F1 for binary masks."""
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = (pred & target).sum()
    return float((2 * intersection + epsilon) / (pred.sum() + target.sum() + epsilon))


def iou_score(pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-6) -> float:
    """IoU = |A∩B|/|A∪B|. Jaccard index."""
    pred, target = pred.astype(bool), target.astype(bool)
    intersection, union = (pred & target).sum(), (pred | target).sum()
    return float((intersection + epsilon) / (union + epsilon))


def _get_surface_distances(pred: np.ndarray, target: np.ndarray):
    """Helper to compute bidirectional surface distances. Returns (d1, d2) or None if scipy missing/empty masks."""
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        return None
    pred_pts, tgt_pts = np.argwhere(pred > 0), np.argwhere(target > 0)
    if len(pred_pts) == 0 or len(tgt_pts) == 0:
        return None
    return cdist(pred_pts, tgt_pts).min(axis=1), cdist(tgt_pts, pred_pts).min(axis=1)


def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    """HD95: 95th percentile surface distance. Returns NaN if scipy unavailable or empty masks."""
    distances = _get_surface_distances(pred, target)
    if distances is None:
        return float("nan")
    d1, d2 = distances
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def assd(pred: np.ndarray, target: np.ndarray) -> float:
    """ASSD: mean symmetric surface distance. Returns NaN if scipy unavailable or empty masks."""
    distances = _get_surface_distances(pred, target)
    if distances is None:
        return float("nan")
    d1, d2 = distances
    return float((d1.mean() + d2.mean()) / 2.0)


# ======================== Legacy Wrappers ========================

def calculate_classification_metrics(outputs, labels):
    """Legacy wrapper. Use classification_metrics() directly."""
    return classification_metrics(np.asarray(labels), np.asarray(outputs))


def calculate_segmentation_metrics(pred_mask, true_mask):
    """Legacy wrapper. Returns {dice, iou, hd95, assd}."""
    return {"dice": dice_coefficient(pred_mask, true_mask), "iou": iou_score(pred_mask, true_mask),
            "hd95": hausdorff_95(pred_mask, true_mask), "assd": assd(pred_mask, true_mask)}