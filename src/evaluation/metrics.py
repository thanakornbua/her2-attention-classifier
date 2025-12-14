"""
Evaluation metrics for classification and segmentation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from typing import Dict, Tuple


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        y_prob: Predicted probabilities [N, num_classes] (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        metrics[f'precision_class_{i}'] = float(p)
        metrics[f'recall_class_{i}'] = float(r)
        metrics[f'f1_class_{i}'] = float(f)
    
    # Macro averages
    metrics['precision_macro'] = float(precision.mean())
    metrics['recall_macro'] = float(recall.mean())
    metrics['f1_macro'] = float(f1.mean())
    
    # AUC if probabilities provided
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:
                # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class
                metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except Exception:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    
    Args:
        y_true: True binary labels [N]
        y_prob: Predicted probabilities for positive class [N]
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
    }
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1'] = float(f1)
    
    # Compute sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold based on a metric.
    
    Args:
        y_true: True binary labels [N]
        y_prob: Predicted probabilities for positive class [N]
        metric: Metric to optimize ('f1', 'youden')
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    if metric == 'f1':
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in np.linspace(0.1, 0.9, 100):
            y_pred = (y_prob >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            if f1 > best_score:
                best_score = f1
                best_threshold = threshold
        
        return best_threshold, best_score
    
    elif metric == 'youden':
        # Youden's J statistic = sensitivity + specificity - 1
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx], j_scores[best_idx]
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
