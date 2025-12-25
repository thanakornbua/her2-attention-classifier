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


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute segmentation metrics.
    
    Args:
        pred_mask: Predicted probability mask [H, W] or [N, H, W]
        true_mask: Ground truth mask [H, W] or [N, H, W] (0 or 1)
        threshold: Threshold for binarization
        
    Returns:
        Dictionary of metrics
    """
    # Binarize
    pred_bin = (pred_mask > threshold).astype(bool)
    true_bin = (true_mask > 0).astype(bool)
    
    # Flatten
    pred_flat = pred_bin.ravel()
    true_flat = true_bin.ravel()
    
    # Intersection and Union
    intersection = (pred_flat & true_flat).sum()
    union = (pred_flat | true_flat).sum()
    
    # Confusion Matrix elements
    tp = intersection
    fp = pred_flat.sum() - tp
    fn = true_flat.sum() - tp
    tn = len(pred_flat) - tp - fp - fn
    
    # Metrics
    dice = 2 * intersection / (pred_flat.sum() + true_flat.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    pixel_acc = (tp + tn) / len(pred_flat)
    sensitivity = tp / (tp + fn + 1e-8) # Recall
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'pixel_accuracy': float(pixel_acc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    }


def compute_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Compute Confidence Interval for AUC using bootstrapping.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bootstraps: Number of bootstrap samples
        alpha: Confidence level (e.g., 0.95)
        seed: Random seed
        
    Returns:
        Tuple (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # Bootstrap indices
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            # Skip if sample doesn't have both classes
            continue
            
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    lower_bound = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + alpha) / 2 * 100)
    
    return lower_bound, upper_bound


def delong_roc_test(y_true, y_pred1, y_pred2):
    """
    Computes the p-value of the DeLong test for comparing two ROC curves.
    
    Args:
        y_true: Ground truth labels
        y_pred1: Probabilities from model 1
        y_pred2: Probabilities from model 2
        
    Returns:
        p-value
    """
    import scipy.stats as stats
    
    # Implementation of DeLong's algorithm for comparing AUCs
    # Based on: https://github.com/yandexdataschool/roc_comparison
    
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        T2[J] = T + 1
        return T2

    def fastDeLong(predictions_sorted_transposed, label_1_count):
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float64)
        ty = np.empty([k, n], dtype=np.float64)
        tz = np.empty([k, m + n], dtype=np.float64)
        for r in range(k):
            tx[r, :] = compute_midrank(positive_examples[r, :])
            ty[r, :] = compute_midrank(negative_examples[r, :])
            tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)

    # Sort by label
    order = np.argsort(y_true)
    y_true = y_true[order]
    y_pred1 = y_pred1[order]
    y_pred2 = y_pred2[order]
    
    # Count positives
    num_pos = np.sum(y_true == 1)
    
    # Stack predictions
    predictions = np.vstack((y_pred1, y_pred2))
    
    # Compute DeLong covariance
    aucs, delongcov = fastDeLong(predictions, num_pos)
    
    # Compute Z-score and p-value
    delta_auc = aucs[0] - aucs[1]
    sigma = np.sqrt(delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1])
    z_score = delta_auc / sigma
    p_value = 2 * stats.norm.sf(abs(z_score))
    
    return p_value
