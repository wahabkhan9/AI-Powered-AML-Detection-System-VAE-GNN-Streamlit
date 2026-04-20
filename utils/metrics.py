"""
utils/metrics.py
================
AML-specific evaluation metrics and threshold optimisation utilities.

In AML contexts, recall (sensitivity) is the primary metric because
missing a true money-laundering transaction (false negative) is far
costlier than a false alarm (false positive).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true  : ground-truth binary labels (0/1)
    y_pred  : predicted binary labels
    y_score : continuous anomaly/probability scores (for AUC)

    Returns
    -------
    dict of metric_name → float
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    f_beta_2 = (1 + 4) * precision * recall / max(4 * precision + recall, 1e-9)

    metrics: Dict[str, float] = {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "f2_score": round(f_beta_2, 4),   # recall-weighted F-beta
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "false_negative_rate": round(fn / max(fn + tp, 1), 4),
    }

    if y_score is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_score), 4)
            metrics["avg_precision"] = round(average_precision_score(y_true, y_score), 4)
        except ValueError:
            pass

    return metrics


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------
def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f2",
    min_recall: float = 0.90,
) -> Tuple[float, Dict[str, float]]:
    """
    Search for the score threshold that maximises a metric subject to
    a minimum recall constraint (important for AML).

    Parameters
    ----------
    y_true     : binary ground-truth labels
    y_score    : continuous anomaly scores
    metric     : optimisation target – 'f1', 'f2', or 'precision'
    min_recall : minimum acceptable recall (default 0.90)

    Returns
    -------
    threshold : optimal float threshold
    best_metrics : dict of metrics at optimal threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns N+1 points; align with N thresholds
    precisions, recalls = precisions[:-1], recalls[:-1]

    best_score = -np.inf
    best_thresh = thresholds[0]
    best_metrics: Dict = {}

    for t, p, r in zip(thresholds, precisions, recalls):
        if r < min_recall:
            continue
        if metric == "f1":
            score = 2 * p * r / max(p + r, 1e-9)
        elif metric == "f2":
            score = 5 * p * r / max(4 * p + r, 1e-9)
        else:
            score = p

        if score > best_score:
            best_score = score
            best_thresh = t
            y_pred = (y_score >= t).astype(int)
            best_metrics = compute_classification_metrics(y_true, y_pred, y_score)
            best_metrics["threshold"] = round(float(t), 8)

    return float(best_thresh), best_metrics


# ---------------------------------------------------------------------------
# SAR cost model
# ---------------------------------------------------------------------------
def sar_cost_model(
    tp: int, fp: int, fn: int,
    cost_fn: float = 100_000.0,
    cost_fp: float = 500.0,
) -> Dict[str, float]:
    """
    Estimate financial/regulatory cost of detection outcomes.

    Parameters
    ----------
    tp, fp, fn      : confusion matrix counts
    cost_fn         : cost of each missed suspicious transaction (USD)
    cost_fp         : cost of each false alert (analyst time, USD)
    """
    total_cost = fn * cost_fn + fp * cost_fp
    return {
        "total_cost_usd": total_cost,
        "missed_laundering_cost_usd": fn * cost_fn,
        "false_alert_cost_usd": fp * cost_fp,
        "cost_per_true_positive_usd": total_cost / max(tp, 1),
    }
