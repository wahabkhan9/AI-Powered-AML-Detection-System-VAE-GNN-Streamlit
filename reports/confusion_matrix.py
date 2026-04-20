"""
reports/confusion_matrix.py
============================
Confusion matrix and classification report visualisation utilities.

Generates publication-quality PNG confusion matrix figures and
structured JSON evaluation reports for compliance documentation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    normalize: bool = True,
) -> None:
    """
    Render and save a styled confusion matrix figure.

    Parameters
    ----------
    y_true      : ground-truth labels
    y_pred      : predicted labels
    class_names : list of class label strings
    title       : figure title
    save_path   : path to save PNG (default: reports/confusion_matrix.png)
    normalize   : if True, normalise rows to percentages
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        log.warning("matplotlib or sklearn not available – skipping plot.")
        return

    if class_names is None:
        class_names = ["Normal", "Suspicious"]

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm_display[i, j]:.2%}" if normalize else str(cm[i, j])
            ax.text(
                j, i, val,
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=13,
            )

    plt.tight_layout()
    save_path = save_path or (REPORTS_DIR / "confusion_matrix.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix saved → %s", save_path)


# ---------------------------------------------------------------------------
def save_evaluation_report(
    metrics: Dict,
    model_name: str = "VAE",
    path: Optional[Path] = None,
) -> None:
    """
    Persist a structured evaluation report JSON to disk.

    Parameters
    ----------
    metrics    : dict of metric_name → value
    model_name : label for the report header
    path       : output JSON path
    """
    from datetime import datetime

    report = {
        "model": model_name,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in metrics.items()},
    }
    path = path or (REPORTS_DIR / f"{model_name.lower()}_evaluation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    log.info("Evaluation report saved → %s", path)


# ---------------------------------------------------------------------------
def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Precision–Recall Curve",
    save_path: Optional[Path] = None,
) -> None:
    """Render and save a Precision–Recall curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import average_precision_score, precision_recall_curve
    except ImportError:
        log.warning("matplotlib / sklearn not available.")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(recall, precision, where="post", color="#1976D2", linewidth=2.0)
    ax.fill_between(recall, precision, alpha=0.15, color="#1976D2", step="post")
    ax.axhline(
        y=y_true.mean(), color="red", linestyle="--", linewidth=1.2,
        label=f"Baseline (prevalence={y_true.mean():.3f})"
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"{title}  (AP = {ap:.4f})", fontsize=13, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = save_path or (REPORTS_DIR / "precision_recall_curve.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PR curve saved → %s", save_path)
