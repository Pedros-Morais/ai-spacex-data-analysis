"""Evaluation metrics for imbalanced binary classification.

``accuracy`` is reported for completeness, but model selection and result
interpretation prioritise ``f1``, ``recall``, ``precision``, ``roc_auc`` and
``pr_auc`` (average precision) — robust under class imbalance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Canonical metric order (used in tables and reports).
METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
)

ArrayLike = Sequence[float] | np.ndarray


def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: ArrayLike | None = None,
) -> dict[str, float]:
    """Compute the full set of binary classification metrics.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).
        y_proba: Positive-class probabilities (for ROC-AUC / PR-AUC).
            If ``None``, those metrics return ``NaN``.

    Returns:
        Dictionary ``{metric_name: value}`` keyed by :data:`METRIC_NAMES`.
        Undefined metrics (e.g. ROC-AUC when only one class is present)
        return ``NaN``.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
    }

    # ROC-AUC and PR-AUC require probabilities and both classes to be present.
    if y_proba is not None and len(np.unique(y_true_arr)) == 2:
        y_proba_arr = np.asarray(y_proba)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba_arr))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_proba_arr))

    return metrics
