"""Tests for evaluation metrics (known values and class imbalance)."""

from __future__ import annotations

import math

import numpy as np

from launch_success.evaluation.metrics import METRIC_NAMES, compute_metrics


def test_perfect_prediction_gives_one() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.2, 0.8]
    metrics = compute_metrics(y_true, y_pred, y_proba)
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert math.isclose(metrics[name], 1.0)


def test_all_metric_keys_present() -> None:
    metrics = compute_metrics([0, 1], [0, 1], [0.2, 0.8])
    assert set(metrics) == set(METRIC_NAMES)


def test_naive_classifier_under_imbalance() -> None:
    # 95% success; model "always predicts success" -> high accuracy but recall=0 for failures.
    y_true = np.array([1] * 95 + [0] * 5)
    y_pred = np.ones(100, dtype=int)  # always predicts success
    metrics = compute_metrics(y_true, y_pred, y_pred.astype(float))
    assert metrics["accuracy"] == 0.95  # high but misleading
    assert metrics["recall"] == 1.0  # recall for the positive class
    # Positive-class F1 is high, but the model is useless for detecting failures.
    assert metrics["precision"] == 0.95


def test_single_class_roc_auc_nan() -> None:
    # Only one class present -> ROC-AUC/PR-AUC are undefined (NaN).
    metrics = compute_metrics([1, 1, 1], [1, 1, 1], [0.9, 0.8, 0.7])
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["pr_auc"])


def test_no_proba_auc_nan() -> None:
    metrics = compute_metrics([0, 1], [0, 1], None)
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["pr_auc"])
