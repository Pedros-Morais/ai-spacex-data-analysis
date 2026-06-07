"""Tests for EDA and evaluation plots (file generation)."""

from __future__ import annotations

import numpy as np

from launch_success.evaluation.plots import (
    plot_confusion_matrix,
    plot_success_rate_by,
    plot_target_distribution,
)


def test_plot_target_distribution(clean_frame, tmp_settings) -> None:
    path = plot_target_distribution(clean_frame, "success", tmp_settings)
    assert path.exists() and path.stat().st_size > 0


def test_plot_success_rate_by(clean_frame, tmp_settings) -> None:
    path = plot_success_rate_by(clean_frame, "orbit", "success", tmp_settings)
    assert path.exists() and path.stat().st_size > 0


def test_plot_confusion_matrix(tmp_settings) -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    path = plot_confusion_matrix(y_true, y_pred, tmp_settings)
    assert path.exists() and path.stat().st_size > 0
