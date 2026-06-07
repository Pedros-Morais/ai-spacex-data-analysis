"""Extended tests for visualization module.

Tests plot generation and file saving across different scenarios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from launch_success.config import Settings
from launch_success.data.synthetic import generate_synthetic_launches
from launch_success.evaluation.plots import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
    plot_success_rate_by,
    plot_target_distribution,
)
from launch_success.features.cleaning import clean_launches


@pytest.fixture
def plot_settings(tmp_path) -> Settings:
    """Settings with temporary figures directory."""
    settings = Settings(figures_dir=tmp_path / "figures")
    settings.ensure_directories()
    return settings


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Sample DataFrame for plotting."""
    df = generate_synthetic_launches(n_rows=200, seed=42)
    return clean_launches(df, target="success")


class TestTargetDistribution:
    """Tests for target distribution plot."""

    def test_creates_png_file(self, sample_data, plot_settings) -> None:
        path = plot_target_distribution(sample_data, target="success", settings=plot_settings)

        assert path.exists()
        assert path.suffix == ".png"

    def test_custom_filename(self, sample_data, plot_settings) -> None:
        path = plot_target_distribution(
            sample_data,
            target="success",
            settings=plot_settings,
            filename="custom_dist.png",
        )

        assert path.name == "custom_dist.png"
        assert path.exists()


class TestSuccessRateBy:
    """Tests for success rate by category plot."""

    def test_by_rocket(self, sample_data, plot_settings) -> None:
        path = plot_success_rate_by(
            sample_data,
            by="rocket",
            target="success",
            settings=plot_settings,
        )

        assert path.exists()
        assert "rocket" in path.name

    def test_by_orbit(self, sample_data, plot_settings) -> None:
        path = plot_success_rate_by(
            sample_data,
            by="orbit",
            target="success",
            settings=plot_settings,
        )

        assert path.exists()

    def test_by_year(self, sample_data, plot_settings) -> None:
        path = plot_success_rate_by(
            sample_data,
            by="year",
            target="success",
            settings=plot_settings,
        )

        assert path.exists()


class TestConfusionMatrix:
    """Tests for confusion matrix plot."""

    def test_creates_png(self, plot_settings) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

        path = plot_confusion_matrix(
            y_true,
            y_pred,
            settings=plot_settings,
            title="Test Model",
        )

        assert path.exists()
        assert path.suffix == ".png"

    def test_perfect_predictions(self, plot_settings) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        path = plot_confusion_matrix(
            y_true,
            y_pred,
            settings=plot_settings,
            filename="cm_perfect.png",
        )

        assert path.exists()

    def test_all_wrong_predictions(self, plot_settings) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        path = plot_confusion_matrix(
            y_true,
            y_pred,
            settings=plot_settings,
            filename="cm_worst.png",
        )

        assert path.exists()


class TestROCCurves:
    """Tests for ROC curve plots."""

    def test_single_model_roc(self, plot_settings) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.9, 0.8, 0.7, 0.2, 0.85, 0.3, 0.75, 0.6])

        results = {
            "model_a": {
                "y_proba": y_proba,
                "metrics": {"roc_auc": 0.85},
            }
        }

        path = plot_roc_curves(results, y_true, settings=plot_settings)

        assert path.exists()
        assert path.suffix == ".png"

    def test_multiple_models_roc(self, plot_settings) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])

        results = {
            "model_a": {
                "y_proba": np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.15, 0.85, 0.25]),
                "metrics": {"roc_auc": 0.90},
            },
            "model_b": {
                "y_proba": np.array([0.2, 0.3, 0.8, 0.7, 0.6, 0.25, 0.75, 0.35]),
                "metrics": {"roc_auc": 0.75},
            },
        }

        path = plot_roc_curves(results, y_true, settings=plot_settings)

        assert path.exists()


class TestPRCurves:
    """Tests for Precision-Recall curve plots."""

    def test_single_model_pr(self, plot_settings) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.9, 0.8, 0.7, 0.2, 0.85, 0.3, 0.75, 0.6])

        results = {
            "model_a": {
                "y_proba": y_proba,
                "metrics": {"pr_auc": 0.88},
            }
        }

        path = plot_pr_curves(results, y_true, settings=plot_settings)

        assert path.exists()
        assert path.suffix == ".png"

    def test_multiple_models_pr(self, plot_settings) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])

        results = {
            "model_a": {
                "y_proba": np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.15, 0.85, 0.25]),
                "metrics": {"pr_auc": 0.92},
            },
            "model_b": {
                "y_proba": np.array([0.2, 0.3, 0.8, 0.7, 0.6, 0.25, 0.75, 0.35]),
                "metrics": {"pr_auc": 0.78},
            },
        }

        path = plot_pr_curves(results, y_true, settings=plot_settings)

        assert path.exists()


class TestModelComparison:
    """Tests for model comparison bar plot."""

    def test_creates_comparison_plot(self, plot_settings) -> None:
        results = {
            "logistic_regression": {
                "cv_mean": 0.85,
                "cv_std": 0.03,
                "metrics": {
                    "accuracy": 0.88,
                    "precision": 0.87,
                    "recall": 0.90,
                    "f1": 0.88,
                    "roc_auc": 0.92,
                    "pr_auc": 0.91,
                },
            },
            "random_forest": {
                "cv_mean": 0.87,
                "cv_std": 0.02,
                "metrics": {
                    "accuracy": 0.90,
                    "precision": 0.89,
                    "recall": 0.91,
                    "f1": 0.90,
                    "roc_auc": 0.94,
                    "pr_auc": 0.93,
                },
            },
        }

        path = plot_model_comparison(results, settings=plot_settings)

        assert path.exists()
        assert path.suffix == ".png"

    def test_single_model_comparison(self, plot_settings) -> None:
        results = {
            "single_model": {
                "cv_mean": 0.80,
                "cv_std": 0.05,
                "metrics": {
                    "accuracy": 0.82,
                    "precision": 0.80,
                    "recall": 0.85,
                    "f1": 0.82,
                    "roc_auc": 0.88,
                    "pr_auc": 0.86,
                },
            },
        }

        path = plot_model_comparison(results, settings=plot_settings)

        assert path.exists()

    def test_many_models_comparison(self, plot_settings) -> None:
        results = {}
        for i in range(5):
            base = 0.7 + i * 0.05
            results[f"model_{i}"] = {
                "cv_mean": base,
                "cv_std": 0.02,
                "metrics": {
                    "accuracy": base + 0.02,
                    "precision": base + 0.01,
                    "recall": base + 0.03,
                    "f1": base + 0.02,
                    "roc_auc": base + 0.05,
                    "pr_auc": base + 0.04,
                },
            }

        path = plot_model_comparison(results, settings=plot_settings)

        assert path.exists()
