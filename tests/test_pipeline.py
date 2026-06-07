"""Smoke test for the end-to-end pipeline on small synthetic data."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from launch_success.data.synthetic import write_synthetic_dataset
from launch_success.evaluation.metrics import METRIC_NAMES
from launch_success.pipeline import build_metrics_table, run_pipeline


def _fast_registry(settings=None):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=30, class_weight="balanced", random_state=42
        ),
    }


def test_pipeline_end_to_end(tmp_settings, monkeypatch) -> None:
    # Reduced registry for a fast smoke test.
    monkeypatch.setattr("launch_success.models.trainer.get_model_registry", _fast_registry)
    write_synthetic_dataset(n_rows=300, settings=tmp_settings)

    summary = run_pipeline(tmp_settings, generate_plots=True, run_shap=True)

    assert summary["best_name"] in {"logistic_regression", "random_forest"}
    assert isinstance(summary["metrics_table"], pd.DataFrame)
    assert summary["model_path"].exists()
    assert tmp_settings.metrics_path.exists()
    # EDA (4) + evaluation (4) + SHAP (3) = 11 expected figures.
    assert len(summary["figures"]) >= 8
    for path in summary["figures"].values():
        assert path.exists()


def test_pipeline_without_plots_or_shap(tmp_settings, monkeypatch) -> None:
    monkeypatch.setattr("launch_success.models.trainer.get_model_registry", _fast_registry)
    write_synthetic_dataset(n_rows=200, settings=tmp_settings)
    summary = run_pipeline(tmp_settings, generate_plots=False, run_shap=False)
    assert summary["figures"] == {}
    assert summary["model_path"].exists()


def test_build_metrics_table_sorted() -> None:
    results = {
        "a": {"cv_mean": 0.7, "cv_std": 0.01, "metrics": {m: 0.7 for m in METRIC_NAMES}},
        "b": {"cv_mean": 0.9, "cv_std": 0.02, "metrics": {m: 0.9 for m in METRIC_NAMES}},
    }
    table = build_metrics_table(results)
    assert list(table.index) == ["b", "a"]  # sorted by cv_mean desc
    assert "f1" in table.columns
