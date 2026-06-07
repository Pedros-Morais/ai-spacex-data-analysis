"""Tests for SHAP analysis (tree and linear explainers + persistence)."""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from launch_success.config import Settings
from launch_success.features.engineering import split_features_target
from launch_success.interpretability.shap_analysis import (
    compute_shap_explanation,
    run_shap_analysis,
)
from launch_success.models.trainer import build_pipeline, stratified_split


def _fit(estimator, clean_frame):
    settings = Settings(cv_folds=3)
    x, y = split_features_target(clean_frame, settings=settings, target="success")
    x_tr, x_te, y_tr, _ = stratified_split(x, y, settings)
    pipeline = build_pipeline(estimator, settings)
    pipeline.fit(x_tr, y_tr)
    return pipeline, x_te


@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestClassifier(n_estimators=20, random_state=42),  # TreeExplainer
        LogisticRegression(max_iter=500, random_state=42),  # LinearExplainer
    ],
)
def test_compute_shap_explanation_shape(estimator, clean_frame) -> None:
    pipeline, x_te = _fit(estimator, clean_frame)
    explanation = compute_shap_explanation(pipeline, x_te, max_samples=30)
    # 2D: (n_samples, n_encoded_features).
    assert explanation.values.ndim == 2
    assert explanation.values.shape[0] == min(30, len(x_te))
    assert len(explanation.feature_names) == explanation.values.shape[1]


def test_run_shap_analysis_saves_figures(clean_frame, tmp_settings) -> None:
    pipeline, x_te = _fit(RandomForestClassifier(n_estimators=20, random_state=42), clean_frame)
    paths = run_shap_analysis(pipeline, x_te, tmp_settings)
    assert set(paths) == {"summary", "bar", "waterfall"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
