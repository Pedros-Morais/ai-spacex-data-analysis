"""Tests for training, cross-validation, and model selection."""

from __future__ import annotations

import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from launch_success.config import Settings
from launch_success.evaluation.metrics import METRIC_NAMES
from launch_success.features.engineering import split_features_target
from launch_success.models.registry import get_param_distributions
from launch_success.models.trainer import (
    build_pipeline,
    select_best_model,
    stratified_split,
    train_all_models,
    train_and_evaluate,
    tune_pipeline,
)


@pytest.fixture
def fast_settings() -> Settings:
    return Settings(cv_folds=3, seed=42)


@pytest.fixture
def small_registry():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=25, class_weight="balanced", random_state=42
        ),
    }


@pytest.fixture
def xy(clean_frame, fast_settings):
    return split_features_target(clean_frame, settings=fast_settings, target="success")


def test_stratified_split_proportions(xy, fast_settings) -> None:
    x, y = xy
    x_train, x_test, y_train, y_test = stratified_split(x, y, fast_settings)
    assert len(x_train) + len(x_test) == len(x)
    # Stratification: positive-class rate should be close between train and test.
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_build_pipeline_type() -> None:
    estimator = LogisticRegression()
    assert isinstance(build_pipeline(estimator, Settings(use_smote=False)), Pipeline)
    smote_pipe = build_pipeline(estimator, Settings(use_smote=True))
    assert isinstance(smote_pipe, ImbPipeline)
    assert "smote" in dict(smote_pipe.steps)


def test_train_and_evaluate_returns_metrics(xy, fast_settings) -> None:
    x, y = xy
    x_tr, x_te, y_tr, y_te = stratified_split(x, y, fast_settings)
    result = train_and_evaluate(
        "logistic_regression",
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        x_tr,
        y_tr,
        x_te,
        y_te,
        fast_settings,
    )
    assert isinstance(result["pipeline"], Pipeline)
    assert set(result["metrics"]) == set(METRIC_NAMES)
    assert "cv_mean" in result and "cv_std" in result
    assert len(result["y_pred"]) == len(y_te)


def test_reproducibility_with_seed(xy, fast_settings) -> None:
    x, y = xy
    x_tr, x_te, y_tr, y_te = stratified_split(x, y, fast_settings)
    est = lambda: RandomForestClassifier(n_estimators=25, random_state=42)  # noqa: E731
    r1 = train_and_evaluate("rf", est(), x_tr, y_tr, x_te, y_te, fast_settings)
    r2 = train_and_evaluate("rf", est(), x_tr, y_tr, x_te, y_te, fast_settings)
    assert r1["metrics"]["f1"] == r2["metrics"]["f1"]
    assert r1["cv_mean"] == r2["cv_mean"]


def test_train_all_and_select_best(xy, fast_settings, small_registry) -> None:
    x, y = xy
    x_tr, x_te, y_tr, y_te = stratified_split(x, y, fast_settings)
    results = train_all_models(x_tr, y_tr, x_te, y_te, fast_settings, registry=small_registry)
    assert set(results) == set(small_registry)
    name, best = select_best_model(results, "f1")
    assert name in small_registry
    assert best["cv_mean"] == max(r["cv_mean"] for r in results.values())


def test_select_best_empty_raises_error() -> None:
    with pytest.raises(ValueError, match="No model results"):
        select_best_model({})


def test_param_distributions_cover_base_models() -> None:
    grids = get_param_distributions()
    assert {"logistic_regression", "random_forest", "gradient_boosting"} <= set(grids)
    # Keys must be pipeline-prefixed so they reach the estimator step.
    for grid in grids.values():
        assert all(key.startswith("model__") for key in grid)


def test_tune_pipeline_returns_fitted_best(xy) -> None:
    x, y = xy
    settings = Settings(cv_folds=3, search_n_iter=2, seed=42)
    x_tr, _, y_tr, _ = stratified_split(x, y, settings)
    pipeline = build_pipeline(
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=42), settings
    )
    best, params, cv_mean, cv_std = tune_pipeline(
        pipeline, {"model__C": [0.1, 1.0]}, x_tr, y_tr, settings
    )
    assert "model__C" in params
    assert 0.0 <= cv_mean <= 1.0 and cv_std >= 0.0
    # best is already refit on the full training set -> can predict.
    assert len(best.predict(x_tr)) == len(x_tr)


def test_train_and_evaluate_with_tuning(xy) -> None:
    x, y = xy
    settings = Settings(cv_folds=3, tune_hyperparameters=True, search_n_iter=2, seed=42)
    x_tr, x_te, y_tr, y_te = stratified_split(x, y, settings)
    result = train_and_evaluate(
        "logistic_regression",
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        x_tr,
        y_tr,
        x_te,
        y_te,
        settings,
        param_grid={"model__C": [0.1, 1.0]},
    )
    assert result["best_params"] is not None
    assert "model__C" in result["best_params"]
    assert set(result["metrics"]) == set(METRIC_NAMES)


def test_tuning_disabled_leaves_best_params_none(xy, fast_settings) -> None:
    x, y = xy
    x_tr, x_te, y_tr, y_te = stratified_split(x, y, fast_settings)
    result = train_and_evaluate(
        "logistic_regression",
        LogisticRegression(max_iter=500, random_state=42),
        x_tr,
        y_tr,
        x_te,
        y_te,
        fast_settings,
        param_grid={"model__C": [0.1, 1.0]},  # ignored because tuning is off
    )
    assert result["best_params"] is None
