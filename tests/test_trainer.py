"""Testes de treino, validação cruzada e seleção de modelos."""

from __future__ import annotations

import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from launch_success.config import Settings
from launch_success.evaluation.metrics import METRIC_NAMES
from launch_success.features.engineering import split_features_target
from launch_success.models.trainer import (
    build_pipeline,
    select_best_model,
    stratified_split,
    train_all_models,
    train_and_evaluate,
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


def test_stratified_split_proporcoes(xy, fast_settings) -> None:
    x, y = xy
    x_train, x_test, y_train, y_test = stratified_split(x, y, fast_settings)
    assert len(x_train) + len(x_test) == len(x)
    # Estratificação: taxa da classe positiva próxima entre treino e teste.
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_build_pipeline_tipo() -> None:
    estimator = LogisticRegression()
    assert isinstance(build_pipeline(estimator, Settings(use_smote=False)), Pipeline)
    smote_pipe = build_pipeline(estimator, Settings(use_smote=True))
    assert isinstance(smote_pipe, ImbPipeline)
    assert "smote" in dict(smote_pipe.steps)


def test_train_and_evaluate_retorna_metricas(xy, fast_settings) -> None:
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


def test_reprodutibilidade_com_seed(xy, fast_settings) -> None:
    x, y = xy
    x_tr, x_te, y_tr, y_te = stratified_split(x, y, fast_settings)
    est = lambda: RandomForestClassifier(n_estimators=25, random_state=42)  # noqa: E731
    r1 = train_and_evaluate("rf", est(), x_tr, y_tr, x_te, y_te, fast_settings)
    r2 = train_and_evaluate("rf", est(), x_tr, y_tr, x_te, y_te, fast_settings)
    assert r1["metrics"]["f1"] == r2["metrics"]["f1"]
    assert r1["cv_mean"] == r2["cv_mean"]


def test_train_all_e_select_best(xy, fast_settings, small_registry) -> None:
    x, y = xy
    x_tr, x_te, y_tr, y_te = stratified_split(x, y, fast_settings)
    results = train_all_models(x_tr, y_tr, x_te, y_te, fast_settings, registry=small_registry)
    assert set(results) == set(small_registry)
    name, best = select_best_model(results, "f1")
    assert name in small_registry
    assert best["cv_mean"] == max(r["cv_mean"] for r in results.values())


def test_select_best_vazio_levanta_erro() -> None:
    with pytest.raises(ValueError, match="Nenhum resultado"):
        select_best_model({})
