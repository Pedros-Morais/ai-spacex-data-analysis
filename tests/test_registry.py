"""Tests for the model candidate registry."""

from __future__ import annotations

from sklearn.base import is_classifier

from launch_success.config import Settings
from launch_success.models.registry import get_model_registry


def test_registry_has_three_base_models() -> None:
    registry = get_model_registry(Settings())
    # Ensures >= 3 models (XGBoost is optional and may or may not be present).
    assert {"logistic_regression", "random_forest", "gradient_boosting"} <= set(registry)
    assert len(registry) >= 3


def test_registry_estimators_are_classifiers() -> None:
    registry = get_model_registry(Settings(seed=1))
    for estimator in registry.values():
        assert is_classifier(estimator)


def test_class_weight_balanced_where_supported() -> None:
    registry = get_model_registry(Settings())
    assert registry["logistic_regression"].class_weight == "balanced"
    assert registry["random_forest"].class_weight == "balanced"
