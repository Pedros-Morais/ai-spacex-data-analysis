"""Testes do registro de modelos candidatos."""

from __future__ import annotations

from sklearn.base import is_classifier

from launch_success.config import Settings
from launch_success.models.registry import get_model_registry


def test_registry_tem_os_tres_modelos_base() -> None:
    registry = get_model_registry(Settings())
    # Garante >= 3 modelos (XGBoost é opcional e pode ou não estar presente).
    assert {"logistic_regression", "random_forest", "gradient_boosting"} <= set(registry)
    assert len(registry) >= 3


def test_registry_estimadores_sao_classificadores() -> None:
    registry = get_model_registry(Settings(seed=1))
    for estimator in registry.values():
        assert is_classifier(estimator)


def test_class_weight_balanced_onde_suportado() -> None:
    registry = get_model_registry(Settings())
    assert registry["logistic_regression"].class_weight == "balanced"
    assert registry["random_forest"].class_weight == "balanced"
