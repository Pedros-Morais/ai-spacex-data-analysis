"""Treino, validação cruzada e avaliação dos modelos candidatos.

Cada modelo é embrulhado em um :class:`~sklearn.pipeline.Pipeline` que contém o
pré-processamento (ajustado **apenas** no treino, sem leakage) e, opcionalmente,
SMOTE aplicado somente ao fold de treino (via pipeline do ``imbalanced-learn``).

A seleção do melhor modelo usa a métrica de validação cruzada estratificada
(``settings.selection_metric``, padrão ``f1``), não a acurácia.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ..config import SETTINGS, Settings
from ..evaluation.metrics import compute_metrics
from ..features.engineering import build_preprocessor
from .registry import get_model_registry

logger = logging.getLogger(__name__)

# Mapeia a métrica de seleção para o scorer correspondente do scikit-learn.
_SCORER_MAP: dict[str, str] = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def stratified_split(
    x: pd.DataFrame,
    y: pd.Series,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide ``(X, y)`` em treino/teste de forma estratificada.

    Args:
        x: Matriz de features.
        y: Vetor alvo.
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        Tupla ``(X_train, X_test, y_train, y_test)``.
    """
    settings = settings or SETTINGS
    return train_test_split(
        x,
        y,
        test_size=settings.test_size,
        stratify=y,
        random_state=settings.seed,
    )


def build_pipeline(estimator: BaseEstimator, settings: Settings | None = None) -> Pipeline:
    """Monta o pipeline ``pré-processamento [-> SMOTE] -> modelo``.

    Args:
        estimator: Estimador candidato (não-ajustado).
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        Pipeline pronto para ``fit`` (do ``imbalanced-learn`` se SMOTE ativo).
    """
    settings = settings or SETTINGS
    preprocessor = build_preprocessor(settings)
    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]

    if settings.use_smote:
        steps.append(("smote", SMOTE(random_state=settings.seed)))
        steps.append(("model", estimator))
        return ImbPipeline(steps=steps)

    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def cross_validate_score(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    settings: Settings | None = None,
) -> tuple[float, float]:
    """Roda validação cruzada estratificada e retorna média e desvio do score.

    Args:
        pipeline: Pipeline não-ajustado.
        x_train: Features de treino.
        y_train: Alvo de treino.
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        Par ``(média, desvio)`` da métrica de seleção nos folds.
    """
    settings = settings or SETTINGS
    scorer = _SCORER_MAP.get(settings.selection_metric, "f1")
    # Garante folds viáveis mesmo em datasets pequenos (ex.: testes).
    min_class = int(y_train.value_counts().min())
    n_splits = max(2, min(settings.cv_folds, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=settings.seed)
    scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring=scorer)
    return float(np.mean(scores)), float(np.std(scores))


def train_and_evaluate(
    name: str,
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Treina, valida (CV) e avalia (teste) um único modelo.

    Args:
        name: Nome do modelo (chave do registro).
        estimator: Estimador candidato.
        x_train: Features de treino.
        y_train: Alvo de treino.
        x_test: Features de teste.
        y_test: Alvo de teste.
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        Dicionário com pipeline ajustado, scores de CV e métricas de teste.
    """
    settings = settings or SETTINGS
    pipeline = build_pipeline(estimator, settings)

    cv_mean, cv_std = cross_validate_score(pipeline, x_train, y_train, settings)
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    logger.info(
        "%s | CV %s=%.3f (+/-%.3f) | teste f1=%.3f roc_auc=%.3f",
        name,
        settings.selection_metric,
        cv_mean,
        cv_std,
        metrics["f1"],
        metrics["roc_auc"],
    )
    return {
        "name": name,
        "pipeline": pipeline,
        "cv_metric": settings.selection_metric,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "metrics": metrics,
        "y_pred": np.asarray(y_pred),
        "y_proba": np.asarray(y_proba),
    }


def train_all_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    settings: Settings | None = None,
    registry: dict[str, BaseEstimator] | None = None,
) -> dict[str, dict[str, Any]]:
    """Treina e avalia todos os modelos do registro.

    Args:
        x_train: Features de treino.
        y_train: Alvo de treino.
        x_test: Features de teste.
        y_test: Alvo de teste.
        settings: Configuração (usa :data:`SETTINGS` se omitida).
        registry: Registro de modelos (usa :func:`get_model_registry` se omitido).

    Returns:
        Mapa ``nome -> resultado`` (ver :func:`train_and_evaluate`).
    """
    settings = settings or SETTINGS
    registry = registry or get_model_registry(settings)
    return {
        name: train_and_evaluate(name, estimator, x_train, y_train, x_test, y_test, settings)
        for name, estimator in registry.items()
    }


def select_best_model(
    results: dict[str, dict[str, Any]],
    metric: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Seleciona o melhor modelo pela média de CV da métrica de seleção.

    A escolha usa o score de validação cruzada (no treino), não o conjunto de
    teste, evitando viés de seleção.

    Args:
        results: Saída de :func:`train_all_models`.
        metric: Métrica de desempate exibida (informativa).

    Returns:
        Par ``(nome, resultado)`` do modelo vencedor.

    Raises:
        ValueError: Se ``results`` estiver vazio.
    """
    if not results:
        raise ValueError("Nenhum resultado de modelo para selecionar.")
    best_name = max(results, key=lambda name: results[name]["cv_mean"])
    logger.info(
        "Modelo vencedor: %s (CV %s=%.3f)",
        best_name,
        metric or results[best_name]["cv_metric"],
        results[best_name]["cv_mean"],
    )
    return best_name, results[best_name]
