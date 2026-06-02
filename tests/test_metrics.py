"""Testes das métricas de avaliação (valores conhecidos e desbalanceamento)."""

from __future__ import annotations

import math

import numpy as np

from launch_success.evaluation.metrics import METRIC_NAMES, compute_metrics


def test_predicao_perfeita_da_um() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.2, 0.8]
    metrics = compute_metrics(y_true, y_pred, y_proba)
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert math.isclose(metrics[name], 1.0)


def test_todas_as_chaves_presentes() -> None:
    metrics = compute_metrics([0, 1], [0, 1], [0.2, 0.8])
    assert set(metrics) == set(METRIC_NAMES)


def test_classificador_ingenuo_sob_desbalanceamento() -> None:
    # 95% sucesso; modelo "chuta sempre sucesso" -> accuracy alta mas recall=0 na falha.
    y_true = np.array([1] * 95 + [0] * 5)
    y_pred = np.ones(100, dtype=int)  # sempre prevê sucesso
    metrics = compute_metrics(y_true, y_pred, y_pred.astype(float))
    assert metrics["accuracy"] == 0.95  # alta e enganosa
    assert metrics["recall"] == 1.0  # recall da classe positiva
    # F1 da classe positiva alto, mas o modelo é inútil para detectar falhas.
    assert metrics["precision"] == 0.95


def test_classe_unica_roc_auc_nan() -> None:
    # Apenas uma classe presente -> ROC-AUC/PR-AUC indefinidos (NaN).
    metrics = compute_metrics([1, 1, 1], [1, 1, 1], [0.9, 0.8, 0.7])
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["pr_auc"])


def test_sem_proba_auc_nan() -> None:
    metrics = compute_metrics([0, 1], [0, 1], None)
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["pr_auc"])
