"""Métricas de avaliação para classificação binária desbalanceada.

O ``accuracy`` é reportado por completude, mas a seleção e a leitura dos
resultados priorizam ``f1``, ``recall``, ``precision``, ``roc_auc`` e
``pr_auc`` (average precision) — robustos sob desbalanceamento de classes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ordem canônica das métricas (usada em tabelas e relatórios).
METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
)

ArrayLike = Sequence[float] | np.ndarray


def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: ArrayLike | None = None,
) -> dict[str, float]:
    """Calcula o conjunto de métricas de classificação binária.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_pred: Rótulos previstos (0/1).
        y_proba: Probabilidades da classe positiva (para ROC-AUC / PR-AUC).
            Se ``None``, essas métricas retornam ``NaN``.

    Returns:
        Dicionário ``{nome_da_métrica: valor}`` com as chaves de
        :data:`METRIC_NAMES`. Métricas indefinidas (ex.: ROC-AUC com uma única
        classe presente) retornam ``NaN``.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
    }

    # ROC-AUC e PR-AUC exigem probabilidades e ambas as classes presentes.
    if y_proba is not None and len(np.unique(y_true_arr)) == 2:
        y_proba_arr = np.asarray(y_proba)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba_arr))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_proba_arr))

    return metrics
