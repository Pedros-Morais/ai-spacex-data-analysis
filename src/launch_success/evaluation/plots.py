"""Geração e persistência de gráficos de EDA e avaliação.

Usa o backend ``Agg`` (não-interativo) para funcionar em ambientes headless
(CI, servidores). Cada função salva um PNG em ``settings.figures_dir`` e
retorna o caminho gerado.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # backend headless — definir antes de importar pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from ..config import SETTINGS, Settings  # noqa: E402
from .metrics import METRIC_NAMES  # noqa: E402

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def _resolve_path(filename: str, settings: Settings) -> Path:
    """Resolve o caminho de saída dentro de ``figures_dir``."""
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    return settings.figures_dir / filename


def plot_target_distribution(
    frame: pd.DataFrame,
    target: str,
    settings: Settings | None = None,
    filename: str = "target_distribution.png",
) -> Path:
    """Plota a distribuição do alvo (evidencia o desbalanceamento)."""
    settings = settings or SETTINGS
    path = _resolve_path(filename, settings)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = frame[target].value_counts().sort_index()
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        ax=ax,
        hue=counts.index.astype(str),
        legend=False,
        palette="viridis",
    )
    ax.set(title=f"Distribuição do alvo '{target}'", xlabel=target, ylabel="contagem")
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_success_rate_by(
    frame: pd.DataFrame,
    by: str,
    target: str,
    settings: Settings | None = None,
    filename: str | None = None,
) -> Path:
    """Plota a taxa média do alvo agrupada por uma feature categórica."""
    settings = settings or SETTINGS
    path = _resolve_path(filename or f"rate_by_{by}.png", settings)
    rates = frame.groupby(by)[target].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=rates.values,
        y=rates.index.astype(str),
        ax=ax,
        hue=rates.index.astype(str),
        legend=False,
        palette="mako",
    )
    ax.set(title=f"Taxa de '{target}' por {by}", xlabel=f"taxa média de {target}", ylabel=by)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    settings: Settings | None = None,
    filename: str = "confusion_matrix.png",
    title: str = "Matriz de confusão",
) -> Path:
    """Plota a matriz de confusão do modelo vencedor."""
    settings = settings or SETTINGS
    path = _resolve_path(filename, settings)
    matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["falha", "sucesso"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curves(
    results: dict[str, dict[str, Any]],
    y_true: np.ndarray,
    settings: Settings | None = None,
    filename: str = "roc_curves.png",
) -> Path:
    """Plota as curvas ROC de todos os modelos em um único gráfico."""
    settings = settings or SETTINGS
    path = _resolve_path(filename, settings)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_true, result["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={result['metrics']['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="aleatório")
    ax.set(title="Curvas ROC", xlabel="FPR", ylabel="TPR")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pr_curves(
    results: dict[str, dict[str, Any]],
    y_true: np.ndarray,
    settings: Settings | None = None,
    filename: str = "pr_curves.png",
) -> Path:
    """Plota as curvas Precision-Recall de todos os modelos."""
    settings = settings or SETTINGS
    path = _resolve_path(filename, settings)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_true, result["y_proba"])
        ax.plot(recall, precision, label=f"{name} (AP={result['metrics']['pr_auc']:.2f})")
    ax.set(title="Curvas Precision-Recall", xlabel="recall", ylabel="precision")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_model_comparison(
    results: dict[str, dict[str, Any]],
    settings: Settings | None = None,
    filename: str = "model_comparison.png",
) -> Path:
    """Plota um comparativo de barras das métricas de teste por modelo."""
    settings = settings or SETTINGS
    path = _resolve_path(filename, settings)
    records = [
        {"model": name, "metric": metric, "value": result["metrics"][metric]}
        for name, result in results.items()
        for metric in METRIC_NAMES
    ]
    data = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=data, x="metric", y="value", hue="model", ax=ax)
    ax.set(title="Comparação de modelos (conjunto de teste)", xlabel="", ylabel="valor")
    ax.set_ylim(0, 1)
    ax.legend(title="modelo", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
