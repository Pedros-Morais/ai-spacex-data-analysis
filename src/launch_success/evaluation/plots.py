"""Generation and persistence of EDA and evaluation plots.

Uses the ``Agg`` (non-interactive) backend so it works in headless environments
(CI, servers). Each function saves a PNG to ``settings.figures_dir`` and
returns the generated path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before importing pyplot

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
    """Resolve the output path inside ``figures_dir``."""
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    return settings.figures_dir / filename


def plot_target_distribution(
    frame: pd.DataFrame,
    target: str,
    settings: Settings | None = None,
    filename: str = "target_distribution.png",
) -> Path:
    """Plot the target distribution (highlights class imbalance)."""
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
    ax.set(title=f"Target distribution '{target}'", xlabel=target, ylabel="count")
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
    """Plot the mean target rate grouped by a categorical feature."""
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
    ax.set(title=f"Rate of '{target}' by {by}", xlabel=f"mean rate of {target}", ylabel=by)
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
    title: str = "Confusion matrix",
) -> Path:
    """Plot the confusion matrix for the winning model."""
    settings = settings or SETTINGS
    path = _resolve_path(filename, settings)
    matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["failure", "success"])
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
    """Plot the ROC curves for all models on a single chart."""
    settings = settings or SETTINGS
    path = _resolve_path(filename, settings)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_true, result["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={result['metrics']['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
    ax.set(title="ROC Curves", xlabel="FPR", ylabel="TPR")
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
    """Plot the Precision-Recall curves for all models."""
    settings = settings or SETTINGS
    path = _resolve_path(filename, settings)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_true, result["y_proba"])
        ax.plot(recall, precision, label=f"{name} (AP={result['metrics']['pr_auc']:.2f})")
    ax.set(title="Precision-Recall Curves", xlabel="recall", ylabel="precision")
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
    """Plot a grouped bar chart comparing test metrics across models."""
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
    ax.set(title="Model comparison (test set)", xlabel="", ylabel="value")
    ax.set_ylim(0, 1)
    ax.legend(title="model", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
