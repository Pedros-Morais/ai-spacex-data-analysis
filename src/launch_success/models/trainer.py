"""Training, cross-validation, and evaluation of candidate models.

Each model is wrapped in a :class:`~sklearn.pipeline.Pipeline` containing the
preprocessor (fitted **only** on the training fold, no leakage) and, optionally,
SMOTE applied solely to the training fold (via the ``imbalanced-learn`` pipeline).

Best-model selection uses the stratified cross-validation metric
(``settings.selection_metric``, default ``f1``), not accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from ..config import SETTINGS, Settings
from ..evaluation.metrics import compute_metrics
from ..features.engineering import build_preprocessor
from .registry import get_model_registry, get_param_distributions

logger = logging.getLogger(__name__)

# Maps the selection metric to the corresponding scikit-learn scorer.
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
    """Splits ``(X, y)`` into train/test sets in a stratified manner.

    Args:
        x: Feature matrix.
        y: Target vector.
        settings: Configuration (uses :data:`SETTINGS` if omitted).

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``.
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
    """Assembles the ``preprocessing [-> SMOTE] -> model`` pipeline.

    Args:
        estimator: Candidate estimator (unfitted).
        settings: Configuration (uses :data:`SETTINGS` if omitted).

    Returns:
        Pipeline ready for ``fit`` (from ``imbalanced-learn`` if SMOTE is active).
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


def _build_cv(y_train: pd.Series, settings: Settings) -> StratifiedKFold:
    """Builds a StratifiedKFold with a fold count viable for small datasets."""
    # Caps folds at the size of the minority class (e.g. tiny test datasets).
    min_class = int(y_train.value_counts().min())
    n_splits = max(2, min(settings.cv_folds, min_class))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=settings.seed)


def cross_validate_score(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    settings: Settings | None = None,
) -> tuple[float, float]:
    """Runs stratified cross-validation and returns the mean and std of the score.

    Args:
        pipeline: Unfitted pipeline.
        x_train: Training features.
        y_train: Training target.
        settings: Configuration (uses :data:`SETTINGS` if omitted).

    Returns:
        Pair ``(mean, std)`` of the selection metric across folds.
    """
    settings = settings or SETTINGS
    scorer = _SCORER_MAP.get(settings.selection_metric, "f1")
    cv = _build_cv(y_train, settings)
    scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring=scorer)
    return float(np.mean(scores)), float(np.std(scores))


def tune_pipeline(
    pipeline: Pipeline,
    param_grid: dict[str, list],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    settings: Settings | None = None,
) -> tuple[Pipeline, dict[str, Any], float, float]:
    """Cross-validated hyperparameter search over a pipeline (no data leakage).

    The search cross-validates **only on the training set**; the preprocessing
    lives inside the pipeline, so each fold is fit independently. With
    ``settings.search_strategy == "grid"`` a full ``GridSearchCV`` is run,
    otherwise a ``RandomizedSearchCV`` samples ``settings.search_n_iter``
    candidates (capped at the grid size).

    Args:
        pipeline: Unfitted pipeline (preprocessor + model).
        param_grid: Search space with ``model__<param>`` keys.
        x_train: Training features.
        y_train: Training target.
        settings: Configuration (uses :data:`SETTINGS` if omitted).

    Returns:
        Tuple ``(best_pipeline, best_params, cv_mean, cv_std)`` where the
        pipeline is already refit on the full training set.
    """
    settings = settings or SETTINGS
    scorer = _SCORER_MAP.get(settings.selection_metric, "f1")
    cv = _build_cv(y_train, settings)

    if settings.search_strategy == "grid":
        search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
            pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1, refit=True
        )
    else:
        total = int(np.prod([len(v) for v in param_grid.values()]))
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=min(settings.search_n_iter, total),
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            random_state=settings.seed,
            refit=True,
        )

    search.fit(x_train, y_train)
    index = search.best_index_
    cv_mean = float(search.cv_results_["mean_test_score"][index])
    cv_std = float(search.cv_results_["std_test_score"][index])
    return search.best_estimator_, search.best_params_, cv_mean, cv_std


def train_and_evaluate(
    name: str,
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    settings: Settings | None = None,
    param_grid: dict[str, list] | None = None,
) -> dict[str, Any]:
    """Trains, cross-validates, and evaluates a single model.

    When ``settings.tune_hyperparameters`` is ``True`` and a ``param_grid`` is
    provided, a cross-validated search selects the hyperparameters before the
    final fit; otherwise the model is trained with its default hyperparameters.

    Args:
        name: Model name (registry key).
        estimator: Candidate estimator.
        x_train: Training features.
        y_train: Training target.
        x_test: Test features.
        y_test: Test target.
        settings: Configuration (uses :data:`SETTINGS` if omitted).
        param_grid: Optional search space (``model__<param>`` keys) for tuning.

    Returns:
        Dictionary with the fitted pipeline, CV scores, test metrics, and the
        selected ``best_params`` (``None`` when tuning is disabled).
    """
    settings = settings or SETTINGS
    pipeline = build_pipeline(estimator, settings)
    best_params: dict[str, Any] | None = None

    if settings.tune_hyperparameters and param_grid:
        # Search refits the best pipeline on the full training set.
        pipeline, best_params, cv_mean, cv_std = tune_pipeline(
            pipeline, param_grid, x_train, y_train, settings
        )
    else:
        cv_mean, cv_std = cross_validate_score(pipeline, x_train, y_train, settings)
        pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    logger.info(
        "%s | CV %s=%.3f (+/-%.3f) | test f1=%.3f roc_auc=%.3f%s",
        name,
        settings.selection_metric,
        cv_mean,
        cv_std,
        metrics["f1"],
        metrics["roc_auc"],
        " [tuned]" if best_params else "",
    )
    return {
        "name": name,
        "pipeline": pipeline,
        "cv_metric": settings.selection_metric,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "metrics": metrics,
        "best_params": best_params,
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
    """Trains and evaluates all models in the registry.

    Args:
        x_train: Training features.
        y_train: Training target.
        x_test: Test features.
        y_test: Test target.
        settings: Configuration (uses :data:`SETTINGS` if omitted).
        registry: Model registry (uses :func:`get_model_registry` if omitted).

    Returns:
        Map ``name -> result`` (see :func:`train_and_evaluate`).
    """
    settings = settings or SETTINGS
    registry = registry or get_model_registry(settings)
    grids = get_param_distributions() if settings.tune_hyperparameters else {}
    return {
        name: train_and_evaluate(
            name, estimator, x_train, y_train, x_test, y_test, settings, grids.get(name)
        )
        for name, estimator in registry.items()
    }


def select_best_model(
    results: dict[str, dict[str, Any]],
    metric: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Selects the best model by the mean CV score of the selection metric.

    The choice uses the cross-validation score (on training data), not the test
    set, avoiding selection bias.

    Args:
        results: Output of :func:`train_all_models`.
        metric: Tiebreaker metric displayed (informational).

    Returns:
        Pair ``(name, result)`` of the winning model.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("No model results to select from.")
    best_name = max(results, key=lambda name: results[name]["cv_mean"])
    logger.info(
        "Winning model: %s (CV %s=%.3f)",
        best_name,
        metric or results[best_name]["cv_metric"],
        results[best_name]["cv_mean"],
    )
    return best_name, results[best_name]
