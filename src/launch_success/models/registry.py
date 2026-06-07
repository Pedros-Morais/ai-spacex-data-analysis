"""Candidate model registry.

Provides a dictionary ``name -> estimator`` with hyperparameters sourced from
the configuration and a fixed ``random_state``. ``class_weight="balanced"`` is
used for models that support it (logistic regression and random forest) to
mitigate class imbalance.

XGBoost is **optional**: it is only added to the registry if the library imports
and loads correctly (depends on the OpenMP runtime), keeping the project
functional without it.
"""

from __future__ import annotations

import logging

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..config import SETTINGS, Settings

logger = logging.getLogger(__name__)


def _xgboost_classifier(seed: int) -> BaseEstimator | None:
    """Returns an ``XGBClassifier`` if XGBoost is available, otherwise ``None``."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # noqa: BLE001 - ImportError ou XGBoostError (libomp)
        logger.warning("XGBoost unavailable, skipping: %s", exc)
        return None

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def get_model_registry(settings: Settings | None = None) -> dict[str, BaseEstimator]:
    """Builds the dictionary of candidate estimators.

    Args:
        settings: Configuration (uses :data:`SETTINGS` if omitted).

    Returns:
        Map ``name -> unfitted estimator``. Includes XGBoost when available,
        always guaranteeing >= 3 models.
    """
    settings = settings or SETTINGS
    seed = settings.seed

    registry: dict[str, BaseEstimator] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=seed,
        ),
    }

    xgb = _xgboost_classifier(seed)
    if xgb is not None:
        registry["xgboost"] = xgb

    logger.info("Candidate models: %s", ", ".join(registry))
    return registry


# Search spaces for hyperparameter tuning. Keys are model names; values map
# pipeline-prefixed parameters (``model__<param>``, since the estimator is the
# ``model`` step of the pipeline) to the candidate values to search over.
_PARAM_DISTRIBUTIONS: dict[str, dict[str, list]] = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
    },
    "random_forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [2, 3],
    },
    "xgboost": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [3, 4, 6],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
    },
}


def get_param_distributions() -> dict[str, dict[str, list]]:
    """Returns the hyperparameter search space for each candidate model.

    Returns:
        Map ``model_name -> {``model__param``: [values]}`` for use with
        ``GridSearchCV`` / ``RandomizedSearchCV``.
    """
    return {name: dict(grid) for name, grid in _PARAM_DISTRIBUTIONS.items()}
