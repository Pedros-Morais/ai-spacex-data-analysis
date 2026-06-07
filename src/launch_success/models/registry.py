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
