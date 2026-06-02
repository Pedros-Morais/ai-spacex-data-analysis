"""Registro de modelos candidatos.

Fornece um dicionário ``nome -> estimador`` com hiperparâmetros vindos da
configuração e ``random_state`` fixo. ``class_weight="balanced"`` é usado nos
modelos que o suportam (regressão logística e random forest) para mitigar o
desbalanceamento de classes.

XGBoost é **opcional**: só entra no registro se a biblioteca importar e carregar
corretamente (depende do runtime OpenMP), mantendo o projeto funcional sem ela.
"""

from __future__ import annotations

import logging

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..config import SETTINGS, Settings

logger = logging.getLogger(__name__)


def _xgboost_classifier(seed: int) -> BaseEstimator | None:
    """Retorna um ``XGBClassifier`` se o XGBoost estiver disponível, senão ``None``."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # noqa: BLE001 - ImportError ou XGBoostError (libomp)
        logger.warning("XGBoost indisponível, será ignorado: %s", exc)
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
    """Constrói o dicionário de estimadores candidatos.

    Args:
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        Mapa ``nome -> estimador`` não-ajustado. Inclui XGBoost quando
        disponível, garantindo sempre >= 3 modelos.
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

    logger.info("Modelos candidatos: %s", ", ".join(registry))
    return registry
