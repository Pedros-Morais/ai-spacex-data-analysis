"""Feature engineering and preprocessor construction (no leakage).

The :class:`~sklearn.compose.ColumnTransformer` is returned **unfitted**; the
``fit`` happens only inside the :class:`~sklearn.pipeline.Pipeline` on the
training fold (see :mod:`launch_success.models.trainer`), preventing data leakage.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import SETTINGS, Settings

logger = logging.getLogger(__name__)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derives features from raw columns.

    Currently ensures the ``year`` column exists (extracted from ``date_utc``
    when absent). Kept as a pure, extensible function for future derivations.

    Args:
        frame: Cleaned DataFrame.

    Returns:
        Copy of the DataFrame with derived features added.
    """
    df = frame.copy()
    if "date_utc" in df.columns:
        # format="ISO8601" handles varying precisions (with/without milliseconds).
        year_from_date = pd.to_datetime(
            df["date_utc"], errors="coerce", utc=True, format="ISO8601"
        ).dt.year
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(year_from_date)
        else:
            df["year"] = year_from_date
    return df


def split_features_target(
    frame: pd.DataFrame,
    settings: Settings | None = None,
    target: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separates the feature matrix ``X`` from the target vector ``y``.

    Args:
        frame: Cleaned and enriched DataFrame.
        settings: Configuration (uses :data:`SETTINGS` if omitted).
        target: Target column; if omitted, uses ``settings.target``.

    Returns:
        Pair ``(X, y)`` where ``X`` contains only ``settings.feature_columns``.
    """
    settings = settings or SETTINGS
    target = target or settings.target
    features = settings.feature_columns
    x = frame[features].copy()
    y = frame[target].astype(int)
    return x, y


def build_preprocessor(settings: Settings | None = None) -> ColumnTransformer:
    """Builds the preprocessing ``ColumnTransformer`` (unfitted).

    * Numeric: median imputation + standardisation (``StandardScaler``).
    * Categorical: constant imputation + ``OneHotEncoder(handle_unknown="ignore")``.
    * Boolean: mode imputation (already on 0/1 scale, no standardisation needed).

    Args:
        settings: Configuration (uses :data:`SETTINGS` if omitted).

    Returns:
        ``ColumnTransformer`` ready to be included in a ``Pipeline``.
    """
    settings = settings or SETTINGS

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    boolean = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, settings.numeric_features),
            ("cat", categorical, settings.categorical_features),
            ("bool", boolean, settings.boolean_features),
        ],
        remainder="drop",
    )
