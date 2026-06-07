"""Raw dataset cleaning: filtering, type coercion, and null handling.

Rules (as documented in the specification):

* discard future launches (``upcoming == True``);
* discard rows whose **target** is null (launches with no known outcome);
* coerce booleans to ``{0.0, 1.0, NaN}`` and numerics to ``float``;
* keep categoricals as text (nulls handled later by the imputer).

Functions operate on a **copy** of the DataFrame (no side effects).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import SETTINGS, Settings
from ..exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Text values interpreted as true/false when reading CSVs.
_TRUE_TOKENS = frozenset({"true", "1", "1.0", "yes", "t"})
_FALSE_TOKENS = frozenset({"false", "0", "0.0", "no", "f"})


def coerce_boolean(series: pd.Series) -> pd.Series:
    """Coerces a series to ``float`` in ``{1.0, 0.0, NaN}``.

    Accepts ``bool``, integers, floats, and strings (``"True"``/``"False"``),
    preserving missing values as ``NaN`` for later imputation.

    Args:
        series: Series to coerce.

    Returns:
        Float series with values in ``{1.0, 0.0, NaN}``.
    """

    def _map(value: object) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if isinstance(value, bool | np.bool_):
            return float(value)
        if isinstance(value, int | float | np.integer | np.floating):
            return float(value)
        token = str(value).strip().lower()
        if token in _TRUE_TOKENS:
            return 1.0
        if token in _FALSE_TOKENS:
            return 0.0
        return np.nan

    return series.map(_map).astype(float)


def _coerce_categorical(series: pd.Series) -> pd.Series:
    """Converts to text, keeping missing values as ``NaN``."""
    return series.map(lambda v: str(v) if pd.notna(v) else np.nan)


def clean_launches(
    frame: pd.DataFrame,
    settings: Settings | None = None,
    target: str | None = None,
) -> pd.DataFrame:
    """Cleans the raw dataset and returns a copy ready for feature engineering.

    Args:
        frame: Raw DataFrame (from ingestion or snapshot).
        settings: Configuration (uses :data:`SETTINGS` if omitted).
        target: Target column to preserve; if omitted, uses ``settings.target``.

    Returns:
        Cleaned DataFrame: no ``upcoming``, no null target, coerced types.

    Raises:
        DataValidationError: If the target column does not exist or if no rows
            remain after cleaning.
    """
    settings = settings or SETTINGS
    target = target or settings.target

    if target not in frame.columns:
        raise DataValidationError(f"Target column '{target}' not found in the dataset.")

    df = frame.copy()

    # 1) Remove future launches, if the column exists.
    if "upcoming" in df.columns:
        upcoming = coerce_boolean(df["upcoming"]).fillna(0.0)
        df = df[upcoming == 0.0].copy()

    # 2) Coerce the target and drop rows with no known outcome.
    df[target] = coerce_boolean(df[target])
    before = len(df)
    df = df[df[target].notna()].copy()
    logger.info("Removed %d rows with null target '%s'", before - len(df), target)
    df[target] = df[target].astype(int)

    # 3) Coerce feature types.
    for column in settings.numeric_features:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in settings.boolean_features:
        if column in df.columns:
            df[column] = coerce_boolean(df[column])
    for column in settings.categorical_features:
        if column in df.columns:
            df[column] = _coerce_categorical(df[column])

    if df.empty:
        raise DataValidationError("No rows left after cleaning.")

    logger.info("Cleaned dataset: %d rows, target '%s'", len(df), target)
    return df.reset_index(drop=True)
