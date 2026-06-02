"""Engenharia de atributos e construção do pré-processador (sem leakage).

O :class:`~sklearn.compose.ColumnTransformer` é devolvido **não-ajustado**; o
``fit`` ocorre apenas dentro do :class:`~sklearn.pipeline.Pipeline` no fold de
treino (ver :mod:`launch_success.models.trainer`), evitando vazamento de dados.
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
    """Deriva atributos a partir das colunas cruas.

    Atualmente garante a coluna ``year`` (extraída de ``date_utc`` quando
    ausente). Mantida como função pura e extensível para novas derivações.

    Args:
        frame: DataFrame limpo.

    Returns:
        Cópia do DataFrame com os atributos derivados.
    """
    df = frame.copy()
    if "date_utc" in df.columns:
        # format="ISO8601" lida com precisões distintas (com/sem milissegundos).
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
    """Separa a matriz de features ``X`` do vetor alvo ``y``.

    Args:
        frame: DataFrame limpo e enriquecido.
        settings: Configuração (usa :data:`SETTINGS` se omitida).
        target: Alvo; se omitido, usa ``settings.target``.

    Returns:
        Par ``(X, y)`` com ``X`` contendo apenas ``settings.feature_columns``.
    """
    settings = settings or SETTINGS
    target = target or settings.target
    features = settings.feature_columns
    x = frame[features].copy()
    y = frame[target].astype(int)
    return x, y


def build_preprocessor(settings: Settings | None = None) -> ColumnTransformer:
    """Constrói o ``ColumnTransformer`` de pré-processamento (não-ajustado).

    * Numéricas: imputação por mediana + padronização (``StandardScaler``).
    * Categóricas: imputação por constante + ``OneHotEncoder(handle_unknown="ignore")``.
    * Booleanas: imputação pela moda (já em escala 0/1, sem padronização).

    Args:
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        ``ColumnTransformer`` pronto para entrar em um ``Pipeline``.
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
