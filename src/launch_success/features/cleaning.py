"""Limpeza do dataset cru: filtragem, coerção de tipos e tratamento de nulos.

Regras (documentadas no enunciado):

* descartar lançamentos futuros (``upcoming == True``);
* descartar linhas cujo **alvo** é nulo (lançamentos sem desfecho conhecido);
* coagir booleanos para ``{0.0, 1.0, NaN}`` e numéricos para ``float``;
* manter categóricas como texto (nulos tratados depois pelo imputer).

As funções operam sobre uma **cópia** do DataFrame (sem efeitos colaterais).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import SETTINGS, Settings
from ..exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Valores textuais interpretados como verdadeiro/falso ao ler CSVs.
_TRUE_TOKENS = frozenset({"true", "1", "1.0", "yes", "t"})
_FALSE_TOKENS = frozenset({"false", "0", "0.0", "no", "f"})


def coerce_boolean(series: pd.Series) -> pd.Series:
    """Coage uma série para ``float`` em ``{1.0, 0.0, NaN}``.

    Aceita ``bool``, inteiros, floats e strings (``"True"``/``"False"``),
    preservando ausências como ``NaN`` para imputação posterior.

    Args:
        series: Série a ser coagida.

    Returns:
        Série de ``float`` com valores em ``{1.0, 0.0, NaN}``.
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
    """Converte para texto, mantendo ausências como ``NaN``."""
    return series.map(lambda v: str(v) if pd.notna(v) else np.nan)


def clean_launches(
    frame: pd.DataFrame,
    settings: Settings | None = None,
    target: str | None = None,
) -> pd.DataFrame:
    """Limpa o dataset cru e devolve uma cópia pronta para a engenharia.

    Args:
        frame: DataFrame cru (da ingestão ou do snapshot).
        settings: Configuração (usa :data:`SETTINGS` se omitida).
        target: Alvo a preservar; se omitido, usa ``settings.target``.

    Returns:
        DataFrame limpo: sem ``upcoming``, sem alvo nulo, tipos coagidos.

    Raises:
        DataValidationError: Se o alvo não existir ou se nada restar após a
            limpeza.
    """
    settings = settings or SETTINGS
    target = target or settings.target

    if target not in frame.columns:
        raise DataValidationError(f"Coluna alvo '{target}' ausente no dataset.")

    df = frame.copy()

    # 1) Remove lançamentos futuros, se a coluna existir.
    if "upcoming" in df.columns:
        upcoming = coerce_boolean(df["upcoming"]).fillna(0.0)
        df = df[upcoming == 0.0].copy()

    # 2) Coage o alvo e descarta linhas sem desfecho conhecido.
    df[target] = coerce_boolean(df[target])
    before = len(df)
    df = df[df[target].notna()].copy()
    logger.info("Removidas %d linhas com alvo '%s' nulo", before - len(df), target)
    df[target] = df[target].astype(int)

    # 3) Coage tipos das features.
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
        raise DataValidationError("Nenhuma linha restante após a limpeza.")

    logger.info("Dataset limpo: %d linhas, alvo '%s'", len(df), target)
    return df.reset_index(drop=True)
