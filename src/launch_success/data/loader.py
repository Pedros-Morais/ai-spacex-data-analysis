"""Carregamento do dataset processado (snapshot real da API ou fallback).

O CSV em ``data/processed/spacex_launches.csv`` pode vir tanto da ingestão real
(:mod:`launch_success.data.ingestion`) quanto do gerador sintético versionado
(:mod:`launch_success.data.synthetic`). O loader é agnóstico à origem.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import SETTINGS, Settings
from ..exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Colunas mínimas que qualquer dataset válido precisa expor.
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "flight_number",
        "rocket",
        "payload_mass_kg",
        "orbit",
        "launch_site",
        "reused",
        "flights",
        "gridfins",
        "legs",
        "landing_success",
        "success",
    }
)


def load_dataset(
    path: str | Path | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Carrega o dataset processado de um CSV.

    Args:
        path: Caminho do CSV. Se omitido, usa ``settings.processed_csv``.
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        DataFrame cru, antes da limpeza.

    Raises:
        DataValidationError: Se o arquivo não existir, estiver vazio ou faltarem
            colunas obrigatórias.
    """
    settings = settings or SETTINGS
    csv_path = Path(path) if path is not None else settings.processed_csv

    if not csv_path.exists():
        raise DataValidationError(
            f"Dataset não encontrado em {csv_path}. "
            "Rode a ingestão (`make ingest`) ou gere o fallback "
            "(`python scripts/generate_dataset.py`)."
        )

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise DataValidationError(f"Dataset em {csv_path} está vazio.")

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DataValidationError(
            f"Dataset em {csv_path} não tem as colunas obrigatórias: {sorted(missing)}"
        )

    logger.info("Dataset carregado de %s (%d linhas)", csv_path, len(frame))
    return frame
