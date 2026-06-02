#!/usr/bin/env python
"""Busca os dados da API v4 da SpaceX e grava o CSV processado.

Caminho canônico da entrega: ingestão real via API. Se a API estiver
indisponível (offline/ambiente de correção), cai para o gerador sintético
versionado, garantindo que o pipeline rode sem travar.

Uso:
    python scripts/run_ingestion.py
"""

from __future__ import annotations

import logging

from launch_success.config import SETTINGS
from launch_success.data.ingestion import ingest
from launch_success.data.synthetic import write_synthetic_dataset
from launch_success.exceptions import IngestionError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_ingestion")


def main() -> None:
    """Tenta a ingestão real; em falha, gera o dataset de fallback."""
    try:
        frame = ingest(settings=SETTINGS)
        print(f"Ingestão real concluída: {len(frame)} lançamentos -> {SETTINGS.processed_csv}")
    except IngestionError as exc:
        logger.warning("API indisponível (%s). Gerando dataset sintético de fallback.", exc)
        frame = write_synthetic_dataset(settings=SETTINGS)
        print(f"Fallback sintético gerado: {len(frame)} linhas -> {SETTINGS.processed_csv}")


if __name__ == "__main__":
    main()
