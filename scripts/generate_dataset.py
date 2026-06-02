#!/usr/bin/env python
"""Gera o dataset sintético de fallback versionado (>= 1.000 linhas).

Use quando a API estiver indisponível ou para regenerar o snapshot de exemplo
que acompanha o repositório.

Uso:
    python scripts/generate_dataset.py [n_linhas]
"""

from __future__ import annotations

import logging
import sys

from launch_success.config import SETTINGS
from launch_success.data.synthetic import write_synthetic_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    """Gera o CSV sintético; aceita o número de linhas como argumento."""
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    frame = write_synthetic_dataset(n_rows=n_rows, settings=SETTINGS)
    print(f"Dataset sintético gerado: {len(frame)} linhas -> {SETTINGS.processed_csv}")
    print(f"Taxa de sucesso de lançamento: {frame['success'].mean():.1%}")
    landings = frame["landing_success"].dropna()
    if not landings.empty:
        print(f"Taxa de sucesso de pouso (entre tentativas): {landings.mean():.1%}")


if __name__ == "__main__":
    main()
