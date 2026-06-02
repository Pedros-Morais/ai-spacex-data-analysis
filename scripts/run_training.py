#!/usr/bin/env python
"""Roda o pipeline de treino completo (treina, compara, escolhe, salva, SHAP).

Uso:
    python scripts/run_training.py
"""

from __future__ import annotations

import logging

from launch_success.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    """Executa o pipeline e imprime a tabela comparativa de modelos."""
    summary = run_pipeline()
    print("\n=== Comparação de modelos (ordenado por CV) ===")
    print(summary["metrics_table"].round(4).to_string())
    print(f"\nMelhor modelo: {summary['best_name']}")
    print(f"Modelo salvo em: {summary['model_path']}")
    print(f"Figuras geradas: {len(summary['figures'])} em reports/figures/")


if __name__ == "__main__":
    main()
