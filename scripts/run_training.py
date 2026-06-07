#!/usr/bin/env python
"""Runs the full training pipeline (train, compare, select, save, SHAP).

Usage:
    python scripts/run_training.py
"""

from __future__ import annotations

import logging

from launch_success.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    """Runs the pipeline and prints the model comparison table."""
    summary = run_pipeline()
    print("\n=== Model comparison (sorted by CV) ===")
    print(summary["metrics_table"].round(4).to_string())
    print(f"\nBest model: {summary['best_name']}")
    print(f"Model saved to: {summary['model_path']}")
    print(f"Figures generated: {len(summary['figures'])} in reports/figures/")


if __name__ == "__main__":
    main()
