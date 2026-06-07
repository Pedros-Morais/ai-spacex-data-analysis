"""End-to-end ML pipeline orchestration.

Chains: loading -> cleaning -> feature engineering -> EDA -> stratified split
-> training all models -> comparison -> best-model selection -> final ``fit``
-> persistence -> evaluation plots -> SHAP analysis.

Can be run via ``python -m launch_success.pipeline`` or through the script
``scripts/run_training.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from .config import SETTINGS, Settings
from .data.loader import load_dataset
from .evaluation import plots
from .evaluation.metrics import METRIC_NAMES
from .features.cleaning import clean_launches
from .features.engineering import add_derived_features, split_features_target
from .interpretability.shap_analysis import run_shap_analysis
from .models.persistence import save_model
from .models.trainer import select_best_model, stratified_split, train_all_models

logger = logging.getLogger(__name__)


def build_metrics_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the comparative metrics table (one row per model).

    Args:
        results: Output of :func:`~launch_success.models.trainer.train_all_models`.

    Returns:
        DataFrame sorted by CV mean (descending).
    """
    rows = []
    for name, result in results.items():
        row = {"model": name, "cv_mean": result["cv_mean"], "cv_std": result["cv_std"]}
        row.update({metric: result["metrics"][metric] for metric in METRIC_NAMES})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    return table.sort_values("cv_mean", ascending=False)


def _generate_eda_plots(frame: pd.DataFrame, target: str, settings: Settings) -> dict[str, Any]:
    """Generate EDA plots (target distribution and rates by category)."""
    figures: dict[str, Any] = {
        "target_distribution": plots.plot_target_distribution(frame, target, settings),
    }
    for column in ("orbit", "rocket", "reused"):
        if column in frame.columns:
            figures[f"rate_by_{column}"] = plots.plot_success_rate_by(
                frame, column, target, settings
            )
    return figures


def _generate_evaluation_plots(
    results: dict[str, dict[str, Any]],
    best: dict[str, Any],
    y_test: pd.Series,
    settings: Settings,
) -> dict[str, Any]:
    """Generate confusion matrix, ROC/PR curves, and model comparison chart."""
    return {
        "confusion_matrix": plots.plot_confusion_matrix(
            y_test.to_numpy(), best["y_pred"], settings
        ),
        "roc_curves": plots.plot_roc_curves(results, y_test.to_numpy(), settings),
        "pr_curves": plots.plot_pr_curves(results, y_test.to_numpy(), settings),
        "model_comparison": plots.plot_model_comparison(results, settings),
    }


def run_pipeline(
    settings: Settings | None = None,
    generate_plots: bool = True,
    run_shap: bool = True,
) -> dict[str, Any]:
    """Run the full pipeline and return a summary of results.

    Args:
        settings: Configuration (uses :data:`SETTINGS` if omitted).
        generate_plots: If ``True``, generate EDA and evaluation plots.
        run_shap: If ``True``, run SHAP analysis on the winning model.

    Returns:
        Dictionary containing ``best_name``, ``metrics_table`` (DataFrame),
        ``model_path``, ``figures``, and ``results``.
    """
    settings = settings or SETTINGS
    settings.ensure_directories()
    target = settings.target

    # 1) Loading + cleaning + feature engineering.
    raw = load_dataset(settings=settings)
    clean = clean_launches(raw, settings=settings, target=target)
    enriched = add_derived_features(clean)

    figures: dict[str, Any] = {}
    if generate_plots:
        figures.update(_generate_eda_plots(enriched, target, settings))

    # 2) Stratified split.
    x, y = split_features_target(enriched, settings=settings, target=target)
    x_train, x_test, y_train, y_test = stratified_split(x, y, settings)

    # 3) Training + comparison + selection.
    results = train_all_models(x_train, y_train, x_test, y_test, settings)
    metrics_table = build_metrics_table(results)
    best_name, best = select_best_model(results, settings.selection_metric)

    # 4) Persistence of the winning pipeline + metadata.
    metadata = {
        "model_name": best_name,
        "target": target,
        "selection_metric": settings.selection_metric,
        "cv_mean": best["cv_mean"],
        "cv_std": best["cv_std"],
        "test_metrics": best["metrics"],
        "feature_columns": settings.feature_columns,
        "numeric_features": settings.numeric_features,
        "categorical_features": settings.categorical_features,
        "boolean_features": settings.boolean_features,
    }
    model_path = save_model(best["pipeline"], settings=settings, metadata=metadata)
    settings.metrics_path.write_text(
        json.dumps(
            {
                name: {
                    "cv_mean": r["cv_mean"],
                    "cv_std": r["cv_std"],
                    "metrics": r["metrics"],
                }
                for name, r in results.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 5) Evaluation plots + SHAP.
    if generate_plots:
        figures.update(_generate_evaluation_plots(results, best, y_test, settings))
    if run_shap:
        try:
            figures.update(run_shap_analysis(best["pipeline"], x_test, settings))
        except Exception as exc:  # noqa: BLE001 - SHAP must not break the pipeline
            logger.warning("SHAP analysis failed and was skipped: %s", exc)

    logger.info("Pipeline complete. Best model: %s", best_name)
    return {
        "best_name": best_name,
        "best_result": best,
        "results": results,
        "metrics_table": metrics_table,
        "model_path": model_path,
        "figures": figures,
    }


def main() -> None:  # pragma: no cover - thin CLI wrapper
    """Command-line entry point for training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run_pipeline()
    print("\n=== Model comparison ===")
    print(summary["metrics_table"].round(4).to_string())
    print(f"\nBest model: {summary['best_name']}")
    print(f"Model saved to: {summary['model_path']}")


if __name__ == "__main__":  # pragma: no cover
    main()
