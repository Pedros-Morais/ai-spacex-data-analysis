"""Model interpretability with SHAP.

Explanation is performed in the **transformed space** (post ``ColumnTransformer``):
the preprocessor transforms ``X`` and provides encoded feature names; the
explainer is chosen based on the model type:

* trees (RandomForest, GradientBoosting, XGBoost) -> :class:`shap.TreeExplainer`;
* logistic regression (linear) -> :class:`shap.LinearExplainer`;
* generic fallback -> :class:`shap.KernelExplainer`.

Three artefacts are generated and saved: *summary plot* (beeswarm),
*bar plot* (global importance), and *waterfall* (individual prediction explanation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — must be set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from ..config import SETTINGS, Settings  # noqa: E402

logger = logging.getLogger(__name__)

# Linear models use LinearExplainer; tree models use TreeExplainer.
_LINEAR_MODELS = {"LogisticRegression"}
_TREE_MODELS = {
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
}


def _densify(matrix: Any) -> np.ndarray:
    """Convert a sparse matrix to dense, if necessary."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _normalize_explanation(explanation: shap.Explanation) -> shap.Explanation:
    """Reduce the explanation to the positive class when a class axis is present.

    Some explainers return values with shape ``(n, n_features, n_classes)``;
    we select the positive class (index 1) to obtain ``(n, n_features)``.

    Args:
        explanation: Raw :class:`shap.Explanation` object.

    Returns:
        2D explanation for the positive class.
    """
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, 1]
        base = base[:, 1] if base.ndim == 2 else base
    return shap.Explanation(
        values=values,
        base_values=base,
        data=explanation.data,
        feature_names=explanation.feature_names,
    )


def compute_shap_explanation(
    pipeline: Pipeline,
    x: pd.DataFrame,
    settings: Settings | None = None,
    max_samples: int = 300,
) -> shap.Explanation:
    """Compute SHAP values for a fitted pipeline.

    Args:
        pipeline: Trained pipeline (preprocessor + model).
        x: Feature sample (raw, pre-transformation).
        settings: Configuration (uses :data:`SETTINGS` if omitted).
        max_samples: Row limit to speed up computation and plotting.

    Returns:
        2D :class:`shap.Explanation` (positive class) with feature names.
    """
    settings = settings or SETTINGS
    sample = x.iloc[:max_samples].copy()

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    x_trans = _densify(preprocessor.transform(sample))
    feature_names = list(preprocessor.get_feature_names_out())

    model_name = type(model).__name__
    if model_name in _TREE_MODELS:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(x_trans, check_additivity=False)
    elif model_name in _LINEAR_MODELS:
        explainer = shap.LinearExplainer(model, x_trans)
        explanation = explainer(x_trans)
    else:  # generic fallback (slow but robust)
        background = shap.utils.sample(x_trans, min(50, len(x_trans)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        values = explainer.shap_values(x_trans, nsamples=100)
        explanation = shap.Explanation(
            values=np.asarray(values),
            base_values=np.asarray(explainer.expected_value),
            data=x_trans,
            feature_names=feature_names,
        )

    explanation.feature_names = feature_names
    logger.info("SHAP computed for %d samples (%s)", len(x_trans), model_name)
    return _normalize_explanation(explanation)


def _save_current_figure(path: Path) -> Path:
    """Save the current matplotlib figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_summary_plot(
    explanation: shap.Explanation,
    settings: Settings | None = None,
    filename: str = "shap_summary.png",
) -> Path:
    """Save the *summary plot* (beeswarm) of feature importance."""
    settings = settings or SETTINGS
    plt.figure()
    shap.summary_plot(
        explanation.values,
        features=explanation.data,
        feature_names=explanation.feature_names,
        show=False,
    )
    return _save_current_figure(settings.figures_dir / filename)


def save_bar_plot(
    explanation: shap.Explanation,
    settings: Settings | None = None,
    filename: str = "shap_bar.png",
) -> Path:
    """Save the *bar plot* of mean global importance (|SHAP|)."""
    settings = settings or SETTINGS
    plt.figure()
    shap.summary_plot(
        explanation.values,
        features=explanation.data,
        feature_names=explanation.feature_names,
        plot_type="bar",
        show=False,
    )
    return _save_current_figure(settings.figures_dir / filename)


def save_waterfall_plot(
    explanation: shap.Explanation,
    index: int = 0,
    settings: Settings | None = None,
    filename: str = "shap_waterfall.png",
) -> Path:
    """Save the *waterfall* plot explaining an individual prediction."""
    settings = settings or SETTINGS
    plt.figure()
    shap.plots.waterfall(explanation[index], show=False)
    return _save_current_figure(settings.figures_dir / filename)


def run_shap_analysis(
    pipeline: Pipeline,
    x: pd.DataFrame,
    settings: Settings | None = None,
) -> dict[str, Path]:
    """Run the full SHAP analysis and save all three plots.

    Args:
        pipeline: Trained pipeline.
        x: Feature sample to explain.
        settings: Configuration (uses :data:`SETTINGS` if omitted).

    Returns:
        Mapping ``{plot_name: path}``.
    """
    settings = settings or SETTINGS
    explanation = compute_shap_explanation(pipeline, x, settings)
    return {
        "summary": save_summary_plot(explanation, settings),
        "bar": save_bar_plot(explanation, settings),
        "waterfall": save_waterfall_plot(explanation, 0, settings),
    }
