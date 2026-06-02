"""Interpretabilidade dos modelos com SHAP.

A explicação ocorre no **espaço transformado** (pós ``ColumnTransformer``): o
pré-processador converte ``X`` e fornece os nomes das features codificadas; o
explainer é escolhido conforme o tipo de modelo:

* árvores (RandomForest, GradientBoosting, XGBoost) -> :class:`shap.TreeExplainer`;
* regressão logística (linear) -> :class:`shap.LinearExplainer`;
* fallback genérico -> :class:`shap.KernelExplainer`.

São gerados e salvos: *summary plot* (beeswarm), *bar plot* (importância global)
e *waterfall* (explicação de uma previsão individual).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — antes de importar pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from ..config import SETTINGS, Settings  # noqa: E402

logger = logging.getLogger(__name__)

# Modelos lineares usam LinearExplainer; árvores usam TreeExplainer.
_LINEAR_MODELS = {"LogisticRegression"}
_TREE_MODELS = {
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
}


def _densify(matrix: Any) -> np.ndarray:
    """Converte uma matriz esparsa em densa, se necessário."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _normalize_explanation(explanation: shap.Explanation) -> shap.Explanation:
    """Reduz a explicação para a classe positiva quando há eixo de classes.

    Alguns explainers retornam valores com shape ``(n, n_features, n_classes)``;
    selecionamos a classe positiva (índice 1) para obter ``(n, n_features)``.

    Args:
        explanation: Objeto :class:`shap.Explanation` cru.

    Returns:
        Explicação 2D para a classe positiva.
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
    """Calcula os valores SHAP de um pipeline ajustado.

    Args:
        pipeline: Pipeline treinado (pré-processador + modelo).
        x: Amostra de features (crua, pré-transformação).
        settings: Configuração (usa :data:`SETTINGS` se omitida).
        max_samples: Limite de linhas para acelerar o cálculo/plotagem.

    Returns:
        :class:`shap.Explanation` 2D (classe positiva) com nomes de features.
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
    else:  # fallback genérico (lento, mas robusto)
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
    logger.info("SHAP calculado para %d amostras (%s)", len(x_trans), model_name)
    return _normalize_explanation(explanation)


def _save_current_figure(path: Path) -> Path:
    """Salva a figura atual do matplotlib e a fecha."""
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
    """Salva o *summary plot* (beeswarm) da importância das features."""
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
    """Salva o *bar plot* da importância global média (|SHAP|)."""
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
    """Salva o *waterfall* explicando uma previsão individual."""
    settings = settings or SETTINGS
    plt.figure()
    shap.plots.waterfall(explanation[index], show=False)
    return _save_current_figure(settings.figures_dir / filename)


def run_shap_analysis(
    pipeline: Pipeline,
    x: pd.DataFrame,
    settings: Settings | None = None,
) -> dict[str, Path]:
    """Executa a análise SHAP completa e salva os três gráficos.

    Args:
        pipeline: Pipeline treinado.
        x: Amostra de features para explicar.
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        Mapa ``{nome_do_gráfico: caminho}``.
    """
    settings = settings or SETTINGS
    explanation = compute_shap_explanation(pipeline, x, settings)
    return {
        "summary": save_summary_plot(explanation, settings),
        "bar": save_bar_plot(explanation, settings),
        "waterfall": save_waterfall_plot(explanation, 0, settings),
    }
