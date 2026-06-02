"""Aplicação Streamlit de inferência do sucesso de lançamentos da SpaceX.

Carrega o pipeline vencedor persistido, recebe os parâmetros do lançamento
(massa do payload, órbita, versão do foguete, booster reutilizado + extras),
exibe a probabilidade de sucesso e a explicação SHAP daquela previsão.

Execução local:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# Garante que o pacote em src/ seja importável quando rodado pelo Streamlit.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from launch_success.config import SETTINGS  # noqa: E402
from launch_success.data.loader import load_dataset  # noqa: E402
from launch_success.exceptions import DataValidationError, ModelNotFoundError  # noqa: E402
from launch_success.interpretability.shap_analysis import (  # noqa: E402
    compute_shap_explanation,
)
from launch_success.models.persistence import load_metadata, load_model  # noqa: E402

# Valores padrão de seleção caso o dataset não esteja disponível.
_DEFAULT_ROCKETS = ["Falcon 1", "Falcon 9", "Falcon Heavy"]
_DEFAULT_ORBITS = ["LEO", "VLEO", "GTO", "ISS", "SSO", "MEO", "PO"]
_DEFAULT_SITES = ["CCSFS SLC 40", "KSC LC 39A", "VAFB SLC 4E", "Kwajalein Atoll"]


@st.cache_resource(show_spinner=False)
def _load_artifacts() -> tuple[object, dict]:
    """Carrega o pipeline treinado e seus metadados (com cache)."""
    pipeline = load_model(settings=SETTINGS)
    metadata = load_metadata(settings=SETTINGS)
    return pipeline, metadata


@st.cache_data(show_spinner=False)
def _category_options() -> dict[str, list[str]]:
    """Lê o dataset para popular as opções dos seletores (com fallback)."""
    try:
        frame = load_dataset(settings=SETTINGS)
    except DataValidationError:
        return {
            "rocket": _DEFAULT_ROCKETS,
            "orbit": _DEFAULT_ORBITS,
            "launch_site": _DEFAULT_SITES,
        }
    return {
        "rocket": sorted(frame["rocket"].dropna().unique().tolist()),
        "orbit": sorted(frame["orbit"].dropna().unique().tolist()),
        "launch_site": sorted(frame["launch_site"].dropna().unique().tolist()),
    }


def _build_input_row(values: dict[str, object]) -> pd.DataFrame:
    """Monta o DataFrame de uma linha com as features esperadas pelo modelo."""
    row = {
        "flight_number": float(values["flight_number"]),
        "year": float(values["year"]),
        "payload_mass_kg": float(values["payload_mass_kg"]),
        "flights": float(values["flights"]),
        "rocket": str(values["rocket"]),
        "orbit": str(values["orbit"]),
        "launch_site": str(values["launch_site"]),
        "reused": 1.0 if values["reused"] else 0.0,
        "gridfins": 1.0 if values["gridfins"] else 0.0,
        "legs": 1.0 if values["legs"] else 0.0,
    }
    return pd.DataFrame([row])[SETTINGS.feature_columns]


def _render_sidebar(options: dict[str, list[str]]) -> dict[str, object]:
    """Renderiza os controles de entrada e retorna os valores escolhidos."""
    st.sidebar.header("Parâmetros do lançamento")
    return {
        "rocket": st.sidebar.selectbox("Versão do foguete", options["rocket"]),
        "orbit": st.sidebar.selectbox("Órbita alvo", options["orbit"]),
        "launch_site": st.sidebar.selectbox("Local de lançamento", options["launch_site"]),
        "payload_mass_kg": st.sidebar.slider("Massa do payload (kg)", 0, 23_000, 5_000, step=100),
        "reused": st.sidebar.checkbox("Booster reutilizado", value=False),
        "flights": st.sidebar.number_input(
            "Voos acumulados do core", min_value=1, max_value=30, value=1
        ),
        "gridfins": st.sidebar.checkbox("Possui grid fins", value=True),
        "legs": st.sidebar.checkbox("Possui pernas de pouso", value=True),
        "flight_number": st.sidebar.number_input(
            "Número do voo", min_value=1, max_value=500, value=200
        ),
        "year": st.sidebar.slider("Ano", 2006, 2030, 2024),
    }


def _render_prediction(pipeline: object, input_row: pd.DataFrame, target: str) -> None:
    """Exibe a probabilidade prevista e o veredito."""
    proba = float(pipeline.predict_proba(input_row)[0, 1])  # type: ignore[attr-defined]
    label = "SUCESSO provável" if proba >= 0.5 else "RISCO de falha"
    col1, col2 = st.columns(2)
    col1.metric(f"Probabilidade de '{target}'", f"{proba:.1%}")
    col2.metric("Veredito", label)
    st.progress(proba)


def _render_shap(pipeline: object, input_row: pd.DataFrame) -> None:
    """Calcula e exibe a explicação SHAP (waterfall) da previsão."""
    st.subheader("Por que esta previsão? (SHAP)")
    try:
        explanation = compute_shap_explanation(pipeline, input_row, SETTINGS, max_samples=1)
        shap.plots.waterfall(explanation[0], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption(
            "Barras vermelhas empurram a previsão para sucesso; azuis, para falha. "
            "O eixo parte do valor base (média do modelo)."
        )
    except Exception as exc:  # noqa: BLE001 - explicação é complementar
        st.info(f"Não foi possível gerar a explicação SHAP: {exc}")


def main() -> None:
    """Ponto de entrada da aplicação Streamlit."""
    st.set_page_config(page_title="SpaceX Launch Success", page_icon="🚀", layout="wide")
    st.title("🚀 Previsão de Sucesso de Lançamentos da SpaceX")
    st.write(
        "Modelo de classificação treinado em dados de lançamentos da SpaceX. "
        "Ajuste os parâmetros na barra lateral e veja a probabilidade de sucesso."
    )

    try:
        pipeline, metadata = _load_artifacts()
    except ModelNotFoundError:
        st.error(
            "Modelo não encontrado. Rode o treino primeiro: `python scripts/run_training.py` "
            "(ou `make train`)."
        )
        st.stop()

    target = metadata.get("target", SETTINGS.target)
    model_name = metadata.get("model_name", "modelo")
    st.caption(f"Modelo em uso: **{model_name}** | alvo: **{target}**")

    options = _category_options()
    values = _render_sidebar(options)
    input_row = _build_input_row(values)

    if st.button("Prever", type="primary"):
        _render_prediction(pipeline, input_row, target)
        _render_shap(pipeline, input_row)
    else:
        st.info("Defina os parâmetros na barra lateral e clique em **Prever**.")


if __name__ == "__main__":
    main()
