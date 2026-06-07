"""Streamlit application for SpaceX launch success inference.

Loads the persisted winning pipeline, accepts launch parameters
(payload mass, orbit, rocket version, reused booster + extras),
and displays the success probability along with the SHAP explanation
for that prediction.

Local execution:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# Ensures the package under src/ is importable when launched by Streamlit.
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

# Default selection values used when the dataset is not available.
_DEFAULT_ROCKETS = ["Falcon 1", "Falcon 9", "Falcon Heavy"]
_DEFAULT_ORBITS = ["LEO", "VLEO", "GTO", "ISS", "SSO", "MEO", "PO"]
_DEFAULT_SITES = ["CCSFS SLC 40", "KSC LC 39A", "VAFB SLC 4E", "Kwajalein Atoll"]


@st.cache_resource(show_spinner=False)
def _load_artifacts() -> tuple[object, dict]:
    """Loads the trained pipeline and its metadata (cached)."""
    pipeline = load_model(settings=SETTINGS)
    metadata = load_metadata(settings=SETTINGS)
    return pipeline, metadata


@st.cache_data(show_spinner=False)
def _category_options() -> dict[str, list[str]]:
    """Reads the dataset to populate selector options (with fallback)."""
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
    """Builds a single-row DataFrame with the features expected by the model."""
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
    """Renders the input controls and returns the chosen values."""
    st.sidebar.header("Launch parameters")
    return {
        "rocket": st.sidebar.selectbox("Rocket version", options["rocket"]),
        "orbit": st.sidebar.selectbox("Target orbit", options["orbit"]),
        "launch_site": st.sidebar.selectbox("Launch site", options["launch_site"]),
        "payload_mass_kg": st.sidebar.slider("Payload mass (kg)", 0, 23_000, 5_000, step=100),
        "reused": st.sidebar.checkbox("Reused booster", value=False),
        "flights": st.sidebar.number_input(
            "Core accumulated flights", min_value=1, max_value=30, value=1
        ),
        "gridfins": st.sidebar.checkbox("Has grid fins", value=True),
        "legs": st.sidebar.checkbox("Has landing legs", value=True),
        "flight_number": st.sidebar.number_input(
            "Flight number", min_value=1, max_value=500, value=200
        ),
        "year": st.sidebar.slider("Year", 2006, 2030, 2024),
    }


def _render_prediction(pipeline: object, input_row: pd.DataFrame, target: str) -> None:
    """Displays the predicted probability and the verdict."""
    proba = float(pipeline.predict_proba(input_row)[0, 1])  # type: ignore[attr-defined]
    label = "Likely SUCCESS" if proba >= 0.5 else "RISK of failure"
    col1, col2 = st.columns(2)
    col1.metric(f"Probability of '{target}'", f"{proba:.1%}")
    col2.metric("Verdict", label)
    st.progress(proba)


def _render_shap(pipeline: object, input_row: pd.DataFrame) -> None:
    """Computes and displays the SHAP (waterfall) explanation for the prediction."""
    st.subheader("Why this prediction? (SHAP)")
    try:
        explanation = compute_shap_explanation(pipeline, input_row, SETTINGS, max_samples=1)
        shap.plots.waterfall(explanation[0], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption(
            "Red bars push the prediction toward success; blue bars push toward failure. "
            "The axis starts from the base value (model average)."
        )
    except Exception as exc:  # noqa: BLE001 - explanation is supplementary
        st.info(f"Could not generate the SHAP explanation: {exc}")


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="SpaceX Launch Success", page_icon="🚀", layout="wide")
    st.title("🚀 SpaceX Launch Success Predictor")
    st.write(
        "Classification model trained on SpaceX launch data. "
        "Adjust the parameters in the sidebar and see the probability of success."
    )

    try:
        pipeline, metadata = _load_artifacts()
    except ModelNotFoundError:
        st.error(
            "Model not found. Run training first: `python scripts/run_training.py` "
            "(or `make train`)."
        )
        st.stop()

    target = metadata.get("target", SETTINGS.target)
    model_name = metadata.get("model_name", "model")
    st.caption(f"Model in use: **{model_name}** | target: **{target}**")

    options = _category_options()
    values = _render_sidebar(options)
    input_row = _build_input_row(values)

    if st.button("Predict", type="primary"):
        _render_prediction(pipeline, input_row, target)
        _render_shap(pipeline, input_row)
    else:
        st.info("Set the parameters in the sidebar and click **Predict**.")


if __name__ == "__main__":
    main()
