"""Testes da limpeza: coerção de tipos, filtragem e tratamento de nulos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from launch_success.exceptions import DataValidationError
from launch_success.features.cleaning import clean_launches, coerce_boolean


def test_coerce_boolean_variados() -> None:
    series = pd.Series([True, False, "True", "false", 1, 0, "1", "0", None, "abc"])
    result = coerce_boolean(series)
    expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, np.nan, np.nan]
    assert result.tolist()[:8] == expected[:8]
    assert np.isnan(result.iloc[8]) and np.isnan(result.iloc[9])


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_number": [1, 2, 3, 4],
            "date_utc": ["2020-01-01T00:00:00Z"] * 4,
            "year": [2020, 2030, 2018, 2019],
            "rocket": ["Falcon 9", "Falcon 9", "Falcon 9", None],
            "payload_mass_kg": [2000.0, None, 5000.0, 1000.0],
            "orbit": ["LEO", "GTO", "LEO", "SSO"],
            "launch_site": ["KSC", "KSC", "VAFB", "KSC"],
            "reused": [True, False, "True", None],
            "flights": [1, 2, 3, 1],
            "gridfins": [True, True, False, True],
            "legs": [True, True, False, True],
            "landing_success": [True, None, False, True],
            "success": [True, None, False, True],
            "upcoming": [False, True, False, False],
        }
    )


def test_remove_upcoming_e_alvo_nulo() -> None:
    cleaned = clean_launches(_raw(), target="success")
    # Linha 2 (upcoming=True, success=None) deve sair.
    assert len(cleaned) == 3
    assert cleaned["success"].notna().all()


def test_coercao_de_tipos() -> None:
    cleaned = clean_launches(_raw(), target="success")
    assert cleaned["success"].dtype.kind in "iu"  # inteiro 0/1
    assert set(cleaned["success"].unique()) <= {0, 1}
    assert cleaned["reused"].dtype.kind == "f"  # float 0/1/NaN
    assert pd.api.types.is_numeric_dtype(cleaned["payload_mass_kg"])


def test_alvo_alternativo_landing_success() -> None:
    cleaned = clean_launches(_raw(), target="landing_success")
    # Linhas com landing_success nulo são descartadas para este alvo.
    assert "landing_success" in cleaned.columns
    assert cleaned["landing_success"].notna().all()


def test_alvo_inexistente_levanta_erro() -> None:
    with pytest.raises(DataValidationError):
        clean_launches(_raw(), target="nao_existe")
