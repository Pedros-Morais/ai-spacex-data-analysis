"""Shared fixtures for the test suite.

Includes a mocked API v4 JSON response (no real network), a ``Settings`` instance pointing
to temporary directories, and sample DataFrames (raw and cleaned).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from launch_success.config import Settings
from launch_success.data.synthetic import generate_synthetic_launches
from launch_success.features.cleaning import clean_launches


@pytest.fixture
def mock_api() -> dict[str, list[dict[str, Any]]]:
    """Mocked JSON for the four collections of the SpaceX API v4.

    Contains one normal launch, one upcoming launch, and one with a null outcome
    (``success`` is null), exercising the cleaning rules.
    """
    launches = [
        {
            "flight_number": 1,
            "date_utc": "2020-05-30T19:22:00.000Z",
            "rocket": "rocket_f9",
            "success": True,
            "upcoming": False,
            "launchpad": "pad_39a",
            "payloads": ["pl_a", "pl_b"],
            "cores": [
                {
                    "reused": False,
                    "flights": 1,
                    "gridfins": True,
                    "legs": True,
                    "landing_success": True,
                }
            ],
        },
        {
            "flight_number": 2,
            "date_utc": "2030-01-01T00:00:00.000Z",
            "rocket": "rocket_f9",
            "success": None,
            "upcoming": True,
            "launchpad": "pad_40",
            "payloads": ["pl_c"],
            "cores": [{"reused": True, "flights": 3, "gridfins": True, "legs": True}],
        },
        {
            "flight_number": 3,
            "date_utc": "2018-02-06T20:45:00.000Z",
            "rocket": "rocket_fh",
            "success": False,
            "upcoming": False,
            "launchpad": "pad_39a",
            "payloads": [],
            "cores": [],
        },
    ]
    rockets = [
        {"id": "rocket_f9", "name": "Falcon 9"},
        {"id": "rocket_fh", "name": "Falcon Heavy"},
    ]
    payloads = [
        {"id": "pl_a", "mass_kg": 2000.0, "orbit": "LEO"},
        {"id": "pl_b", "mass_kg": 500.0, "orbit": "LEO"},
        {"id": "pl_c", "mass_kg": None, "orbit": "GTO"},
    ]
    launchpads = [
        {"id": "pad_39a", "name": "KSC LC 39A"},
        {"id": "pad_40", "name": "CCSFS SLC 40"},
    ]
    return {
        "launches": launches,
        "rockets": rockets,
        "payloads": payloads,
        "launchpads": launchpads,
    }


@pytest.fixture
def tmp_settings(tmp_path) -> Settings:
    """Settings with isolated directories under ``tmp_path`` and reduced CV folds."""
    return Settings(
        seed=42,
        cv_folds=3,
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        processed_dir=tmp_path / "data" / "processed",
        models_dir=tmp_path / "models",
        figures_dir=tmp_path / "figures",
    )


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Small deterministic synthetic raw DataFrame."""
    return generate_synthetic_launches(n_rows=200, seed=7)


@pytest.fixture
def clean_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Cleaned DataFrame for the primary target ``success``."""
    return clean_launches(raw_frame, target="success")
