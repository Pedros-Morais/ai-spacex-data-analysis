"""Tests for the synthetic dataset generator (reproducibility and calibration)."""

from __future__ import annotations

from launch_success.data.ingestion import DATASET_COLUMNS
from launch_success.data.synthetic import generate_synthetic_launches


def test_generates_correct_count_and_schema() -> None:
    frame = generate_synthetic_launches(n_rows=1000, seed=42)
    assert len(frame) == 1000
    for column in DATASET_COLUMNS:
        assert column in frame.columns


def test_deterministic_with_seed() -> None:
    a = generate_synthetic_launches(n_rows=300, seed=42)
    b = generate_synthetic_launches(n_rows=300, seed=42)
    assert a.equals(b)
    c = generate_synthetic_launches(n_rows=300, seed=7)
    assert not a.equals(c)


def test_statistical_calibration() -> None:
    frame = generate_synthetic_launches(n_rows=1500, seed=1)
    # Overall success rate is high (reflecting the real SpaceX class imbalance).
    assert frame["success"].mean() > 0.85
    # Falcon 1 is much less reliable than Falcon 9.
    f1_rate = frame.loc[frame["rocket"] == "Falcon 1", "success"].mean()
    f9_rate = frame.loc[frame["rocket"] == "Falcon 9", "success"].mean()
    assert f1_rate < f9_rate
    # Some payload masses are missing to exercise imputation.
    assert frame["payload_mass_kg"].isna().any()
    # Landing outcomes only exist when an attempt was made; the alternative target is more balanced.
    attempts = frame["landing_success"].dropna()
    assert 0.3 < attempts.mean() < 0.97
