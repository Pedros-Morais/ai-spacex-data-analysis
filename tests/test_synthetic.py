"""Testes do gerador de dataset sintético (reprodutibilidade e calibração)."""

from __future__ import annotations

from launch_success.data.ingestion import DATASET_COLUMNS
from launch_success.data.synthetic import generate_synthetic_launches


def test_gera_quantidade_e_schema() -> None:
    frame = generate_synthetic_launches(n_rows=1000, seed=42)
    assert len(frame) == 1000
    for column in DATASET_COLUMNS:
        assert column in frame.columns


def test_determinismo_com_seed() -> None:
    a = generate_synthetic_launches(n_rows=300, seed=42)
    b = generate_synthetic_launches(n_rows=300, seed=42)
    assert a.equals(b)
    c = generate_synthetic_launches(n_rows=300, seed=7)
    assert not a.equals(c)


def test_calibracao_estatistica() -> None:
    frame = generate_synthetic_launches(n_rows=1500, seed=1)
    # Taxa de sucesso global alta (desbalanceamento real da SpaceX).
    assert frame["success"].mean() > 0.85
    # Falcon 1 é bem menos confiável que Falcon 9.
    f1_rate = frame.loc[frame["rocket"] == "Falcon 1", "success"].mean()
    f9_rate = frame.loc[frame["rocket"] == "Falcon 9", "success"].mean()
    assert f1_rate < f9_rate
    # Há massas ausentes para exercitar imputação.
    assert frame["payload_mass_kg"].isna().any()
    # Pousos só existem quando há tentativa; alvo alternativo mais balanceado.
    attempts = frame["landing_success"].dropna()
    assert 0.3 < attempts.mean() < 0.97
