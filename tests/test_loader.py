"""Testes do carregamento de dataset (validação de schema e erros)."""

from __future__ import annotations

import pandas as pd
import pytest

from launch_success.data.loader import load_dataset
from launch_success.data.synthetic import write_synthetic_dataset
from launch_success.exceptions import DataValidationError


def test_load_dataset_ok(tmp_settings) -> None:
    write_synthetic_dataset(n_rows=50, settings=tmp_settings)
    frame = load_dataset(settings=tmp_settings)
    assert len(frame) == 50
    assert "success" in frame.columns


def test_load_dataset_arquivo_ausente(tmp_settings) -> None:
    with pytest.raises(DataValidationError, match="não encontrado"):
        load_dataset(settings=tmp_settings)


def test_load_dataset_vazio(tmp_settings) -> None:
    tmp_settings.ensure_directories()
    tmp_settings.processed_csv.write_text("col\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        load_dataset(settings=tmp_settings)


def test_load_dataset_colunas_faltando(tmp_settings) -> None:
    tmp_settings.ensure_directories()
    pd.DataFrame({"flight_number": [1, 2]}).to_csv(tmp_settings.processed_csv, index=False)
    with pytest.raises(DataValidationError, match="obrigatórias"):
        load_dataset(settings=tmp_settings)
