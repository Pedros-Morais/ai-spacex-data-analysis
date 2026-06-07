"""Tests for the centralised configuration."""

from __future__ import annotations

from launch_success.config import SETTINGS, Settings


def test_feature_columns_concatenates_groups() -> None:
    settings = Settings()
    assert settings.feature_columns == [
        *settings.numeric_features,
        *settings.categorical_features,
        *settings.boolean_features,
    ]


def test_derived_paths() -> None:
    settings = Settings()
    assert settings.processed_csv.name == "spacex_launches.csv"
    assert settings.best_model_path.suffix == ".joblib"
    assert settings.raw_json.parent == settings.raw_dir


def test_ensure_directories_creates_folders(tmp_path) -> None:
    settings = Settings(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "proc",
        models_dir=tmp_path / "models",
        figures_dir=tmp_path / "figs",
    )
    settings.ensure_directories()
    assert settings.raw_dir.exists()
    assert settings.figures_dir.exists()


def test_default_target_and_override() -> None:
    assert SETTINGS.target in {"success", "landing_success"}
    assert Settings(target="landing_success").target == "landing_success"
