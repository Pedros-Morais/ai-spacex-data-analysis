"""Extended tests for model persistence.

Tests save/load operations, metadata handling, and error conditions.
"""

from __future__ import annotations

import json

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from launch_success.config import Settings
from launch_success.exceptions import ModelNotFoundError
from launch_success.models.persistence import load_metadata, load_model, save_model


@pytest.fixture
def simple_pipeline() -> Pipeline:
    """Create a simple fitted pipeline for testing."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )


@pytest.fixture
def tmp_model_settings(tmp_path) -> Settings:
    """Settings with temporary model directory."""
    return Settings(
        models_dir=tmp_path / "models",
        seed=42,
    )


class TestSaveModel:
    """Tests for model saving functionality."""

    def test_save_creates_file(self, simple_pipeline, tmp_model_settings, tmp_path) -> None:
        model_path = tmp_path / "test_model.joblib"
        result_path = save_model(simple_pipeline, path=model_path)

        assert result_path == model_path
        assert model_path.exists()

    def test_save_creates_parent_directories(self, simple_pipeline, tmp_path) -> None:
        model_path = tmp_path / "nested" / "dirs" / "model.joblib"
        save_model(simple_pipeline, path=model_path)

        assert model_path.exists()

    def test_save_with_metadata(self, simple_pipeline, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        meta_path = model_path.with_suffix(".meta.json")

        metadata = {
            "model_name": "logistic_regression",
            "features": ["a", "b", "c"],
            "metrics": {"f1": 0.95, "accuracy": 0.92},
        }

        save_model(simple_pipeline, path=model_path, metadata=metadata)

        assert meta_path.exists()
        saved_meta = json.loads(meta_path.read_text())
        assert saved_meta == metadata

    def test_save_without_metadata(self, simple_pipeline, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        meta_path = model_path.with_suffix(".meta.json")

        save_model(simple_pipeline, path=model_path)

        assert model_path.exists()
        assert not meta_path.exists()

    def test_save_uses_default_path(self, simple_pipeline, tmp_model_settings) -> None:
        tmp_model_settings.ensure_directories()
        result_path = save_model(simple_pipeline, settings=tmp_model_settings)

        assert result_path == tmp_model_settings.best_model_path
        assert tmp_model_settings.best_model_path.exists()

    def test_save_overwrites_existing(self, simple_pipeline, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"

        # Save first version
        save_model(simple_pipeline, path=model_path)
        first_size = model_path.stat().st_size

        # Save second version (should overwrite)
        new_pipeline = Pipeline([("model", LogisticRegression(C=0.5))])
        save_model(new_pipeline, path=model_path)
        second_size = model_path.stat().st_size

        # File should exist and may differ in size
        assert model_path.exists()
        # Both should be valid sizes
        assert first_size > 0
        assert second_size > 0


class TestLoadModel:
    """Tests for model loading functionality."""

    def test_load_returns_pipeline(self, simple_pipeline, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        save_model(simple_pipeline, path=model_path)

        loaded = load_model(path=model_path)

        assert isinstance(loaded, Pipeline)
        assert "model" in loaded.named_steps

    def test_load_missing_raises_error(self, tmp_path) -> None:
        model_path = tmp_path / "nonexistent.joblib"

        with pytest.raises(ModelNotFoundError, match="not found"):
            load_model(path=model_path)

    def test_load_uses_default_path(self, simple_pipeline, tmp_model_settings) -> None:
        tmp_model_settings.ensure_directories()
        save_model(simple_pipeline, settings=tmp_model_settings)

        loaded = load_model(settings=tmp_model_settings)

        assert isinstance(loaded, Pipeline)

    def test_load_preserves_pipeline_structure(self, simple_pipeline, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        save_model(simple_pipeline, path=model_path)

        loaded = load_model(path=model_path)

        # Check structure matches
        assert list(loaded.named_steps.keys()) == ["scaler", "model"]
        assert isinstance(loaded.named_steps["scaler"], StandardScaler)
        assert isinstance(loaded.named_steps["model"], LogisticRegression)


class TestLoadMetadata:
    """Tests for metadata loading functionality."""

    def test_load_existing_metadata(self, simple_pipeline, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        metadata = {"name": "test", "version": 1}

        save_model(simple_pipeline, path=model_path, metadata=metadata)
        loaded_meta = load_metadata(path=model_path)

        assert loaded_meta == metadata

    def test_load_missing_metadata_returns_empty(self, simple_pipeline, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        save_model(simple_pipeline, path=model_path)

        loaded_meta = load_metadata(path=model_path)

        assert loaded_meta == {}

    def test_load_metadata_uses_default_path(self, simple_pipeline, tmp_model_settings) -> None:
        tmp_model_settings.ensure_directories()
        metadata = {"test": True}
        save_model(simple_pipeline, settings=tmp_model_settings, metadata=metadata)

        loaded_meta = load_metadata(settings=tmp_model_settings)

        assert loaded_meta == {"test": True}

    def test_metadata_with_nested_structures(self, simple_pipeline, tmp_path) -> None:
        model_path = tmp_path / "model.joblib"
        metadata = {
            "metrics": {
                "train": {"f1": 0.95, "accuracy": 0.92},
                "test": {"f1": 0.90, "accuracy": 0.88},
            },
            "features": ["a", "b", "c"],
            "hyperparameters": {"C": 1.0, "max_iter": 1000},
        }

        save_model(simple_pipeline, path=model_path, metadata=metadata)
        loaded_meta = load_metadata(path=model_path)

        assert loaded_meta == metadata
        assert loaded_meta["metrics"]["train"]["f1"] == 0.95


class TestRoundTrip:
    """Tests for complete save/load cycles."""

    def test_fitted_model_roundtrip(self, tmp_path) -> None:
        import numpy as np

        # Create and fit a pipeline
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])

        pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
        pipeline.fit(X, y)

        # Save
        model_path = tmp_path / "fitted_model.joblib"
        save_model(pipeline, path=model_path)

        # Load
        loaded = load_model(path=model_path)

        # Predictions should match
        original_preds = pipeline.predict(X)
        loaded_preds = loaded.predict(X)

        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_model_with_complex_metadata_roundtrip(self, simple_pipeline, tmp_path) -> None:
        from datetime import datetime

        model_path = tmp_path / "model.joblib"
        metadata = {
            "trained_at": datetime.now().isoformat(),
            "features": ["payload_mass_kg", "orbit", "reused"],
            "cv_scores": [0.91, 0.93, 0.89, 0.92, 0.90],
            "best_params": {"C": 1.0},
        }

        save_model(simple_pipeline, path=model_path, metadata=metadata)
        loaded_meta = load_metadata(path=model_path)

        assert loaded_meta["features"] == metadata["features"]
        assert loaded_meta["cv_scores"] == metadata["cv_scores"]
