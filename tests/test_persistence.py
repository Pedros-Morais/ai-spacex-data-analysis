"""Tests for model serialization and loading with joblib."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from launch_success.config import Settings
from launch_success.exceptions import ModelNotFoundError
from launch_success.features.engineering import split_features_target
from launch_success.models.persistence import load_metadata, load_model, save_model
from launch_success.models.trainer import build_pipeline, stratified_split


@pytest.fixture
def fitted_pipeline(clean_frame):
    settings = Settings(cv_folds=3)
    x, y = split_features_target(clean_frame, settings=settings, target="success")
    x_tr, _, y_tr, _ = stratified_split(x, y, settings)
    pipeline = build_pipeline(LogisticRegression(max_iter=500, random_state=42), settings)
    pipeline.fit(x_tr, y_tr)
    return pipeline, x


def test_save_load_predicts_identically(fitted_pipeline, tmp_settings) -> None:
    pipeline, x = fitted_pipeline
    before = pipeline.predict(x)
    save_model(pipeline, settings=tmp_settings)
    loaded = load_model(settings=tmp_settings)
    after = loaded.predict(x)
    assert np.array_equal(before, after)


def test_metadata_roundtrip(fitted_pipeline, tmp_settings) -> None:
    pipeline, _ = fitted_pipeline
    meta = {"model_name": "logistic_regression", "target": "success"}
    save_model(pipeline, settings=tmp_settings, metadata=meta)
    assert load_metadata(settings=tmp_settings) == meta


def test_load_nonexistent_raises_error(tmp_settings) -> None:
    with pytest.raises(ModelNotFoundError):
        load_model(settings=tmp_settings)


def test_missing_metadata_returns_empty_dict(fitted_pipeline, tmp_settings) -> None:
    pipeline, _ = fitted_pipeline
    save_model(pipeline, settings=tmp_settings)  # no metadata
    assert load_metadata(settings=tmp_settings) == {}
