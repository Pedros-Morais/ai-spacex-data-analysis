"""Tests for feature engineering and the preprocessor (no leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from launch_success.config import Settings
from launch_success.features.engineering import (
    add_derived_features,
    build_preprocessor,
    split_features_target,
)


def test_add_derived_features_extracts_year() -> None:
    frame = pd.DataFrame(
        {"date_utc": ["2018-02-06T20:45:00.000Z", "2022-12-01T00:00:00Z"], "year": [np.nan, np.nan]}
    )
    out = add_derived_features(frame)
    assert out["year"].tolist() == [2018, 2022]


def test_split_features_target(clean_frame: pd.DataFrame) -> None:
    settings = Settings()
    x, y = split_features_target(clean_frame, settings=settings, target="success")
    assert list(x.columns) == settings.feature_columns
    assert "success" not in x.columns
    assert set(y.unique()) <= {0, 1}


def test_preprocessor_shape_and_no_leakage(clean_frame: pd.DataFrame) -> None:
    settings = Settings()
    x, y = split_features_target(clean_frame, settings=settings, target="success")
    pre = build_preprocessor(settings)

    # Fit only on the "train" half and transform the "test" half.
    half = len(x) // 2
    x_train, x_test = x.iloc[:half], x.iloc[half:]
    pre.fit(x_train)
    transformed_train = pre.transform(x_train)
    transformed_test = pre.transform(x_test)

    # Train and test must have the same number of columns (OneHot handle_unknown="ignore").
    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert transformed_train.shape[0] == len(x_train)
    # No NaNs remain after imputation.
    dense_test = (
        transformed_test.toarray() if hasattr(transformed_test, "toarray") else transformed_test
    )
    assert not np.isnan(dense_test).any()


def test_onehot_ignores_unknown_category() -> None:
    settings = Settings()
    train = pd.DataFrame(
        {
            "flight_number": [1, 2],
            "year": [2020, 2021],
            "payload_mass_kg": [1000.0, 2000.0],
            "flights": [1, 1],
            "rocket": ["Falcon 9", "Falcon 9"],
            "orbit": ["LEO", "GTO"],
            "launch_site": ["KSC", "VAFB"],
            "reused": [0.0, 1.0],
            "gridfins": [1.0, 1.0],
            "legs": [1.0, 1.0],
        }
    )
    test = train.copy()
    test.loc[0, "rocket"] = "Starship"  # category never seen during training
    pre = build_preprocessor(settings)
    pre.fit(train)
    # Must not raise an error thanks to handle_unknown="ignore".
    assert pre.transform(test).shape[1] == pre.transform(train).shape[1]
