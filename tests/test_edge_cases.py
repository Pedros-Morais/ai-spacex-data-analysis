"""Edge case tests for comprehensive coverage.

Tests edge cases and boundary conditions across the codebase to ensure
robustness and complete coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from launch_success.config import Settings
from launch_success.data.synthetic import generate_synthetic_launches
from launch_success.evaluation.metrics import METRIC_NAMES, compute_metrics
from launch_success.exceptions import DataValidationError
from launch_success.features.cleaning import clean_launches, coerce_boolean
from launch_success.features.engineering import (
    add_derived_features,
    build_preprocessor,
    split_features_target,
)
from launch_success.models.trainer import (
    build_pipeline,
    cross_validate_score,
    stratified_split,
)


# --------------------------------------------------------------------------- #
# coerce_boolean edge cases
# --------------------------------------------------------------------------- #
class TestCoerceBooleanEdgeCases:
    """Edge case tests for boolean coercion."""

    def test_empty_series_returns_empty(self) -> None:
        result = coerce_boolean(pd.Series([], dtype=object))
        assert len(result) == 0
        assert result.dtype == float

    def test_all_nan_values(self) -> None:
        series = pd.Series([None, np.nan, float("nan"), None])
        result = coerce_boolean(series)
        assert all(np.isnan(result))

    def test_mixed_case_strings(self) -> None:
        series = pd.Series(["TRUE", "FALSE", "True", "False", "tRuE", "fAlSe"])
        result = coerce_boolean(series)
        expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        assert result.tolist() == expected

    def test_yes_no_variants(self) -> None:
        series = pd.Series(["yes", "no", "YES", "NO", "Yes", "No"])
        result = coerce_boolean(series)
        expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        assert result.tolist() == expected

    def test_t_f_shorthand(self) -> None:
        series = pd.Series(["t", "f", "T", "F"])
        result = coerce_boolean(series)
        expected = [1.0, 0.0, 1.0, 0.0]
        assert result.tolist() == expected

    def test_numeric_floats(self) -> None:
        series = pd.Series([1.0, 0.0, 2.5, -1.0])
        result = coerce_boolean(series)
        # Non-0/1 floats are passed through as floats
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == 2.5
        assert result.iloc[3] == -1.0

    def test_numpy_bool(self) -> None:
        series = pd.Series([np.bool_(True), np.bool_(False)])
        result = coerce_boolean(series)
        assert result.tolist() == [1.0, 0.0]

    def test_unrecognized_strings_become_nan(self) -> None:
        series = pd.Series(["maybe", "unknown", "N/A", ""])
        result = coerce_boolean(series)
        assert all(np.isnan(result))


# --------------------------------------------------------------------------- #
# clean_launches edge cases
# --------------------------------------------------------------------------- #
class TestCleanLaunchesEdgeCases:
    """Edge case tests for launch cleaning."""

    def test_all_upcoming_launches_raises_error(self) -> None:
        df = pd.DataFrame(
            {
                "flight_number": [1, 2],
                "date_utc": ["2030-01-01", "2030-01-02"],
                "year": [2030, 2030],
                "rocket": ["Falcon 9", "Falcon 9"],
                "payload_mass_kg": [1000, 2000],
                "orbit": ["LEO", "LEO"],
                "launch_site": ["KSC", "KSC"],
                "reused": [False, False],
                "flights": [1, 1],
                "gridfins": [True, True],
                "legs": [True, True],
                "landing_success": [True, True],
                "success": [True, True],
                "upcoming": [True, True],
            }
        )
        with pytest.raises(DataValidationError, match="No rows left"):
            clean_launches(df, target="success")

    def test_all_null_target_raises_error(self) -> None:
        df = pd.DataFrame(
            {
                "flight_number": [1, 2],
                "date_utc": ["2020-01-01", "2020-01-02"],
                "year": [2020, 2020],
                "rocket": ["Falcon 9", "Falcon 9"],
                "payload_mass_kg": [1000, 2000],
                "orbit": ["LEO", "LEO"],
                "launch_site": ["KSC", "KSC"],
                "reused": [False, False],
                "flights": [1, 1],
                "gridfins": [True, True],
                "legs": [True, True],
                "landing_success": [True, True],
                "success": [None, None],
                "upcoming": [False, False],
            }
        )
        with pytest.raises(DataValidationError, match="No rows left"):
            clean_launches(df, target="success")

    def test_no_upcoming_column(self) -> None:
        df = pd.DataFrame(
            {
                "flight_number": [1],
                "date_utc": ["2020-01-01"],
                "year": [2020],
                "rocket": ["Falcon 9"],
                "payload_mass_kg": [1000],
                "orbit": ["LEO"],
                "launch_site": ["KSC"],
                "reused": [False],
                "flights": [1],
                "gridfins": [True],
                "legs": [True],
                "landing_success": [True],
                "success": [True],
            }
        )
        result = clean_launches(df, target="success")
        assert len(result) == 1

    def test_preserves_non_target_columns(self) -> None:
        df = pd.DataFrame(
            {
                "flight_number": [1, 2, 3],
                "date_utc": ["2020-01-01"] * 3,
                "year": [2020, 2020, 2020],
                "rocket": ["Falcon 9", "Falcon Heavy", "Falcon 9"],
                "payload_mass_kg": [1000, 2000, 3000],
                "orbit": ["LEO", "GTO", "SSO"],
                "launch_site": ["KSC", "VAFB", "KSC"],
                "reused": [False, True, False],
                "flights": [1, 2, 1],
                "gridfins": [True, True, False],
                "legs": [True, True, False],
                "landing_success": [True, False, None],
                "success": [True, False, True],
                "upcoming": [False, False, False],
            }
        )
        result = clean_launches(df, target="success")
        assert "rocket" in result.columns
        assert "Falcon Heavy" in result["rocket"].values


# --------------------------------------------------------------------------- #
# Feature engineering edge cases
# --------------------------------------------------------------------------- #
class TestFeatureEngineeringEdgeCases:
    """Edge case tests for feature engineering."""

    def test_add_derived_features_no_date_utc(self) -> None:
        df = pd.DataFrame({"year": [2020, 2021], "other": [1, 2]})
        result = add_derived_features(df)
        assert "year" in result.columns
        assert result["year"].tolist() == [2020, 2021]

    def test_add_derived_features_invalid_dates(self) -> None:
        df = pd.DataFrame({"date_utc": ["invalid", "not-a-date", "2020-01-01T00:00:00Z"]})
        result = add_derived_features(df)
        assert "year" in result.columns
        # Invalid dates should become NaN, valid date should be 2020
        assert np.isnan(result["year"].iloc[0])
        assert np.isnan(result["year"].iloc[1])
        assert result["year"].iloc[2] == 2020

    def test_preprocessor_handles_unknown_categories(self) -> None:
        settings = Settings()
        preprocessor = build_preprocessor(settings)

        # Create training data with all required features
        train_df = pd.DataFrame(
            {
                "flight_number": [1, 2, 3],
                "payload_mass_kg": [1000, 2000, 3000],
                "year": [2020, 2021, 2022],
                "flights": [1, 2, 3],
                "rocket": ["Falcon 9", "Falcon 9", "Falcon Heavy"],
                "orbit": ["LEO", "GTO", "LEO"],
                "launch_site": ["KSC", "VAFB", "KSC"],
                "reused": [0.0, 1.0, 0.0],
                "gridfins": [1.0, 1.0, 0.0],
                "legs": [1.0, 1.0, 0.0],
            }
        )

        # Create test data with unknown category
        test_df = pd.DataFrame(
            {
                "flight_number": [4],
                "payload_mass_kg": [1500],
                "year": [2023],
                "flights": [4],
                "rocket": ["Starship"],  # Unknown rocket
                "orbit": ["Mars"],  # Unknown orbit
                "launch_site": ["Boca Chica"],  # Unknown site
                "reused": [1.0],
                "gridfins": [1.0],
                "legs": [1.0],
            }
        )

        preprocessor.fit(train_df)
        # Should not raise an error with handle_unknown="ignore"
        result = preprocessor.transform(test_df)
        assert result.shape[0] == 1


# --------------------------------------------------------------------------- #
# Metrics edge cases
# --------------------------------------------------------------------------- #
class TestMetricsEdgeCases:
    """Edge case tests for evaluation metrics."""

    def test_all_zeros_prediction(self) -> None:
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 0, 0, 0]
        y_proba = [0.1, 0.2, 0.3, 0.4, 0.4]
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["f1"] == 0.0
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics

    def test_all_ones_prediction(self) -> None:
        y_true = [0, 0, 1, 1, 1]
        y_pred = [1, 1, 1, 1, 1]
        y_proba = [0.6, 0.7, 0.8, 0.9, 0.95]
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 0.6  # 3/5

    def test_single_sample(self) -> None:
        y_true = [1]
        y_pred = [1]
        y_proba = [0.9]
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert metrics["accuracy"] == 1.0
        # ROC-AUC undefined for single sample
        assert np.isnan(metrics["roc_auc"])

    def test_metric_names_constant(self) -> None:
        assert len(METRIC_NAMES) == 6
        assert "accuracy" in METRIC_NAMES
        assert "precision" in METRIC_NAMES
        assert "recall" in METRIC_NAMES
        assert "f1" in METRIC_NAMES
        assert "roc_auc" in METRIC_NAMES
        assert "pr_auc" in METRIC_NAMES


# --------------------------------------------------------------------------- #
# Trainer edge cases
# --------------------------------------------------------------------------- #
class TestTrainerEdgeCases:
    """Edge case tests for model training."""

    def test_stratified_split_preserves_indices(self) -> None:
        df = generate_synthetic_launches(n_rows=100, seed=42)
        df = clean_launches(df, target="success")
        settings = Settings(test_size=0.2, seed=42)
        x, y = split_features_target(df, settings=settings, target="success")

        x_train, x_test, y_train, y_test = stratified_split(x, y, settings)

        # Indices should not overlap
        train_idx = set(x_train.index)
        test_idx = set(x_test.index)
        assert train_idx.isdisjoint(test_idx)

    def test_cross_validate_minimum_folds(self) -> None:
        # Create minimal dataset with all required features
        df = pd.DataFrame(
            {
                "flight_number": [1, 2, 3, 4],
                "payload_mass_kg": [1000, 2000, 3000, 4000],
                "year": [2020, 2020, 2020, 2020],
                "flights": [1, 2, 1, 2],
                "rocket": ["F9", "F9", "F9", "F9"],
                "orbit": ["LEO", "LEO", "LEO", "LEO"],
                "launch_site": ["KSC", "KSC", "KSC", "KSC"],
                "reused": [0.0, 1.0, 0.0, 1.0],
                "gridfins": [1.0, 1.0, 1.0, 1.0],
                "legs": [1.0, 1.0, 1.0, 1.0],
            }
        )
        y = pd.Series([0, 1, 0, 1])

        from sklearn.linear_model import LogisticRegression

        settings = Settings(cv_folds=10, seed=42)  # More folds than samples per class
        pipeline = build_pipeline(LogisticRegression(max_iter=500), settings)

        # Should automatically reduce folds to viable count
        mean, std = cross_validate_score(pipeline, df, y, settings)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert not np.isnan(mean)

    def test_build_pipeline_with_different_estimators(self) -> None:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        settings = Settings(use_smote=False)

        estimators = [
            LogisticRegression(),
            RandomForestClassifier(n_estimators=10),
            GradientBoostingClassifier(n_estimators=10),
        ]

        for est in estimators:
            pipe = build_pipeline(est, settings)
            assert isinstance(pipe, Pipeline)
            assert "preprocessor" in pipe.named_steps
            assert "model" in pipe.named_steps


# --------------------------------------------------------------------------- #
# Synthetic data edge cases
# --------------------------------------------------------------------------- #
class TestSyntheticDataEdgeCases:
    """Edge case tests for synthetic data generation."""

    def test_generate_single_row(self) -> None:
        df = generate_synthetic_launches(n_rows=1, seed=42)
        assert len(df) == 1
        assert "success" in df.columns

    def test_generate_large_dataset(self) -> None:
        df = generate_synthetic_launches(n_rows=5000, seed=42)
        assert len(df) == 5000
        # Check class distribution is reasonable
        success_rate = df["success"].mean()
        assert 0.8 < success_rate < 1.0  # SpaceX has high success rate

    def test_reproducibility_across_calls(self) -> None:
        df1 = generate_synthetic_launches(n_rows=100, seed=123)
        df2 = generate_synthetic_launches(n_rows=100, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self) -> None:
        df1 = generate_synthetic_launches(n_rows=100, seed=1)
        df2 = generate_synthetic_launches(n_rows=100, seed=2)
        # DataFrames should differ
        assert not df1.equals(df2)
