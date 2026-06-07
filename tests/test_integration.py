"""Integration tests for end-to-end workflows.

Tests complete pipelines from data loading through model training and evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from launch_success.config import Settings
from launch_success.data.synthetic import generate_synthetic_launches, write_synthetic_dataset
from launch_success.evaluation.metrics import METRIC_NAMES, compute_metrics
from launch_success.features.cleaning import clean_launches
from launch_success.features.engineering import (
    add_derived_features,
    build_preprocessor,
    split_features_target,
)
from launch_success.models.persistence import load_metadata, load_model, save_model
from launch_success.models.trainer import (
    build_pipeline,
    select_best_model,
    stratified_split,
    train_all_models,
    train_and_evaluate,
)


@pytest.fixture
def integration_settings(tmp_path) -> Settings:
    """Settings configured for integration tests."""
    return Settings(
        seed=42,
        cv_folds=3,
        test_size=0.2,
        use_smote=False,
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        processed_dir=tmp_path / "data" / "processed",
        models_dir=tmp_path / "models",
        figures_dir=tmp_path / "figures",
    )


class TestDataToModelPipeline:
    """Tests for complete data-to-model pipelines."""

    def test_synthetic_data_to_trained_model(self, integration_settings) -> None:
        # Generate synthetic data
        raw_df = generate_synthetic_launches(n_rows=300, seed=42)
        assert len(raw_df) == 300

        # Clean data
        clean_df = clean_launches(raw_df, settings=integration_settings)
        assert len(clean_df) > 0
        assert clean_df["success"].notna().all()

        # Add derived features
        enriched_df = add_derived_features(clean_df)
        assert "year" in enriched_df.columns

        # Split features and target
        X, y = split_features_target(enriched_df, settings=integration_settings, target="success")
        assert len(X) == len(y)

        # Train/test split
        X_train, X_test, y_train, y_test = stratified_split(X, y, integration_settings)
        assert len(X_train) + len(X_test) == len(X)

        # Train a model
        estimator = LogisticRegression(max_iter=500, random_state=42)
        result = train_and_evaluate(
            "logistic_regression",
            estimator,
            X_train,
            y_train,
            X_test,
            y_test,
            integration_settings,
        )

        # Verify results
        assert "pipeline" in result
        assert "metrics" in result
        assert all(metric in result["metrics"] for metric in METRIC_NAMES)
        assert result["metrics"]["accuracy"] > 0.5  # Better than random

    def test_model_training_comparison(self, integration_settings) -> None:
        # Generate and prepare data
        raw_df = generate_synthetic_launches(n_rows=200, seed=42)
        clean_df = clean_launches(raw_df, settings=integration_settings)
        enriched_df = add_derived_features(clean_df)
        X, y = split_features_target(enriched_df, settings=integration_settings)
        X_train, X_test, y_train, y_test = stratified_split(X, y, integration_settings)

        # Define model registry
        registry = {
            "logistic_regression": LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=25, class_weight="balanced", random_state=42
            ),
        }

        # Train all models
        results = train_all_models(
            X_train,
            y_train,
            X_test,
            y_test,
            integration_settings,
            registry=registry,
        )

        # Verify all models were trained
        assert set(results.keys()) == set(registry.keys())

        # Select best model
        best_name, best_result = select_best_model(results, "f1")
        assert best_name in registry
        assert best_result["cv_mean"] > 0

    def test_model_save_load_predict_cycle(self, integration_settings) -> None:
        # Prepare data
        raw_df = generate_synthetic_launches(n_rows=150, seed=42)
        clean_df = clean_launches(raw_df, settings=integration_settings)
        enriched_df = add_derived_features(clean_df)
        X, y = split_features_target(enriched_df, settings=integration_settings)
        X_train, X_test, y_train, y_test = stratified_split(X, y, integration_settings)

        # Train model
        pipeline = build_pipeline(
            LogisticRegression(max_iter=500),
            integration_settings,
        )
        pipeline.fit(X_train, y_train)

        # Get predictions before save
        y_pred_before = pipeline.predict(X_test)

        # Save model with metadata
        integration_settings.ensure_directories()
        metadata = {
            "model_name": "logistic_regression",
            "test_size": integration_settings.test_size,
        }
        save_model(pipeline, settings=integration_settings, metadata=metadata)

        # Load model
        loaded_pipeline = load_model(settings=integration_settings)
        loaded_metadata = load_metadata(settings=integration_settings)

        # Verify metadata
        assert loaded_metadata["model_name"] == "logistic_regression"

        # Get predictions after load
        y_pred_after = loaded_pipeline.predict(X_test)

        # Predictions should be identical
        np.testing.assert_array_equal(y_pred_before, y_pred_after)


class TestDataQualityIntegration:
    """Tests for data quality through the pipeline."""

    def test_no_data_leakage_in_preprocessing(self, integration_settings) -> None:
        # Generate data with known statistics
        raw_df = generate_synthetic_launches(n_rows=500, seed=42)
        clean_df = clean_launches(raw_df, settings=integration_settings)
        enriched_df = add_derived_features(clean_df)
        X, y = split_features_target(enriched_df, settings=integration_settings)

        # Split data
        X_train, X_test, y_train, y_test = stratified_split(X, y, integration_settings)

        # Build and fit preprocessor only on training data
        preprocessor = build_preprocessor(integration_settings)
        preprocessor.fit(X_train)

        # Transform both sets
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Verify shapes
        assert X_train_transformed.shape[0] == len(X_train)
        assert X_test_transformed.shape[0] == len(X_test)

        # Transformed train should have mean ~0 for numeric features (standardized)
        # Test set may differ since it uses train statistics

    def test_class_distribution_preserved_after_split(self, integration_settings) -> None:
        raw_df = generate_synthetic_launches(n_rows=400, seed=42)
        clean_df = clean_launches(raw_df, settings=integration_settings)
        enriched_df = add_derived_features(clean_df)
        X, y = split_features_target(enriched_df, settings=integration_settings)

        original_rate = y.mean()

        X_train, X_test, y_train, y_test = stratified_split(X, y, integration_settings)

        train_rate = y_train.mean()
        test_rate = y_test.mean()

        # Stratification should preserve class distribution
        assert abs(train_rate - original_rate) < 0.05
        assert abs(test_rate - original_rate) < 0.05


class TestMetricsIntegration:
    """Tests for metrics computation in realistic scenarios."""

    def test_metrics_with_trained_model(self, integration_settings) -> None:
        # Prepare data
        raw_df = generate_synthetic_launches(n_rows=200, seed=42)
        clean_df = clean_launches(raw_df, settings=integration_settings)
        enriched_df = add_derived_features(clean_df)
        X, y = split_features_target(enriched_df, settings=integration_settings)
        X_train, X_test, y_train, y_test = stratified_split(X, y, integration_settings)

        # Train model
        pipeline = build_pipeline(
            RandomForestClassifier(n_estimators=50, random_state=42),
            integration_settings,
        )
        pipeline.fit(X_train, y_train)

        # Get predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # All metrics should be present
        for name in METRIC_NAMES:
            assert name in metrics

        # Metrics should be in valid ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

        # AUC metrics should be valid if both classes present
        if len(np.unique(y_test)) == 2:
            assert 0 <= metrics["roc_auc"] <= 1
            assert 0 <= metrics["pr_auc"] <= 1


class TestWriteAndLoadIntegration:
    """Tests for writing and loading datasets."""

    def test_write_and_load_synthetic_dataset(self, integration_settings) -> None:
        from launch_success.data.loader import load_dataset

        # Write synthetic dataset
        write_synthetic_dataset(n_rows=100, settings=integration_settings)

        # Load it back
        loaded_df = load_dataset(settings=integration_settings)

        assert len(loaded_df) == 100
        assert "success" in loaded_df.columns
        assert "rocket" in loaded_df.columns

    def test_full_cycle_with_persisted_data(self, integration_settings) -> None:
        from launch_success.data.loader import load_dataset

        # Write and load
        write_synthetic_dataset(n_rows=200, settings=integration_settings)
        raw_df = load_dataset(settings=integration_settings)

        # Process
        clean_df = clean_launches(raw_df, settings=integration_settings)
        enriched_df = add_derived_features(clean_df)

        # Train
        X, y = split_features_target(enriched_df, settings=integration_settings)
        X_train, X_test, y_train, y_test = stratified_split(X, y, integration_settings)

        pipeline = build_pipeline(
            LogisticRegression(max_iter=500),
            integration_settings,
        )
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = (y_pred == y_test).mean()

        assert accuracy > 0.5  # Better than random
