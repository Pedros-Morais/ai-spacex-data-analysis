"""Extended tests for configuration module.

Tests Settings class, path handling, and feature column configurations.
"""

from __future__ import annotations

from launch_success.config import PROJECT_ROOT, Settings


class TestSettingsDefaults:
    """Tests for default Settings values."""

    def test_default_seed(self) -> None:
        settings = Settings()
        assert settings.seed == 42

    def test_default_test_size(self) -> None:
        settings = Settings()
        assert settings.test_size == 0.2

    def test_default_cv_folds(self) -> None:
        settings = Settings()
        assert settings.cv_folds == 5

    def test_default_target(self) -> None:
        settings = Settings()
        assert settings.target == "success"

    def test_default_selection_metric(self) -> None:
        settings = Settings()
        assert settings.selection_metric == "f1"

    def test_default_use_smote(self) -> None:
        settings = Settings()
        assert settings.use_smote is False


class TestSettingsPaths:
    """Tests for path-related settings."""

    def test_data_dir_can_be_created(self, tmp_path) -> None:
        settings = Settings(
            data_dir=tmp_path / "data",
            raw_dir=tmp_path / "data" / "raw",
            processed_dir=tmp_path / "data" / "processed",
            models_dir=tmp_path / "models",
            figures_dir=tmp_path / "figures",
        )
        settings.ensure_directories()
        assert settings.raw_dir.exists()
        assert settings.processed_dir.exists()

    def test_processed_csv_path(self, tmp_path) -> None:
        settings = Settings(processed_dir=tmp_path / "processed")
        expected = tmp_path / "processed" / "spacex_launches.csv"
        assert settings.processed_csv == expected

    def test_raw_json_path(self, tmp_path) -> None:
        settings = Settings(raw_dir=tmp_path / "raw")
        expected = tmp_path / "raw" / "spacex_launches_raw.json"
        assert settings.raw_json == expected

    def test_best_model_path(self, tmp_path) -> None:
        settings = Settings(models_dir=tmp_path / "models")
        expected = tmp_path / "models" / "best_model.joblib"
        assert settings.best_model_path == expected

    def test_ensure_directories_creates_all(self, tmp_path) -> None:
        settings = Settings(
            data_dir=tmp_path / "data",
            raw_dir=tmp_path / "data" / "raw",
            processed_dir=tmp_path / "data" / "processed",
            models_dir=tmp_path / "models",
            figures_dir=tmp_path / "figures",
        )
        settings.ensure_directories()

        assert settings.data_dir.exists()
        assert settings.raw_dir.exists()
        assert settings.processed_dir.exists()
        assert settings.models_dir.exists()
        assert settings.figures_dir.exists()


class TestSettingsFeatures:
    """Tests for feature configuration."""

    def test_feature_columns_property(self) -> None:
        settings = Settings()
        features = settings.feature_columns

        assert isinstance(features, list)
        assert len(features) > 0

        # Should include all feature types
        for feat in settings.numeric_features:
            assert feat in features
        for feat in settings.categorical_features:
            assert feat in features
        for feat in settings.boolean_features:
            assert feat in features

    def test_default_numeric_features(self) -> None:
        settings = Settings()
        assert "payload_mass_kg" in settings.numeric_features
        assert "year" in settings.numeric_features
        assert "flights" in settings.numeric_features

    def test_default_categorical_features(self) -> None:
        settings = Settings()
        assert "rocket" in settings.categorical_features
        assert "orbit" in settings.categorical_features
        assert "launch_site" in settings.categorical_features

    def test_default_boolean_features(self) -> None:
        settings = Settings()
        assert "reused" in settings.boolean_features
        assert "gridfins" in settings.boolean_features
        assert "legs" in settings.boolean_features

    def test_custom_features(self) -> None:
        settings = Settings(
            numeric_features=["mass"],
            categorical_features=["type"],
            boolean_features=["flag"],
        )
        assert settings.feature_columns == ["mass", "type", "flag"]


class TestSettingsAPI:
    """Tests for API configuration."""

    def test_default_api_base_url(self) -> None:
        settings = Settings()
        assert "spacexdata.com" in settings.api_base_url

    def test_default_api_timeout(self) -> None:
        settings = Settings()
        assert settings.api_timeout > 0

    def test_default_api_retries(self) -> None:
        settings = Settings()
        assert settings.api_max_retries >= 0

    def test_custom_api_settings(self) -> None:
        settings = Settings(
            api_base_url="https://custom.api.com/v5",
            api_timeout=60.0,
            api_max_retries=5,
            api_backoff_factor=2.0,
        )
        assert settings.api_base_url == "https://custom.api.com/v5"
        assert settings.api_timeout == 60.0
        assert settings.api_max_retries == 5
        assert settings.api_backoff_factor == 2.0


class TestSettingsSelectionMetric:
    """Tests for selection metric configuration."""

    def test_default_selection_metric_is_f1(self) -> None:
        settings = Settings()
        assert settings.selection_metric == "f1"

    def test_selection_metric_can_be_changed(self) -> None:
        settings = Settings(selection_metric="roc_auc")
        assert settings.selection_metric == "roc_auc"

    def test_valid_selection_metrics(self) -> None:
        valid_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        for metric in valid_metrics:
            settings = Settings(selection_metric=metric)
            assert settings.selection_metric == metric


class TestProjectRoot:
    """Tests for project root constant."""

    def test_project_root_exists(self) -> None:
        assert PROJECT_ROOT.exists()

    def test_project_root_is_directory(self) -> None:
        assert PROJECT_ROOT.is_dir()

    def test_project_root_contains_src(self) -> None:
        src_dir = PROJECT_ROOT / "src"
        assert src_dir.exists()


class TestSettingsEnvironmentOverride:
    """Tests for environment variable overrides."""

    def test_seed_override(self, monkeypatch) -> None:
        monkeypatch.setenv("LAUNCH_SEED", "123")
        settings = Settings()
        assert settings.seed == 123

    def test_test_size_override(self, monkeypatch) -> None:
        monkeypatch.setenv("LAUNCH_TEST_SIZE", "0.3")
        settings = Settings()
        assert settings.test_size == 0.3

    def test_cv_folds_override(self, monkeypatch) -> None:
        monkeypatch.setenv("LAUNCH_CV_FOLDS", "10")
        settings = Settings()
        assert settings.cv_folds == 10

    def test_selection_metric_override(self, monkeypatch) -> None:
        monkeypatch.setenv("LAUNCH_SELECTION_METRIC", "roc_auc")
        settings = Settings()
        assert settings.selection_metric == "roc_auc"

    def test_use_smote_override(self, monkeypatch) -> None:
        monkeypatch.setenv("LAUNCH_USE_SMOTE", "true")
        settings = Settings()
        assert settings.use_smote is True
