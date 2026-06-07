"""Centralised project configuration.

All "magic" constants (paths, seed, hyperparameters, feature lists, target)
live here. Other modules import :data:`SETTINGS` or construct a
:class:`Settings` instance, ensuring reproducibility and a single source of
truth for the configuration.

Settings can be overridden by environment variables with the prefix
``LAUNCH_`` (e.g. ``LAUNCH_TARGET=landing_success``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: this file is src/launch_success/config.py.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

TargetName = Literal["success", "landing_success"]
SearchStrategy = Literal["random", "grid"]


class Settings(BaseSettings):
    """Typed and validated pipeline configuration.

    Attributes:
        seed: Global seed for reproducibility.
        target: Target column for binary classification.
        test_size: Fraction held out for testing in the stratified split.
        cv_folds: Number of StratifiedKFold folds.
        selection_metric: Metric used to select the best model.
        use_smote: If ``True``, applies SMOTE only on the training fold.
        tune_hyperparameters: If ``True``, runs a cross-validated hyperparameter
            search per model before the final fit.
        search_strategy: ``"random"`` (RandomizedSearchCV) or ``"grid"``
            (GridSearchCV).
        search_n_iter: Sampled candidates for the randomized search.
    """

    model_config = SettingsConfigDict(
        env_prefix="LAUNCH_",
        env_file=".env",
        extra="ignore",
    )

    # --- Reproducibility / problem ---------------------------------------- #
    seed: int = 42
    target: TargetName = "success"
    test_size: float = 0.2
    cv_folds: int = 5
    selection_metric: str = "f1"
    use_smote: bool = False

    # --- Hyperparameter search -------------------------------------------- #
    # Disabled by default to keep the test suite / CI fast; enable with
    # LAUNCH_TUNE_HYPERPARAMETERS=true for a tuned run.
    tune_hyperparameters: bool = False
    search_strategy: SearchStrategy = "random"
    search_n_iter: int = 15

    # --- SpaceX API v4 ---------------------------------------------------- #
    api_base_url: str = "https://api.spacexdata.com/v4"
    api_timeout: float = 30.0
    api_max_retries: int = 3
    api_backoff_factor: float = 0.5

    # --- Paths ------------------------------------------------------------ #
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"

    # --- Feature schema --------------------------------------------------- #
    numeric_features: list[str] = Field(
        default_factory=lambda: ["flight_number", "year", "payload_mass_kg", "flights"]
    )
    categorical_features: list[str] = Field(
        default_factory=lambda: ["rocket", "orbit", "launch_site"]
    )
    boolean_features: list[str] = Field(default_factory=lambda: ["reused", "gridfins", "legs"])

    @property
    def feature_columns(self) -> list[str]:
        """Full list of columns used as model input."""
        return [*self.numeric_features, *self.categorical_features, *self.boolean_features]

    @property
    def processed_csv(self) -> Path:
        """Path to the processed CSV (real snapshot or fallback)."""
        return self.processed_dir / "spacex_launches.csv"

    @property
    def raw_json(self) -> Path:
        """Path to the consolidated raw JSON from the API."""
        return self.raw_dir / "spacex_launches_raw.json"

    @property
    def best_model_path(self) -> Path:
        """Path to the serialised winning pipeline."""
        return self.models_dir / "best_model.joblib"

    @property
    def metrics_path(self) -> Path:
        """Path to the JSON file containing the model metrics comparison."""
        return self.models_dir / "metrics.json"

    def ensure_directories(self) -> None:
        """Creates output directories if they do not already exist."""
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# Default instance reusable across the entire project.
SETTINGS = Settings()
