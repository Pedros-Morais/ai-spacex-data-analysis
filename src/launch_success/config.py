"""Configuração centralizada do projeto.

Todas as constantes "mágicas" (caminhos, seed, hiperparâmetros, listas de
features, alvo) vivem aqui. Outros módulos importam :data:`SETTINGS` ou
constroem uma instância de :class:`Settings`, garantindo reprodutibilidade e
um único ponto de verdade para a configuração.

As configurações podem ser sobrescritas por variáveis de ambiente com o
prefixo ``LAUNCH_`` (ex.: ``LAUNCH_TARGET=landing_success``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto: este arquivo é src/launch_success/config.py.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

TargetName = Literal["success", "landing_success"]


class Settings(BaseSettings):
    """Configuração tipada e validada do pipeline.

    Attributes:
        seed: Semente global para reprodutibilidade.
        target: Coluna alvo da classificação binária.
        test_size: Fração reservada para teste no split estratificado.
        cv_folds: Número de folds do StratifiedKFold.
        selection_metric: Métrica usada para escolher o melhor modelo.
        use_smote: Se ``True``, aplica SMOTE apenas no fold de treino.
    """

    model_config = SettingsConfigDict(
        env_prefix="LAUNCH_",
        env_file=".env",
        extra="ignore",
    )

    # --- Reprodutibilidade / problema ----------------------------------- #
    seed: int = 42
    target: TargetName = "success"
    test_size: float = 0.2
    cv_folds: int = 5
    selection_metric: str = "f1"
    use_smote: bool = False

    # --- API SpaceX v4 -------------------------------------------------- #
    api_base_url: str = "https://api.spacexdata.com/v4"
    api_timeout: float = 30.0
    api_max_retries: int = 3
    api_backoff_factor: float = 0.5

    # --- Caminhos ------------------------------------------------------- #
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"

    # --- Schema de features --------------------------------------------- #
    numeric_features: list[str] = Field(
        default_factory=lambda: ["flight_number", "year", "payload_mass_kg", "flights"]
    )
    categorical_features: list[str] = Field(
        default_factory=lambda: ["rocket", "orbit", "launch_site"]
    )
    boolean_features: list[str] = Field(default_factory=lambda: ["reused", "gridfins", "legs"])

    @property
    def feature_columns(self) -> list[str]:
        """Lista completa de colunas usadas como entrada do modelo."""
        return [*self.numeric_features, *self.categorical_features, *self.boolean_features]

    @property
    def processed_csv(self) -> Path:
        """Caminho do CSV processado (snapshot real ou fallback)."""
        return self.processed_dir / "spacex_launches.csv"

    @property
    def raw_json(self) -> Path:
        """Caminho do JSON cru consolidado da API."""
        return self.raw_dir / "spacex_launches_raw.json"

    @property
    def best_model_path(self) -> Path:
        """Caminho do pipeline vencedor serializado."""
        return self.models_dir / "best_model.joblib"

    @property
    def metrics_path(self) -> Path:
        """Caminho do JSON com a comparação de métricas dos modelos."""
        return self.models_dir / "metrics.json"

    def ensure_directories(self) -> None:
        """Cria os diretórios de saída, se ainda não existirem."""
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# Instância padrão reutilizável em todo o projeto.
SETTINGS = Settings()
