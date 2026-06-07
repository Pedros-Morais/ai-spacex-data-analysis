"""Loading of the processed dataset (real API snapshot or fallback).

The CSV at ``data/processed/spacex_launches.csv`` may come from either the real
ingestion (:mod:`launch_success.data.ingestion`) or the versioned synthetic
generator (:mod:`launch_success.data.synthetic`). The loader is agnostic to the
source.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import SETTINGS, Settings
from ..exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Minimum columns that any valid dataset must expose.
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "flight_number",
        "rocket",
        "payload_mass_kg",
        "orbit",
        "launch_site",
        "reused",
        "flights",
        "gridfins",
        "legs",
        "landing_success",
        "success",
    }
)


def load_dataset(
    path: str | Path | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Loads the processed dataset from a CSV file.

    Args:
        path: Path to the CSV. If omitted, uses ``settings.processed_csv``.
        settings: Configuration (uses :data:`SETTINGS` if omitted).

    Returns:
        Raw DataFrame, before cleaning.

    Raises:
        DataValidationError: If the file does not exist, is empty, or required
            columns are missing.
    """
    settings = settings or SETTINGS
    csv_path = Path(path) if path is not None else settings.processed_csv

    if not csv_path.exists():
        raise DataValidationError(
            f"Dataset not found at {csv_path}. Run ingestion (`make ingest`) or generate the"
            f" fallback (`python scripts/generate_dataset.py`)."
        )

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise DataValidationError(f"Dataset at {csv_path} is empty.")

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DataValidationError(
            f"Dataset at {csv_path} is missing required columns: {sorted(missing)}"
        )

    logger.info("Dataset loaded from %s (%d rows)", csv_path, len(frame))
    return frame
