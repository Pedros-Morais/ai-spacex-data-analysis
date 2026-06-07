"""Serialisation and loading of trained pipelines with ``joblib``.

Persists the full pipeline (preprocessing + model) together with metadata
useful for the inference app (model name, target, features, metrics).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from ..config import SETTINGS, Settings
from ..exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


def save_model(
    pipeline: Pipeline,
    path: str | Path | None = None,
    settings: Settings | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Saves the trained pipeline (and optional metadata) to disk.

    Args:
        pipeline: Fitted pipeline to persist.
        path: Destination path; uses ``settings.best_model_path`` if omitted.
        settings: Configuration (uses :data:`SETTINGS` if omitted).
        metadata: Serialisable dictionary saved alongside the model (``*.meta.json``).

    Returns:
        The path where the model was saved.
    """
    settings = settings or SETTINGS
    model_path = Path(path) if path is not None else settings.best_model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    if metadata is not None:
        meta_path = model_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Model saved to %s", model_path)
    return model_path


def load_model(
    path: str | Path | None = None,
    settings: Settings | None = None,
) -> Pipeline:
    """Loads a serialised pipeline.

    Args:
        path: Path to the artefact; uses ``settings.best_model_path`` if omitted.
        settings: Configuration (uses :data:`SETTINGS` if omitted).

    Returns:
        The deserialised pipeline.

    Raises:
        ModelNotFoundError: If the file does not exist.
    """
    settings = settings or SETTINGS
    model_path = Path(path) if path is not None else settings.best_model_path
    if not model_path.exists():
        raise ModelNotFoundError(
            f"Model not found at {model_path}. Run training first (`make train`)."
        )
    logger.info("Model loaded from %s", model_path)
    return joblib.load(model_path)


def load_metadata(
    path: str | Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Loads the metadata associated with a model, if it exists.

    Args:
        path: Path to the model (``*.joblib``); uses the default if omitted.
        settings: Configuration (uses :data:`SETTINGS` if omitted).

    Returns:
        Metadata dictionary, or ``{}`` if no file is present.
    """
    settings = settings or SETTINGS
    model_path = Path(path) if path is not None else settings.best_model_path
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
