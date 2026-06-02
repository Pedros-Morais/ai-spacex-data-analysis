"""Serialização e carregamento de pipelines treinados com ``joblib``.

Persiste o pipeline completo (pré-processamento + modelo) junto a metadados
úteis para o app de inferência (nome do modelo, alvo, features, métricas).
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
    """Salva o pipeline treinado (e metadados opcionais) em disco.

    Args:
        pipeline: Pipeline ajustado a persistir.
        path: Caminho de destino; usa ``settings.best_model_path`` se omitido.
        settings: Configuração (usa :data:`SETTINGS` se omitida).
        metadata: Dicionário serializável salvo ao lado do modelo (``*.meta.json``).

    Returns:
        O caminho onde o modelo foi salvo.
    """
    settings = settings or SETTINGS
    model_path = Path(path) if path is not None else settings.best_model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    if metadata is not None:
        meta_path = model_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Modelo salvo em %s", model_path)
    return model_path


def load_model(
    path: str | Path | None = None,
    settings: Settings | None = None,
) -> Pipeline:
    """Carrega um pipeline serializado.

    Args:
        path: Caminho do artefato; usa ``settings.best_model_path`` se omitido.
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        O pipeline desserializado.

    Raises:
        ModelNotFoundError: Se o arquivo não existir.
    """
    settings = settings or SETTINGS
    model_path = Path(path) if path is not None else settings.best_model_path
    if not model_path.exists():
        raise ModelNotFoundError(
            f"Modelo não encontrado em {model_path}. Rode o treino (`make train`)."
        )
    logger.info("Modelo carregado de %s", model_path)
    return joblib.load(model_path)


def load_metadata(
    path: str | Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Carrega os metadados associados a um modelo, se existirem.

    Args:
        path: Caminho do modelo (``*.joblib``); usa o padrão se omitido.
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        Dicionário de metadados, ou ``{}`` se não houver arquivo.
    """
    settings = settings or SETTINGS
    model_path = Path(path) if path is not None else settings.best_model_path
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
