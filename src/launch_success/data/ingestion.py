"""Ingestão: busca os endpoints da API v4, resolve os ids e produz o dataset.

Estratégia: a API expõe coleções completas em ``/rockets``, ``/payloads`` e
``/launchpads``; construímos tabelas de lookup ``id -> valor`` e resolvemos
cada lançamento em **uma linha** com as ~12 features do projeto.

As funções de transformação (ex.: :func:`aggregate_payload_mass`) são puras e
testáveis isoladamente; o I/O (rede e disco) fica concentrado em :func:`ingest`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from ..config import SETTINGS, Settings
from ..exceptions import IngestionError
from .schemas import Core, Payload
from .spacex_client import SpaceXClient

logger = logging.getLogger(__name__)

# Colunas (na ordem) que o dataset processado deve conter.
DATASET_COLUMNS: tuple[str, ...] = (
    "flight_number",
    "date_utc",
    "year",
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
)


# --------------------------------------------------------------------------- #
# Funções de transformação puras
# --------------------------------------------------------------------------- #
def aggregate_payload_mass(payloads: list[Payload]) -> float | None:
    """Soma a massa (kg) de todos os payloads de um lançamento.

    Args:
        payloads: Payloads já resolvidos do lançamento.

    Returns:
        Massa total em kg, ou ``None`` se nenhum payload informar massa.
    """
    masses = [p.mass_kg for p in payloads if p.mass_kg is not None]
    return float(sum(masses)) if masses else None


def primary_orbit(payloads: list[Payload]) -> str | None:
    """Retorna a órbita do payload principal (o primeiro com órbita definida).

    Args:
        payloads: Payloads já resolvidos do lançamento.

    Returns:
        Sigla da órbita, ou ``None`` se nenhuma estiver definida.
    """
    for payload in payloads:
        if payload.orbit:
            return payload.orbit
    return None


def select_primary_core(cores: list[Core]) -> Core | None:
    """Seleciona o core principal de um lançamento.

    O Falcon Heavy possui 3 cores; por convenção usamos o **primeiro** da lista
    (core central/primário) para derivar ``reused``, ``flights`` etc. — escolha
    documentada no ``data/README.md``.

    Args:
        cores: Lista de cores do lançamento.

    Returns:
        O core principal, ou ``None`` se a lista estiver vazia.
    """
    return cores[0] if cores else None


def parse_year(date_utc: str | None) -> int | None:
    """Extrai o ano (UTC) de um timestamp ISO-8601 da API.

    Args:
        date_utc: Timestamp como ``"2006-03-24T22:30:00.000Z"``.

    Returns:
        O ano como inteiro, ou ``None`` se a data for inválida/ausente.
    """
    if not date_utc or len(date_utc) < 4:
        return None
    try:
        return int(date_utc[:4])
    except ValueError:
        return None


def build_lookup(items: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    """Constrói um dicionário ``id -> item[value_key]`` a partir de uma coleção.

    Args:
        items: Coleção de entidades da API (cada uma com a chave ``"id"``).
        value_key: Chave cujo valor será mapeado.

    Returns:
        Mapa de id para o valor correspondente.
    """
    return {item["id"]: item.get(value_key) for item in items if "id" in item}


def _resolve_payloads(
    payload_ids: list[str], payload_lookup: dict[str, dict[str, Any]]
) -> list[Payload]:
    """Resolve ids de payload em objetos :class:`Payload`."""
    resolved: list[Payload] = []
    for pid in payload_ids:
        raw = payload_lookup.get(pid)
        if raw is not None:
            resolved.append(Payload(mass_kg=raw.get("mass_kg"), orbit=raw.get("orbit")))
    return resolved


def resolve_launch(
    launch: dict[str, Any],
    rocket_lookup: dict[str, str],
    payload_lookup: dict[str, dict[str, Any]],
    launchpad_lookup: dict[str, str],
) -> dict[str, Any]:
    """Resolve um lançamento cru em uma linha do dataset (uma feature por chave).

    Args:
        launch: Objeto de lançamento cru da API.
        rocket_lookup: Mapa ``rocket_id -> nome``.
        payload_lookup: Mapa ``payload_id -> {mass_kg, orbit}``.
        launchpad_lookup: Mapa ``launchpad_id -> nome``.

    Returns:
        Dicionário com as colunas de :data:`DATASET_COLUMNS`.
    """
    payloads = _resolve_payloads(launch.get("payloads", []), payload_lookup)
    cores = [Core.model_validate(c) for c in launch.get("cores", [])]
    core = select_primary_core(cores) or Core()

    return {
        "flight_number": launch.get("flight_number"),
        "date_utc": launch.get("date_utc"),
        "year": parse_year(launch.get("date_utc")),
        "rocket": rocket_lookup.get(launch.get("rocket")),
        "payload_mass_kg": aggregate_payload_mass(payloads),
        "orbit": primary_orbit(payloads),
        "launch_site": launchpad_lookup.get(launch.get("launchpad")),
        "reused": core.reused,
        "flights": core.flights,
        "gridfins": core.gridfins,
        "legs": core.legs,
        "landing_success": core.landing_success,
        "success": launch.get("success"),
        # Mantido apenas para filtragem posterior; removido na limpeza.
        "upcoming": launch.get("upcoming", False),
    }


def launches_to_dataframe(
    launches: list[dict[str, Any]],
    rockets: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    launchpads: list[dict[str, Any]],
) -> pd.DataFrame:
    """Junta as quatro coleções da API em um DataFrame (uma linha por lançamento).

    Args:
        launches: Lançamentos crus.
        rockets: Foguetes crus.
        payloads: Payloads crus.
        launchpads: Launchpads crus.

    Returns:
        DataFrame com as colunas de :data:`DATASET_COLUMNS` (+ ``upcoming``).
    """
    rocket_lookup = build_lookup(rockets, "name")
    launchpad_lookup = build_lookup(launchpads, "name")
    payload_lookup = {p["id"]: p for p in payloads if "id" in p}

    rows = [
        resolve_launch(launch, rocket_lookup, payload_lookup, launchpad_lookup)
        for launch in launches
    ]
    frame = pd.DataFrame(rows)
    ordered = [*DATASET_COLUMNS, "upcoming"]
    # Garante todas as colunas mesmo se a API omitir alguma.
    for column in ordered:
        if column not in frame.columns:
            frame[column] = None
    return frame[ordered]


# --------------------------------------------------------------------------- #
# Orquestração de I/O
# --------------------------------------------------------------------------- #
def ingest(
    client: SpaceXClient | None = None,
    settings: Settings | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Busca a API, consolida o dataset e (opcionalmente) persiste em disco.

    Args:
        client: Cliente da API (injetável; criado se omitido).
        settings: Configuração (usa :data:`SETTINGS` se omitida).
        save: Se ``True``, salva o JSON cru e o CSV processado.

    Returns:
        DataFrame cru consolidado (antes da limpeza).

    Raises:
        IngestionError: Se qualquer endpoint falhar.
    """
    settings = settings or SETTINGS
    client = client or SpaceXClient(settings)

    logger.info("Iniciando ingestão da API v4 da SpaceX")
    launches = client.get_launches()
    rockets = client.get_rockets()
    payloads = client.get_payloads()
    launchpads = client.get_launchpads()

    if not launches:
        raise IngestionError("Endpoint /launches retornou uma lista vazia")

    frame = launches_to_dataframe(launches, rockets, payloads, launchpads)
    logger.info("Consolidados %d lançamentos", len(frame))

    if save:
        settings.ensure_directories()
        settings.raw_json.write_text(
            json.dumps(
                {
                    "launches": launches,
                    "rockets": rockets,
                    "payloads": payloads,
                    "launchpads": launchpads,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        frame.to_csv(settings.processed_csv, index=False)
        logger.info("Dataset salvo em %s", settings.processed_csv)

    return frame


def main() -> None:  # pragma: no cover - thin CLI wrapper
    """Ponto de entrada de linha de comando para a ingestão."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = ingest()
    print(f"Ingestão concluída: {len(frame)} lançamentos salvos em {SETTINGS.processed_csv}")


if __name__ == "__main__":  # pragma: no cover
    main()
