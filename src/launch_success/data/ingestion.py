"""Ingestion: fetches API v4 endpoints, resolves IDs, and produces the dataset.

Strategy: the API exposes full collections at ``/rockets``, ``/payloads``, and
``/launchpads``; we build ``id -> value`` lookup tables and resolve each launch
into **one row** with the ~12 project features.

The transformation functions (e.g. :func:`aggregate_payload_mass`) are pure and
testable in isolation; I/O (network and disk) is concentrated in :func:`ingest`.
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

# Columns (in order) that the processed dataset must contain.
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
# Pure transformation functions
# --------------------------------------------------------------------------- #
def aggregate_payload_mass(payloads: list[Payload]) -> float | None:
    """Sums the mass (kg) of all payloads for a launch.

    Args:
        payloads: Resolved payloads for the launch.

    Returns:
        Total mass in kg, or ``None`` if no payload reports a mass.
    """
    masses = [p.mass_kg for p in payloads if p.mass_kg is not None]
    return float(sum(masses)) if masses else None


def primary_orbit(payloads: list[Payload]) -> str | None:
    """Returns the orbit of the primary payload (the first one with a defined orbit).

    Args:
        payloads: Resolved payloads for the launch.

    Returns:
        Orbit code, or ``None`` if none is defined.
    """
    for payload in payloads:
        if payload.orbit:
            return payload.orbit
    return None


def select_primary_core(cores: list[Core]) -> Core | None:
    """Selects the primary core of a launch.

    Falcon Heavy has 3 cores; by convention we use the **first** in the list
    (the central/primary core) to derive ``reused``, ``flights``, etc. — a
    choice documented in ``data/README.md``.

    Args:
        cores: List of cores for the launch.

    Returns:
        The primary core, or ``None`` if the list is empty.
    """
    return cores[0] if cores else None


def parse_year(date_utc: str | None) -> int | None:
    """Extracts the year (UTC) from an ISO-8601 timestamp returned by the API.

    Args:
        date_utc: Timestamp such as ``"2006-03-24T22:30:00.000Z"``.

    Returns:
        The year as an integer, or ``None`` if the date is invalid or absent.
    """
    if not date_utc or len(date_utc) < 4:
        return None
    try:
        return int(date_utc[:4])
    except ValueError:
        return None


def build_lookup(items: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    """Builds an ``id -> item[value_key]`` dictionary from a collection.

    Args:
        items: Collection of API entities (each with an ``"id"`` key).
        value_key: Key whose value will be mapped.

    Returns:
        Map from id to the corresponding value.
    """
    return {item["id"]: item.get(value_key) for item in items if "id" in item}


def _resolve_payloads(
    payload_ids: list[str], payload_lookup: dict[str, dict[str, Any]]
) -> list[Payload]:
    """Resolves payload IDs into :class:`Payload` objects."""
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
    """Resolves a raw launch into a dataset row (one feature per key).

    Args:
        launch: Raw launch object from the API.
        rocket_lookup: Map ``rocket_id -> name``.
        payload_lookup: Map ``payload_id -> {mass_kg, orbit}``.
        launchpad_lookup: Map ``launchpad_id -> name``.

    Returns:
        Dictionary with the columns from :data:`DATASET_COLUMNS`.
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
        # Kept only for downstream filtering; removed during cleaning.
        "upcoming": launch.get("upcoming", False),
    }


def launches_to_dataframe(
    launches: list[dict[str, Any]],
    rockets: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    launchpads: list[dict[str, Any]],
) -> pd.DataFrame:
    """Joins the four API collections into a DataFrame (one row per launch).

    Args:
        launches: Raw launches.
        rockets: Raw rockets.
        payloads: Raw payloads.
        launchpads: Raw launchpads.

    Returns:
        DataFrame with the columns from :data:`DATASET_COLUMNS` (+ ``upcoming``).
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
    # Ensures all columns are present even if the API omits some.
    for column in ordered:
        if column not in frame.columns:
            frame[column] = None
    return frame[ordered]


# --------------------------------------------------------------------------- #
# I/O orchestration
# --------------------------------------------------------------------------- #
def ingest(
    client: SpaceXClient | None = None,
    settings: Settings | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Fetches the API, consolidates the dataset, and optionally persists it to disk.

    Args:
        client: API client (injectable; created if omitted).
        settings: Configuration (uses :data:`SETTINGS` if omitted).
        save: If ``True``, saves the raw JSON and the processed CSV.

    Returns:
        Consolidated raw DataFrame (before cleaning).

    Raises:
        IngestionError: If any endpoint fails.
    """
    settings = settings or SETTINGS
    client = client or SpaceXClient(settings)

    logger.info("Starting ingestion from the SpaceX API v4")
    launches = client.get_launches()
    rockets = client.get_rockets()
    payloads = client.get_payloads()
    launchpads = client.get_launchpads()

    if not launches:
        raise IngestionError("Endpoint /launches returned an empty list")

    frame = launches_to_dataframe(launches, rockets, payloads, launchpads)
    logger.info("Consolidated %d launches", len(frame))

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
        logger.info("Dataset saved to %s", settings.processed_csv)

    return frame


def main() -> None:  # pragma: no cover - thin CLI wrapper
    """Command-line entry point for ingestion."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = ingest()
    print(f"Ingestion complete: {len(frame)} launches saved to {SETTINGS.processed_csv}")


if __name__ == "__main__":  # pragma: no cover
    main()
