"""Tests for the transformation functions and ingestion orchestration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from launch_success.data.ingestion import (
    DATASET_COLUMNS,
    aggregate_payload_mass,
    build_lookup,
    ingest,
    launches_to_dataframe,
    parse_year,
    primary_orbit,
    resolve_launch,
    select_primary_core,
)
from launch_success.data.schemas import Core, Payload


def test_aggregate_payload_mass_ignores_nulls() -> None:
    payloads = [Payload(mass_kg=2000), Payload(mass_kg=None), Payload(mass_kg=500)]
    assert aggregate_payload_mass(payloads) == 2500


def test_aggregate_payload_mass_all_null_returns_none() -> None:
    assert aggregate_payload_mass([Payload(mass_kg=None)]) is None
    assert aggregate_payload_mass([]) is None


def test_primary_orbit_picks_first_defined() -> None:
    payloads = [Payload(orbit=None), Payload(orbit="GTO"), Payload(orbit="LEO")]
    assert primary_orbit(payloads) == "GTO"
    assert primary_orbit([Payload(orbit=None)]) is None


def test_select_primary_core() -> None:
    cores = [Core(reused=True), Core(reused=False)]
    assert select_primary_core(cores).reused is True
    assert select_primary_core([]) is None


def test_parse_year() -> None:
    assert parse_year("2020-05-30T19:22:00.000Z") == 2020
    assert parse_year(None) is None
    assert parse_year("xx") is None


def test_build_lookup() -> None:
    items = [{"id": "a", "name": "Falcon 9"}, {"id": "b", "name": "Falcon Heavy"}]
    assert build_lookup(items, "name") == {"a": "Falcon 9", "b": "Falcon Heavy"}


def test_resolve_launch_consolidates_one_row(mock_api: dict[str, list[dict[str, Any]]]) -> None:
    rocket_lookup = build_lookup(mock_api["rockets"], "name")
    launchpad_lookup = build_lookup(mock_api["launchpads"], "name")
    payload_lookup = {p["id"]: p for p in mock_api["payloads"]}

    row = resolve_launch(mock_api["launches"][0], rocket_lookup, payload_lookup, launchpad_lookup)
    assert row["rocket"] == "Falcon 9"
    assert row["payload_mass_kg"] == 2500.0  # 2000 + 500
    assert row["orbit"] == "LEO"
    assert row["launch_site"] == "KSC LC 39A"
    assert row["reused"] is False
    assert row["success"] is True
    assert row["year"] == 2020


def test_launches_to_dataframe_schema(mock_api: dict[str, list[dict[str, Any]]]) -> None:
    frame = launches_to_dataframe(
        mock_api["launches"],
        mock_api["rockets"],
        mock_api["payloads"],
        mock_api["launchpads"],
    )
    assert len(frame) == 3
    for column in DATASET_COLUMNS:
        assert column in frame.columns


class _FakeClient:
    """Fake client that returns the mocked JSON without touching the network."""

    def __init__(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self._data = data

    def get_launches(self) -> list[dict[str, Any]]:
        return self._data["launches"]

    def get_rockets(self) -> list[dict[str, Any]]:
        return self._data["rockets"]

    def get_payloads(self) -> list[dict[str, Any]]:
        return self._data["payloads"]

    def get_launchpads(self) -> list[dict[str, Any]]:
        return self._data["launchpads"]


def test_ingest_saves_csv_and_json(mock_api, tmp_settings) -> None:
    client = _FakeClient(mock_api)
    frame = ingest(client=client, settings=tmp_settings, save=True)

    assert isinstance(frame, pd.DataFrame)
    assert tmp_settings.processed_csv.exists()
    assert tmp_settings.raw_json.exists()
    reloaded = pd.read_csv(tmp_settings.processed_csv)
    assert len(reloaded) == len(frame) == 3
