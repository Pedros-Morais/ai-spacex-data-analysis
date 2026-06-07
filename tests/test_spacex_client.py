"""Tests for the API v4 HTTP client (no real network — mocked responses)."""

from __future__ import annotations

import pytest
import requests
import responses

from launch_success.config import Settings
from launch_success.data.spacex_client import SpaceXClient
from launch_success.exceptions import IngestionError

BASE = "https://api.spacexdata.com/v4"


@pytest.fixture
def client() -> SpaceXClient:
    # No retries to keep tests fast and deterministic.
    return SpaceXClient(Settings(api_max_retries=0))


@responses.activate
def test_get_launches_parses_json(client: SpaceXClient) -> None:
    payload = [{"flight_number": 1, "success": True}]
    responses.add(responses.GET, f"{BASE}/launches", json=payload, status=200)
    assert client.get_launches() == payload


@responses.activate
def test_each_endpoint_hits_correct_url(client: SpaceXClient) -> None:
    for endpoint in ("rockets", "payloads", "launchpads"):
        responses.add(responses.GET, f"{BASE}/{endpoint}", json=[{"id": endpoint}], status=200)
    assert client.get_rockets() == [{"id": "rockets"}]
    assert client.get_payloads() == [{"id": "payloads"}]
    assert client.get_launchpads() == [{"id": "launchpads"}]


@responses.activate
def test_http_error_raises_ingestion_error(client: SpaceXClient) -> None:
    responses.add(responses.GET, f"{BASE}/launches", status=500)
    with pytest.raises(IngestionError):
        client.get_launches()


@responses.activate
def test_connection_error_raises_ingestion_error(client: SpaceXClient) -> None:
    responses.add(
        responses.GET,
        f"{BASE}/launches",
        body=requests.exceptions.ConnectionError("timeout"),
    )
    with pytest.raises(IngestionError):
        client.get_launches()


@responses.activate
def test_invalid_json_raises_ingestion_error(client: SpaceXClient) -> None:
    responses.add(responses.GET, f"{BASE}/launches", body="not-json", status=200)
    with pytest.raises(IngestionError):
        client.get_launches()
