"""Extended tests for the SpaceX API client.

Tests HTTP handling, retry logic, and error conditions.
"""

from __future__ import annotations

import pytest
import responses

from launch_success.config import Settings
from launch_success.data.spacex_client import SpaceXClient
from launch_success.exceptions import IngestionError


@pytest.fixture
def client_settings() -> Settings:
    return Settings(
        api_base_url="https://api.spacexdata.com/v4",
        api_timeout=5.0,
        api_max_retries=2,
        api_backoff_factor=0.1,
    )


class TestSpaceXClientEndpoints:
    """Tests for individual API endpoints."""

    @responses.activate
    def test_get_launches_returns_list(self, client_settings) -> None:
        responses.add(
            responses.GET,
            "https://api.spacexdata.com/v4/launches",
            json=[{"id": "1", "name": "Test Launch"}],
            status=200,
        )

        client = SpaceXClient(client_settings)
        result = client.get_launches()

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "Test Launch"

    @responses.activate
    def test_get_rockets_returns_list(self, client_settings) -> None:
        responses.add(
            responses.GET,
            "https://api.spacexdata.com/v4/rockets",
            json=[{"id": "r1", "name": "Falcon 9"}],
            status=200,
        )

        client = SpaceXClient(client_settings)
        result = client.get_rockets()

        assert len(result) == 1
        assert result[0]["name"] == "Falcon 9"

    @responses.activate
    def test_get_payloads_returns_list(self, client_settings) -> None:
        responses.add(
            responses.GET,
            "https://api.spacexdata.com/v4/payloads",
            json=[{"id": "p1", "mass_kg": 1000}],
            status=200,
        )

        client = SpaceXClient(client_settings)
        result = client.get_payloads()

        assert len(result) == 1
        assert result[0]["mass_kg"] == 1000

    @responses.activate
    def test_get_launchpads_returns_list(self, client_settings) -> None:
        responses.add(
            responses.GET,
            "https://api.spacexdata.com/v4/launchpads",
            json=[{"id": "lp1", "name": "KSC LC-39A"}],
            status=200,
        )

        client = SpaceXClient(client_settings)
        result = client.get_launchpads()

        assert len(result) == 1
        assert result[0]["name"] == "KSC LC-39A"


class TestSpaceXClientErrors:
    """Tests for error handling."""

    @responses.activate
    def test_404_raises_ingestion_error(self, client_settings) -> None:
        responses.add(
            responses.GET,
            "https://api.spacexdata.com/v4/launches",
            json={"error": "Not Found"},
            status=404,
        )

        client = SpaceXClient(client_settings)
        with pytest.raises(IngestionError, match="Failed to fetch"):
            client.get_launches()

    @responses.activate
    def test_500_with_retries_exhausted(self, client_settings) -> None:
        # Add multiple 500 responses to exhaust retries
        for _ in range(5):
            responses.add(
                responses.GET,
                "https://api.spacexdata.com/v4/launches",
                json={"error": "Server Error"},
                status=500,
            )

        client = SpaceXClient(client_settings)
        with pytest.raises(IngestionError):
            client.get_launches()

    @responses.activate
    def test_invalid_json_response(self, client_settings) -> None:
        responses.add(
            responses.GET,
            "https://api.spacexdata.com/v4/launches",
            body="not valid json",
            status=200,
            content_type="text/plain",
        )

        client = SpaceXClient(client_settings)
        with pytest.raises(IngestionError, match="Failed to fetch"):
            client.get_launches()

    @responses.activate
    def test_connection_error(self, client_settings) -> None:
        from requests.exceptions import ConnectionError as RequestsConnectionError

        responses.add(
            responses.GET,
            "https://api.spacexdata.com/v4/launches",
            body=RequestsConnectionError("Connection failed"),
        )

        client = SpaceXClient(client_settings)
        with pytest.raises(IngestionError, match="Failed to fetch"):
            client.get_launches()

    @responses.activate
    def test_empty_response_body(self, client_settings) -> None:
        responses.add(
            responses.GET,
            "https://api.spacexdata.com/v4/launches",
            json=[],
            status=200,
        )

        client = SpaceXClient(client_settings)
        result = client.get_launches()
        assert result == []


class TestSpaceXClientSession:
    """Tests for session management."""

    def test_close_session(self, client_settings) -> None:
        client = SpaceXClient(client_settings)
        # Should not raise
        client.close()

    def test_custom_session_injection(self, client_settings) -> None:
        import requests

        custom_session = requests.Session()
        custom_session.headers["X-Custom"] = "test"

        client = SpaceXClient(client_settings, session=custom_session)
        assert client._session is custom_session

    @responses.activate
    def test_leading_slash_in_endpoint(self, client_settings) -> None:
        responses.add(
            responses.GET,
            "https://api.spacexdata.com/v4/launches",
            json=[],
            status=200,
        )

        client = SpaceXClient(client_settings)
        # Should handle leading slash gracefully
        result = client._get("/launches")
        assert result == []
