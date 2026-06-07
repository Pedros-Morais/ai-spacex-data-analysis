"""HTTP client for the SpaceX public API v4.

Wraps a :class:`requests.Session` with timeout, exponential retry, and error
handling, exposing one method per endpoint used by the ingestion layer.
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import SETTINGS, Settings
from ..exceptions import IngestionError

logger = logging.getLogger(__name__)

# Idempotent methods for which retrying is safe.
_RETRY_METHODS = frozenset({"GET"})
# Transient status codes that warrant a retry.
_RETRY_STATUS = (429, 500, 502, 503, 504)


class SpaceXClient:
    """Thin client over the SpaceX API v4.

    Args:
        settings: Configuration with base URL, timeout, and retry policy.
        session: Optional HTTP session (injectable for testing).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._settings = settings or SETTINGS
        self._session = session or self._build_session()

    def _build_session(self) -> requests.Session:
        """Creates a session with a configured retry adapter."""
        retry = Retry(
            total=self._settings.api_max_retries,
            backoff_factor=self._settings.api_backoff_factor,
            status_forcelist=_RETRY_STATUS,
            allowed_methods=_RETRY_METHODS,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get(self, endpoint: str) -> Any:
        """Performs a GET request and returns the deserialised JSON.

        Args:
            endpoint: Path relative to the base URL (e.g. ``"launches"``).

        Returns:
            JSON response body (list or dictionary).

        Raises:
            IngestionError: On network error, timeout, or invalid HTTP status.
        """
        url = f"{self._settings.api_base_url}/{endpoint.lstrip('/')}"
        logger.info("GET %s", url)
        try:
            response = self._session.get(url, timeout=self._settings.api_timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:  # covers timeout, connection, HTTP
            raise IngestionError(f"Failed to fetch {url}: {exc}") from exc
        except ValueError as exc:  # invalid JSON
            raise IngestionError(f"Non-JSON response from {url}: {exc}") from exc

    def get_launches(self) -> list[dict[str, Any]]:
        """Returns all launches (``GET /launches``)."""
        return self._get("launches")

    def get_rockets(self) -> list[dict[str, Any]]:
        """Returns all rockets (``GET /rockets``)."""
        return self._get("rockets")

    def get_payloads(self) -> list[dict[str, Any]]:
        """Returns all payloads (``GET /payloads``)."""
        return self._get("payloads")

    def get_launchpads(self) -> list[dict[str, Any]]:
        """Returns all launch sites (``GET /launchpads``)."""
        return self._get("launchpads")

    def close(self) -> None:
        """Closes the underlying HTTP session."""
        self._session.close()
