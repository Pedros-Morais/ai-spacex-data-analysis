"""Cliente HTTP para a API pública v4 da SpaceX.

Encapsula uma :class:`requests.Session` com timeout, retry exponencial e
tratamento de erro, expondo um método por endpoint usado pela ingestão.
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

# Métodos idempotentes para os quais o retry é seguro.
_RETRY_METHODS = frozenset({"GET"})
# Status transitórios que justificam nova tentativa.
_RETRY_STATUS = (429, 500, 502, 503, 504)


class SpaceXClient:
    """Cliente fino sobre a API v4 da SpaceX.

    Args:
        settings: Configuração com URL base, timeout e política de retry.
        session: Sessão HTTP opcional (injetável para testes).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._settings = settings or SETTINGS
        self._session = session or self._build_session()

    def _build_session(self) -> requests.Session:
        """Cria uma sessão com adapter de retry configurado."""
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
        """Executa um GET e devolve o JSON desserializado.

        Args:
            endpoint: Caminho relativo à URL base (ex.: ``"launches"``).

        Returns:
            Corpo JSON da resposta (lista ou dicionário).

        Raises:
            IngestionError: Em erro de rede, timeout ou status HTTP inválido.
        """
        url = f"{self._settings.api_base_url}/{endpoint.lstrip('/')}"
        logger.info("GET %s", url)
        try:
            response = self._session.get(url, timeout=self._settings.api_timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:  # cobre timeout, conexão, HTTP
            raise IngestionError(f"Falha ao buscar {url}: {exc}") from exc
        except ValueError as exc:  # JSON inválido
            raise IngestionError(f"Resposta não-JSON de {url}: {exc}") from exc

    def get_launches(self) -> list[dict[str, Any]]:
        """Retorna todos os lançamentos (``GET /launches``)."""
        return self._get("launches")

    def get_rockets(self) -> list[dict[str, Any]]:
        """Retorna todos os foguetes (``GET /rockets``)."""
        return self._get("rockets")

    def get_payloads(self) -> list[dict[str, Any]]:
        """Retorna todos os payloads (``GET /payloads``)."""
        return self._get("payloads")

    def get_launchpads(self) -> list[dict[str, Any]]:
        """Retorna todos os locais de lançamento (``GET /launchpads``)."""
        return self._get("launchpads")

    def close(self) -> None:
        """Fecha a sessão HTTP subjacente."""
        self._session.close()
