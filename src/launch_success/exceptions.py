"""Exceções customizadas do projeto.

Centralizar as exceções facilita o tratamento de erros nas camadas de
ingestão, carregamento e validação de dados, evitando o uso de ``Exception``
genérica e tornando o fluxo de erros explícito e testável.
"""

from __future__ import annotations


class LaunchSuccessError(Exception):
    """Exceção base para todos os erros do projeto."""


class IngestionError(LaunchSuccessError):
    """Erro ao buscar ou consolidar dados da API da SpaceX."""


class DataValidationError(LaunchSuccessError):
    """Dados ausentes, vazios ou fora do schema esperado."""


class ModelNotFoundError(LaunchSuccessError):
    """Tentativa de carregar um artefato de modelo inexistente."""
