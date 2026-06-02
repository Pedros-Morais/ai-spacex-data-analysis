"""Modelos de domínio que representam entidades da API v4 da SpaceX.

Esses modelos isolam o parsing/validação do JSON cru da API das funções de
transformação puras (ex.: :func:`aggregate_payload_mass`), tornando-as fáceis
de testar com objetos simples em vez de dicionários soltos.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Payload(BaseModel):
    """Payload resolvido de um lançamento.

    Attributes:
        mass_kg: Massa do payload em quilogramas (``None`` se desconhecida).
        orbit: Sigla da órbita alvo (ex.: ``"LEO"``, ``"GTO"``).
    """

    model_config = ConfigDict(extra="ignore")

    mass_kg: float | None = None
    orbit: str | None = None


class Core(BaseModel):
    """Core (1º estágio) de um lançamento.

    Attributes:
        reused: Se o booster já havia voado antes.
        flights: Número de voos acumulados pelo core.
        gridfins: Presença de grid fins.
        legs: Presença de pernas de pouso.
        landing_success: Se o pouso/recuperação teve sucesso.
    """

    model_config = ConfigDict(extra="ignore")

    reused: bool | None = None
    flights: int | None = None
    gridfins: bool | None = None
    legs: bool | None = None
    landing_success: bool | None = None
