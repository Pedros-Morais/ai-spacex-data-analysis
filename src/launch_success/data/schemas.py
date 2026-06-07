"""Domain models representing entities from the SpaceX API v4.

These models isolate the parsing/validation of raw API JSON from the pure
transformation functions (e.g. :func:`aggregate_payload_mass`), making them
easy to test with simple objects instead of loose dictionaries.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Payload(BaseModel):
    """Resolved payload of a launch.

    Attributes:
        mass_kg: Payload mass in kilograms (``None`` if unknown).
        orbit: Target orbit code (e.g. ``"LEO"``, ``"GTO"``).
    """

    model_config = ConfigDict(extra="ignore")

    mass_kg: float | None = None
    orbit: str | None = None


class Core(BaseModel):
    """Core (first stage) of a launch.

    Attributes:
        reused: Whether the booster had flown before.
        flights: Total number of flights accumulated by the core.
        gridfins: Whether grid fins were present.
        legs: Whether landing legs were present.
        landing_success: Whether the landing/recovery was successful.
    """

    model_config = ConfigDict(extra="ignore")

    reused: bool | None = None
    flights: int | None = None
    gridfins: bool | None = None
    legs: bool | None = None
    landing_success: bool | None = None
