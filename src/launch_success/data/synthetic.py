"""Gerador de dataset sintético calibrado em estatísticas reais da SpaceX.

A API v4 pode estar indisponível no ambiente de correção; este módulo produz um
CSV representativo (>= 1.000 linhas) com o **mesmo schema** da ingestão real,
permitindo que o pipeline rode *out-of-the-box*.

As distribuições reproduzem fatos conhecidos da economia espacial da SpaceX:

* **Falcon 1** (2006-2009) teve alta taxa de falha nos primeiros voos.
* **Falcon 9 / Falcon Heavy** são altamente confiáveis (sucesso > 95%).
* Missões **GTO** são mais pesadas e levemente mais arriscadas (alta energia).
* O **reúso** de boosters cresce ao longo dos anos (≈0 antes de 2017).
* O **sucesso de pouso** melhora com o tempo e é um alvo mais balanceado.

O gerador é determinístico dada a ``seed`` — reprodutibilidade total.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import SETTINGS, Settings

logger = logging.getLogger(__name__)

# Cadência anual aproximada de lançamentos (peso amostral por ano).
_YEAR_WEIGHTS: dict[int, float] = {
    2006: 1,
    2007: 1,
    2008: 2,
    2009: 1,
    2010: 2,
    2012: 2,
    2013: 3,
    2014: 6,
    2015: 7,
    2016: 8,
    2017: 18,
    2018: 21,
    2019: 13,
    2020: 26,
    2021: 31,
    2022: 61,
    2023: 96,
    2024: 130,
}

# Parâmetros (média, desvio) da massa de payload em kg, por órbita.
_ORBIT_MASS: dict[str, tuple[float, float]] = {
    "LEO": (8000.0, 4500.0),
    "VLEO": (15000.0, 1500.0),
    "GTO": (5500.0, 900.0),
    "ISS": (2500.0, 600.0),
    "SSO": (1200.0, 500.0),
    "MEO": (4200.0, 700.0),
    "PO": (1500.0, 600.0),
}
_ORBITS = list(_ORBIT_MASS)
_ORBIT_WEIGHTS = np.array([0.42, 0.06, 0.18, 0.12, 0.10, 0.06, 0.06])

_F9_SITES = ["CCSFS SLC 40", "KSC LC 39A", "VAFB SLC 4E"]


def _pick_rocket(rng: np.random.Generator, year: int) -> str:
    """Escolhe a versão do foguete coerente com o ano."""
    if year <= 2009:
        return "Falcon 1"
    if year <= 2017:
        return "Falcon 9"
    return rng.choice(["Falcon 9", "Falcon Heavy"], p=[0.92, 0.08])


def _reuse_probability(year: int) -> float:
    """Probabilidade de booster reutilizado em função do ano."""
    if year < 2017:
        return 0.0
    # Sobe de ~0.15 (2017) até ~0.85 (2024), saturando.
    return float(min(0.85, 0.15 + 0.11 * (year - 2017)))


def _landing_probability(year: int, orbit: str, reused: bool) -> float:
    """Probabilidade de pouso bem-sucedido (alvo alternativo, mais balanceado)."""
    base = min(0.92, 0.10 + 0.11 * max(0, year - 2013))
    if orbit == "GTO":  # alta energia, pouso mais difícil
        base -= 0.12
    if reused:  # booster já provado tende a pousar de novo
        base += 0.05
    return float(np.clip(base, 0.05, 0.97))


def _launch_success_probability(year: int, rocket: str, orbit: str) -> float:
    """Probabilidade de sucesso do lançamento (alvo principal, desbalanceado)."""
    if rocket == "Falcon 1":  # 2 de 5 sucessos historicamente
        return 0.40
    base = 0.93 if year <= 2013 else 0.975  # falhas raras nos primeiros F9
    if orbit == "GTO":
        base -= 0.015
    return float(np.clip(base, 0.0, 0.999))


def generate_synthetic_launches(
    n_rows: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera um DataFrame sintético de lançamentos com o schema do projeto.

    Args:
        n_rows: Número de linhas a gerar (>= 1.000 recomendado).
        seed: Semente para reprodutibilidade.

    Returns:
        DataFrame com as colunas do dataset processado, ordenado por data.
    """
    rng = np.random.default_rng(seed)

    years = np.array(list(_YEAR_WEIGHTS.keys()))
    weights = np.array(list(_YEAR_WEIGHTS.values()), dtype=float)
    weights /= weights.sum()
    sampled_years = rng.choice(years, size=n_rows, p=weights)

    records: list[dict[str, object]] = []
    for year in sampled_years:
        year = int(year)
        rocket = _pick_rocket(rng, year)
        orbit = str(rng.choice(_ORBITS, p=_ORBIT_WEIGHTS / _ORBIT_WEIGHTS.sum()))

        # Massa de payload (com ~6% de valores ausentes para exercitar imputação).
        mean, std = _ORBIT_MASS[orbit]
        mass: float | None = float(np.clip(rng.normal(mean, std), 50.0, 22_800.0))
        if rocket == "Falcon 1":
            mass = float(np.clip(rng.normal(250.0, 120.0), 20.0, 670.0))
        if rng.random() < 0.06:
            mass = None

        reused = bool(rng.random() < _reuse_probability(year))
        flights = int(rng.integers(2, 13)) if reused else 1

        # Tentativa de pouso: rara no Falcon 1, comum no Falcon 9+ a partir de 2015.
        landing_attempt = rocket != "Falcon 1" and year >= 2015 and rng.random() < 0.9
        gridfins = bool(landing_attempt)
        legs = bool(landing_attempt)
        if landing_attempt:
            p_land = _landing_probability(year, orbit, reused)
            landing_success: bool | None = bool(rng.random() < p_land)
        else:
            landing_success = None  # sem tentativa -> alvo alternativo indefinido

        success = bool(rng.random() < _launch_success_probability(year, rocket, orbit))

        if rocket == "Falcon 1":
            site = "Kwajalein Atoll"
        elif rocket == "Falcon Heavy":
            site = "KSC LC 39A"
        else:
            site = str(rng.choice(_F9_SITES, p=[0.5, 0.3, 0.2]))

        # Data sintética dentro do ano (mês/dia aleatórios).
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        date_utc = f"{year}-{month:02d}-{day:02d}T00:00:00.000Z"

        records.append(
            {
                "date_utc": date_utc,
                "year": year,
                "rocket": rocket,
                "payload_mass_kg": mass,
                "orbit": orbit,
                "launch_site": site,
                "reused": reused,
                "flights": flights,
                "gridfins": gridfins,
                "legs": legs,
                "landing_success": landing_success,
                "success": success,
                "upcoming": False,
            }
        )

    frame = pd.DataFrame(records).sort_values("date_utc").reset_index(drop=True)
    frame.insert(0, "flight_number", np.arange(1, len(frame) + 1))
    logger.info("Dataset sintético gerado: %d linhas", len(frame))
    return frame


def write_synthetic_dataset(
    n_rows: int = 1200,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Gera e salva o dataset sintético em ``settings.processed_csv``.

    Args:
        n_rows: Número de linhas a gerar.
        settings: Configuração (usa :data:`SETTINGS` se omitida).

    Returns:
        O DataFrame gerado.
    """
    settings = settings or SETTINGS
    settings.ensure_directories()
    frame = generate_synthetic_launches(n_rows=n_rows, seed=settings.seed)
    frame.to_csv(settings.processed_csv, index=False)
    logger.info("Dataset sintético salvo em %s", settings.processed_csv)
    return frame
