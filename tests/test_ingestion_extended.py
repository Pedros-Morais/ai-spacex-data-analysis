"""Extended tests for data ingestion module.

Tests pure transformation functions and edge cases in the ingestion layer.
"""

from __future__ import annotations

import pytest

from launch_success.data.ingestion import (
    DATASET_COLUMNS,
    aggregate_payload_mass,
    build_lookup,
    launches_to_dataframe,
    parse_year,
    primary_orbit,
    resolve_launch,
    select_primary_core,
)
from launch_success.data.schemas import Core, Payload


# --------------------------------------------------------------------------- #
# aggregate_payload_mass tests
# --------------------------------------------------------------------------- #
class TestAggregatePayloadMass:
    """Tests for payload mass aggregation."""

    def test_empty_list_returns_none(self) -> None:
        assert aggregate_payload_mass([]) is None

    def test_all_none_masses_returns_none(self) -> None:
        payloads = [Payload(mass_kg=None), Payload(mass_kg=None)]
        assert aggregate_payload_mass(payloads) is None

    def test_single_payload(self) -> None:
        payloads = [Payload(mass_kg=1500.0)]
        assert aggregate_payload_mass(payloads) == 1500.0

    def test_multiple_payloads_sum(self) -> None:
        payloads = [
            Payload(mass_kg=1000.0),
            Payload(mass_kg=500.0),
            Payload(mass_kg=250.0),
        ]
        assert aggregate_payload_mass(payloads) == 1750.0

    def test_mixed_none_and_values(self) -> None:
        payloads = [
            Payload(mass_kg=1000.0),
            Payload(mass_kg=None),
            Payload(mass_kg=500.0),
        ]
        assert aggregate_payload_mass(payloads) == 1500.0

    def test_zero_mass(self) -> None:
        payloads = [Payload(mass_kg=0.0)]
        assert aggregate_payload_mass(payloads) == 0.0


# --------------------------------------------------------------------------- #
# primary_orbit tests
# --------------------------------------------------------------------------- #
class TestPrimaryOrbit:
    """Tests for orbit extraction."""

    def test_empty_list_returns_none(self) -> None:
        assert primary_orbit([]) is None

    def test_all_none_orbits(self) -> None:
        payloads = [Payload(orbit=None), Payload(orbit=None)]
        assert primary_orbit(payloads) is None

    def test_single_orbit(self) -> None:
        payloads = [Payload(orbit="LEO")]
        assert primary_orbit(payloads) == "LEO"

    def test_first_non_none_orbit(self) -> None:
        payloads = [
            Payload(orbit=None),
            Payload(orbit="GTO"),
            Payload(orbit="LEO"),
        ]
        assert primary_orbit(payloads) == "GTO"

    def test_various_orbit_types(self) -> None:
        orbits = ["LEO", "GTO", "MEO", "SSO", "ISS", "GEO", "HEO", "TLI"]
        for orbit in orbits:
            payloads = [Payload(orbit=orbit)]
            assert primary_orbit(payloads) == orbit


# --------------------------------------------------------------------------- #
# select_primary_core tests
# --------------------------------------------------------------------------- #
class TestSelectPrimaryCore:
    """Tests for primary core selection."""

    def test_empty_list_returns_none(self) -> None:
        assert select_primary_core([]) is None

    def test_single_core(self) -> None:
        core = Core(reused=True, flights=5)
        assert select_primary_core([core]) == core

    def test_multiple_cores_returns_first(self) -> None:
        cores = [
            Core(reused=False, flights=1),  # Primary/center core
            Core(reused=True, flights=3),  # Side booster
            Core(reused=True, flights=4),  # Side booster
        ]
        result = select_primary_core(cores)
        assert result == cores[0]
        assert result.reused is False
        assert result.flights == 1


# --------------------------------------------------------------------------- #
# parse_year tests
# --------------------------------------------------------------------------- #
class TestParseYear:
    """Tests for year parsing from ISO-8601 timestamps."""

    def test_none_returns_none(self) -> None:
        assert parse_year(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_year("") is None

    def test_short_string_returns_none(self) -> None:
        assert parse_year("202") is None

    def test_valid_iso8601(self) -> None:
        assert parse_year("2020-05-30T19:22:00.000Z") == 2020

    def test_various_years(self) -> None:
        assert parse_year("2006-03-24T22:30:00.000Z") == 2006
        assert parse_year("2015-12-22T01:29:00.000Z") == 2015
        assert parse_year("2024-01-01T00:00:00.000Z") == 2024

    def test_invalid_year_returns_none(self) -> None:
        assert parse_year("XXXX-01-01T00:00:00Z") is None

    def test_just_year(self) -> None:
        assert parse_year("2020") == 2020


# --------------------------------------------------------------------------- #
# build_lookup tests
# --------------------------------------------------------------------------- #
class TestBuildLookup:
    """Tests for building ID lookup dictionaries."""

    def test_empty_list(self) -> None:
        assert build_lookup([], "name") == {}

    def test_simple_lookup(self) -> None:
        items = [
            {"id": "a", "name": "Alpha"},
            {"id": "b", "name": "Beta"},
        ]
        lookup = build_lookup(items, "name")
        assert lookup == {"a": "Alpha", "b": "Beta"}

    def test_missing_id_skipped(self) -> None:
        items = [
            {"id": "a", "name": "Alpha"},
            {"name": "NoID"},  # No id key
        ]
        lookup = build_lookup(items, "name")
        assert lookup == {"a": "Alpha"}

    def test_missing_value_key(self) -> None:
        items = [{"id": "a", "other": "value"}]
        lookup = build_lookup(items, "name")
        assert lookup == {"a": None}

    def test_different_value_keys(self) -> None:
        items = [{"id": "r1", "type": "Rocket", "name": "Falcon 9"}]
        assert build_lookup(items, "name") == {"r1": "Falcon 9"}
        assert build_lookup(items, "type") == {"r1": "Rocket"}


# --------------------------------------------------------------------------- #
# resolve_launch tests
# --------------------------------------------------------------------------- #
class TestResolveLaunch:
    """Tests for resolving raw launch data into dataset rows."""

    @pytest.fixture
    def lookups(self) -> tuple[dict, dict, dict]:
        rocket_lookup = {"r_f9": "Falcon 9", "r_fh": "Falcon Heavy"}
        payload_lookup = {
            "p1": {"mass_kg": 1000.0, "orbit": "LEO"},
            "p2": {"mass_kg": 500.0, "orbit": "GTO"},
        }
        launchpad_lookup = {"lp_39a": "KSC LC 39A", "lp_40": "CCSFS SLC 40"}
        return rocket_lookup, payload_lookup, launchpad_lookup

    def test_complete_launch(self, lookups) -> None:
        rocket_lookup, payload_lookup, launchpad_lookup = lookups
        launch = {
            "flight_number": 100,
            "date_utc": "2020-05-30T19:22:00.000Z",
            "rocket": "r_f9",
            "success": True,
            "upcoming": False,
            "launchpad": "lp_39a",
            "payloads": ["p1", "p2"],
            "cores": [
                {
                    "reused": True,
                    "flights": 5,
                    "gridfins": True,
                    "legs": True,
                    "landing_success": True,
                }
            ],
        }
        row = resolve_launch(launch, rocket_lookup, payload_lookup, launchpad_lookup)

        assert row["flight_number"] == 100
        assert row["year"] == 2020
        assert row["rocket"] == "Falcon 9"
        assert row["payload_mass_kg"] == 1500.0  # 1000 + 500
        assert row["orbit"] == "LEO"  # First payload's orbit
        assert row["launch_site"] == "KSC LC 39A"
        assert row["reused"] is True
        assert row["flights"] == 5
        assert row["success"] is True
        assert row["upcoming"] is False

    def test_launch_with_no_payloads(self, lookups) -> None:
        rocket_lookup, payload_lookup, launchpad_lookup = lookups
        launch = {
            "flight_number": 1,
            "date_utc": "2020-01-01T00:00:00Z",
            "rocket": "r_f9",
            "success": True,
            "upcoming": False,
            "launchpad": "lp_40",
            "payloads": [],
            "cores": [],
        }
        row = resolve_launch(launch, rocket_lookup, payload_lookup, launchpad_lookup)

        assert row["payload_mass_kg"] is None
        assert row["orbit"] is None
        assert row["reused"] is None  # Default Core()
        assert row["flights"] is None

    def test_launch_with_unknown_rocket(self, lookups) -> None:
        rocket_lookup, payload_lookup, launchpad_lookup = lookups
        launch = {
            "flight_number": 1,
            "date_utc": "2020-01-01T00:00:00Z",
            "rocket": "unknown_rocket",
            "success": True,
            "upcoming": False,
            "launchpad": "lp_39a",
            "payloads": [],
            "cores": [],
        }
        row = resolve_launch(launch, rocket_lookup, payload_lookup, launchpad_lookup)
        assert row["rocket"] is None


# --------------------------------------------------------------------------- #
# launches_to_dataframe tests
# --------------------------------------------------------------------------- #
class TestLaunchesToDataframe:
    """Tests for converting API collections to DataFrame."""

    def test_empty_launches(self) -> None:
        df = launches_to_dataframe([], [], [], [])
        assert len(df) == 0
        # Should still have all columns
        for col in DATASET_COLUMNS:
            assert col in df.columns

    def test_single_launch(self) -> None:
        launches = [
            {
                "flight_number": 1,
                "date_utc": "2020-06-01T00:00:00Z",
                "rocket": "r1",
                "success": True,
                "upcoming": False,
                "launchpad": "lp1",
                "payloads": ["p1"],
                "cores": [{"reused": False, "flights": 1}],
            }
        ]
        rockets = [{"id": "r1", "name": "Falcon 9"}]
        payloads = [{"id": "p1", "mass_kg": 5000.0, "orbit": "ISS"}]
        launchpads = [{"id": "lp1", "name": "Kennedy Space Center"}]

        df = launches_to_dataframe(launches, rockets, payloads, launchpads)

        assert len(df) == 1
        assert df.iloc[0]["flight_number"] == 1
        assert df.iloc[0]["rocket"] == "Falcon 9"
        assert df.iloc[0]["payload_mass_kg"] == 5000.0
        assert df.iloc[0]["orbit"] == "ISS"

    def test_multiple_launches_preserves_order(self) -> None:
        launches = [
            {
                "flight_number": i,
                "date_utc": f"202{i}-01-01T00:00:00Z",
                "rocket": "r1",
                "success": True,
                "upcoming": False,
                "launchpad": "lp1",
                "payloads": [],
                "cores": [],
            }
            for i in range(1, 4)
        ]
        rockets = [{"id": "r1", "name": "Falcon 9"}]
        df = launches_to_dataframe(launches, rockets, [], [])

        assert len(df) == 3
        assert df["flight_number"].tolist() == [1, 2, 3]

    def test_columns_in_expected_order(self) -> None:
        launches = [
            {
                "flight_number": 1,
                "date_utc": "2020-01-01T00:00:00Z",
                "rocket": "r1",
                "success": True,
                "upcoming": False,
                "launchpad": "lp1",
                "payloads": [],
                "cores": [],
            }
        ]
        df = launches_to_dataframe(launches, [], [], [])

        expected_cols = list(DATASET_COLUMNS) + ["upcoming"]
        assert list(df.columns) == expected_cols
