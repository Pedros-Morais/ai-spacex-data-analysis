"""Tests for custom exception classes.

Ensures exception hierarchy and messages work correctly.
"""

from __future__ import annotations

import pytest

from launch_success.exceptions import (
    DataValidationError,
    IngestionError,
    LaunchSuccessError,
    ModelNotFoundError,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_ingestion_error_is_launch_success_error(self) -> None:
        exc = IngestionError("test")
        assert isinstance(exc, LaunchSuccessError)
        assert isinstance(exc, Exception)

    def test_data_validation_error_is_launch_success_error(self) -> None:
        exc = DataValidationError("test")
        assert isinstance(exc, LaunchSuccessError)
        assert isinstance(exc, Exception)

    def test_model_not_found_error_is_launch_success_error(self) -> None:
        exc = ModelNotFoundError("test")
        assert isinstance(exc, LaunchSuccessError)
        assert isinstance(exc, Exception)

    def test_base_error_is_exception(self) -> None:
        exc = LaunchSuccessError("test")
        assert isinstance(exc, Exception)


class TestExceptionMessages:
    """Tests for exception message handling."""

    def test_launch_success_error_message(self) -> None:
        exc = LaunchSuccessError("Base error message")
        assert str(exc) == "Base error message"

    def test_ingestion_error_message(self) -> None:
        exc = IngestionError("Failed to fetch data")
        assert str(exc) == "Failed to fetch data"

    def test_data_validation_error_message(self) -> None:
        exc = DataValidationError("Invalid data format")
        assert str(exc) == "Invalid data format"

    def test_model_not_found_error_message(self) -> None:
        exc = ModelNotFoundError("Model file missing")
        assert str(exc) == "Model file missing"


class TestExceptionRaising:
    """Tests for raising and catching exceptions."""

    def test_catch_ingestion_error_as_base(self) -> None:
        with pytest.raises(LaunchSuccessError):
            raise IngestionError("test")

    def test_catch_data_validation_error_as_base(self) -> None:
        with pytest.raises(LaunchSuccessError):
            raise DataValidationError("test")

    def test_catch_model_not_found_error_as_base(self) -> None:
        with pytest.raises(LaunchSuccessError):
            raise ModelNotFoundError("test")

    def test_catch_specific_exception(self) -> None:
        with pytest.raises(IngestionError):
            raise IngestionError("specific")

        with pytest.raises(DataValidationError):
            raise DataValidationError("specific")

        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("specific")


class TestExceptionInContext:
    """Tests for exceptions in realistic contexts."""

    def test_ingestion_error_with_url(self) -> None:
        url = "https://api.spacexdata.com/v4/launches"
        exc = IngestionError(f"Failed to fetch {url}: Connection timeout")
        assert url in str(exc)
        assert "timeout" in str(exc).lower()

    def test_data_validation_error_with_details(self) -> None:
        missing_cols = ["rocket", "payload_mass_kg"]
        exc = DataValidationError(f"Missing required columns: {missing_cols}")
        assert "rocket" in str(exc)
        assert "payload_mass_kg" in str(exc)

    def test_model_not_found_with_path(self) -> None:
        path = "/path/to/model.joblib"
        exc = ModelNotFoundError(f"Model not found at {path}")
        assert path in str(exc)

    def test_chained_exception(self) -> None:
        try:
            try:
                raise ConnectionError("Network failed")
            except ConnectionError as e:
                raise IngestionError("API call failed") from e
        except IngestionError as exc:
            assert exc.__cause__ is not None
            assert isinstance(exc.__cause__, ConnectionError)
