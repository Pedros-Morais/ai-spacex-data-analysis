"""Custom exceptions for the project.

Centralising exceptions makes error handling easier across the ingestion,
loading, and data validation layers, avoiding the use of the generic
``Exception`` and making the error flow explicit and testable.
"""

from __future__ import annotations


class LaunchSuccessError(Exception):
    """Base exception for all project errors."""


class IngestionError(LaunchSuccessError):
    """Error while fetching or consolidating data from the SpaceX API."""


class DataValidationError(LaunchSuccessError):
    """Data is missing, empty, or outside the expected schema."""


class ModelNotFoundError(LaunchSuccessError):
    """Attempted to load a model artifact that does not exist."""
