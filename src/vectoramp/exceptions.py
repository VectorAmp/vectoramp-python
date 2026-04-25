"""VectorAmp SDK exceptions."""

from __future__ import annotations

from typing import Any


class VectorAmpError(Exception):
    """Base SDK exception."""


class AuthenticationError(VectorAmpError):
    """Raised when no API key is configured."""


class APIError(VectorAmpError):
    """Raised for non-successful API responses."""

    def __init__(self, status_code: int, message: str, *, response: Any = None) -> None:
        self.status_code = status_code
        self.response = response
        super().__init__(f"VectorAmp API error {status_code}: {message}")
