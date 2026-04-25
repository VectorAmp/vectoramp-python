"""Transport abstraction for VectorAmp.

The public client depends on this small interface rather than directly on httpx.  REST is the
current implementation; a future gRPC transport can implement the same request/stream surface.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Mapping, Optional
from urllib.parse import urljoin

import httpx

from .exceptions import APIError, AuthenticationError


class BaseTransport(ABC):
    """Minimal transport contract used by resource clients."""

    @abstractmethod
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Any:
        """Send one request and return decoded response content."""

    @abstractmethod
    def stream(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield JSON-decoded Server-Sent Event data chunks."""

    @abstractmethod
    def close(self) -> None:
        """Release transport resources."""


class RestTransport(BaseTransport):
    """httpx-backed REST transport."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.vectoramp.com",
        timeout: float = 30.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        if not api_key:
            raise AuthenticationError(
                "An API key is required. Pass api_key or set VECTORAMP_API_KEY."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Any:
        response = self.client.request(
            method,
            self._url(path),
            params=self._clean(params),
            json=json_body,
            headers=self._headers(headers),
        )
        return self._decode_response(response)

    def stream(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        request_headers = self._headers({"Accept": "text/event-stream", **dict(headers or {})})
        with self.client.stream(
            method, self._url(path), json=json_body, headers=request_headers
        ) as response:
            self._raise_for_status(response)
            for line in response.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    yield json.loads(payload)

    def put_url(
        self,
        url: str,
        *,
        content: bytes,
        content_type: Optional[str] = None,
    ) -> None:
        """Upload bytes to a presigned URL without VectorAmp auth headers."""
        headers = {"Content-Type": content_type} if content_type else None
        response = self.client.put(url, content=content, headers=headers)
        self._raise_for_status(response)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def _headers(self, headers: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        merged = {"X-API-Key": self.api_key, "User-Agent": "vectoramp-python/0.1.0"}
        if headers:
            merged.update(headers)
        return merged

    @staticmethod
    def _clean(params: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if params is None:
            return None
        return {key: value for key, value in params.items() if value is not None}

    @classmethod
    def _decode_response(cls, response: httpx.Response) -> Any:
        cls._raise_for_status(response)
        if response.status_code == 204 or not response.content:
            return None
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        try:
            return response.json()
        except ValueError:
            return response.text

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            payload = response.json()
            message = (
                payload.get("message")
                or payload.get("detail")
                or payload.get("error")
                or str(payload)
            )
        except ValueError:
            message = response.text
        raise APIError(response.status_code, message, response=response)
