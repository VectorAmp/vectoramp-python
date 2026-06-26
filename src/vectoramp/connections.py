"""OAuth connection management for the VectorAmp SDK.

Connections hold the stored OAuth grant for a provider (Google Drive, Atlassian,
etc.). They live at the gateway root ``/connections`` namespace (not under
``/ingestion``) and are referenced from a source via ``connection_id``.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from .transport import BaseTransport
from .types import JSON


def _print_authorization_url(url: str) -> None:
    """Default ``on_url`` handler: print the authorization URL and instructions."""
    print(
        "To authorize this connection, open the following URL in your browser:\n"
        f"  {url}\n"
        "Waiting for authorization to complete..."
    )


class ConnectionsResource:
    """OAuth connection management.

    A connection captures the OAuth authorization for a provider so that
    ingestion sources can reuse it via ``connection_id`` instead of embedding
    raw credentials. Use :meth:`connect` for the full interactive
    create-authorize-poll flow, or the lower-level CRUD methods directly.
    """

    def __init__(self, transport: BaseTransport) -> None:
        self._transport = transport

    def list(self, *, provider: Optional[str] = None) -> JSON:
        """List connections for the calling organization.

        Args:
            provider: Optional provider filter (e.g. ``"google_drive"``).

        Returns:
            Connection list JSON from ``/connections``.
        """
        return self._transport.request("GET", "/connections", params={"provider": provider})

    def create(self, provider: str, *, source_type: Optional[str] = None) -> JSON:
        """Create a pending connection and return its authorization URL.

        Args:
            provider: OAuth provider identifier (e.g. ``"google_drive"``).
            source_type: Optional source type the connection will be used for.

        Returns:
            Created connection JSON with ``id``, ``provider``, ``status``, and
            ``authorization_url``.
        """
        body: JSON = {"provider": provider}
        if source_type is not None:
            body["source_type"] = source_type
        return self._transport.request("POST", "/connections", json_body=body)

    def get(self, connection_id: str) -> JSON:
        """Return one connection by id."""
        return self._transport.request("GET", f"/connections/{connection_id}")

    def delete(self, connection_id: str) -> JSON:
        """Delete a connection."""
        return self._transport.request("DELETE", f"/connections/{connection_id}")

    def connect(
        self,
        provider: str,
        *,
        source_type: Optional[str] = None,
        on_url: Optional[Callable[[str], None]] = None,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> JSON:
        """Run the full create-authorize-poll connection flow.

        Creates a connection, surfaces its ``authorization_url`` (via ``on_url``,
        which defaults to printing the URL and instructions), then polls until the
        connection reports ``status == "connected"`` or ``timeout`` elapses.

        Args:
            provider: OAuth provider identifier (e.g. ``"google_drive"``).
            source_type: Optional source type the connection will be used for.
            on_url: Callback invoked with the authorization URL. Defaults to
                printing the URL and instructions to stdout.
            poll_interval: Seconds between status polls. Defaults to ``2.0``.
            timeout: Maximum seconds to wait for authorization. Defaults to ``300.0``.

        Returns:
            The connected connection JSON.

        Raises:
            ValueError: If the create response lacks an id.
            TimeoutError: If the connection is not connected before ``timeout``.
        """
        connection = self.create(provider, source_type=source_type)
        connection_id = connection.get("id") or connection.get("connection_id")
        if connection_id is None:
            raise ValueError("Connection creation response did not include id.")
        connection_id = str(connection_id)

        authorization_url = connection.get("authorization_url")
        if authorization_url is not None:
            callback = on_url or _print_authorization_url
            callback(str(authorization_url))

        deadline = time.monotonic() + timeout
        while True:
            current = self.get(connection_id)
            if current.get("status") == "connected":
                return current
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Connection {connection_id} was not connected within {timeout}s."
                )
            time.sleep(poll_interval)
