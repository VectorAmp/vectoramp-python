"""Top-level VectorAmp client."""

from __future__ import annotations

import os
from types import TracebackType
from typing import Iterator, Optional, Type

import httpx

from .connections import ConnectionsResource
from .resources import (
    DatasetsResource,
    IngestionResource,
    IntelligenceResource,
    OrgSecretsResource,
    SchedulesResource,
)
from .transport import BaseTransport, RestTransport
from .types import JSON, ConversationTurn


class VectorAmp:
    """Synchronous VectorAmp API client.

    Args:
        api_key: API key to send as bearer auth. Defaults to
            ``VECTORAMP_API_KEY`` when omitted.
        base_url: API origin. Defaults to ``https://api.vectoramp.com``.
        timeout: Request timeout in seconds. Defaults to ``30.0``.
        transport: Optional custom transport, primarily for tests or adapters.
        http_client: Optional ``httpx.Client`` used by the default REST transport.

    Attributes:
        datasets: Dataset management, search, embedding, and insertion APIs.
        ingestion: Source, upload, and ingestion-job APIs.
        sources: Alias for ``ingestion``.
        connections: OAuth connection management for providers.
        schedules: Recurring ingestion schedule management.
        intelligence: RAG query and streaming APIs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = "https://api.vectoramp.com",
        timeout: float = 30.0,
        transport: Optional[BaseTransport] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("VECTORAMP_API_KEY") or ""
        self.transport = transport or RestTransport(
            api_key=resolved_api_key,
            base_url=base_url,
            timeout=timeout,
            client=http_client,
        )
        self.datasets = DatasetsResource(self.transport, client=self)
        self.ingestion = IngestionResource(self.transport)
        self.sources = self.ingestion
        self.connections = ConnectionsResource(self.transport)
        self.secrets = OrgSecretsResource(self.transport)
        self.org_secrets = self.secrets
        self.schedules = SchedulesResource(self.transport)
        self.intelligence = IntelligenceResource(self.transport)

    def ask(
        self,
        query: str,
        *,
        dataset_id: Optional[str] = None,
        top_k: int = 5,
        conversation_history: Optional[list[ConversationTurn]] = None,
        include_sources: bool = True,
    ) -> JSON:
        """Ask a non-streaming RAG question.

        Args:
            query: Natural-language question.
            dataset_id: Optional dataset to ground the answer in. When omitted,
                the API chooses its configured/default scope.
            top_k: Number of retrieved chunks to consider. Defaults to ``5``.
            conversation_history: Optional prior chat turns.
            include_sources: Whether to include source chunks. Defaults to ``True``.

        Returns:
            JSON response from ``/intelligence/query``.
        """
        return self.intelligence.query(
            query,
            dataset_id=dataset_id,
            top_k=top_k,
            conversation_history=conversation_history,
            include_sources=include_sources,
        )

    def ask_stream(
        self,
        query: str,
        *,
        dataset_id: Optional[str] = None,
        top_k: int = 5,
        conversation_history: Optional[list[ConversationTurn]] = None,
        include_sources: bool = True,
    ) -> Iterator[JSON]:
        """Yield Server-Sent Event chunks for a streaming RAG answer.

        Args:
            query: Natural-language question.
            dataset_id: Optional dataset to ground the answer in. When omitted,
                the API chooses its configured/default scope.
            top_k: Number of retrieved chunks to consider. Defaults to ``5``.
            conversation_history: Optional prior chat turns.
            include_sources: Whether to include source chunks. Defaults to ``True``.

        Returns:
            Iterator of JSON Server-Sent Event payloads.
        """
        yield from self.intelligence.stream(
            query,
            dataset_id=dataset_id,
            top_k=top_k,
            conversation_history=conversation_history,
            include_sources=include_sources,
        )

    def close(self) -> None:
        """Close the underlying transport and HTTP connection pool."""
        self.transport.close()

    def __enter__(self) -> "VectorAmp":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()
