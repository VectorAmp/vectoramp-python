"""Top-level VectorAmp client."""

from __future__ import annotations

import os
from types import TracebackType
from typing import Iterator, Optional, Type

import httpx

from .resources import DatasetsResource, IngestionResource, IntelligenceResource
from .transport import BaseTransport, RestTransport
from .types import JSON, ConversationTurn


class VectorAmp:
    """Synchronous VectorAmp API client."""

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
        """Ask a non-streaming RAG question."""
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
        """Yield Server-Sent Event chunks for a streaming RAG answer."""
        yield from self.intelligence.stream(
            query,
            dataset_id=dataset_id,
            top_k=top_k,
            conversation_history=conversation_history,
            include_sources=include_sources,
        )

    def close(self) -> None:
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
