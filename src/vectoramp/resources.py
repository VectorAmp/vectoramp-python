"""Resource clients for the VectorAmp API."""

from __future__ import annotations

import mimetypes
import uuid
from collections.abc import ItemsView, KeysView, ValuesView
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union
from urllib.parse import quote

from .embeddings import (
    EMBEDDING_DIMENSIONS,
    OPENAI_TEXT_EMBEDDING_3_SMALL,
    VECTORAMP_EMBEDDING_4B,
    OpenAIEmbeddingSize,
    openai,
)
from .sources import (
    ConfluenceSource,
    FileUploadSource,
    GCSSource,
    GoogleDriveSource,
    JiraSource,
    S3Source,
    SourceBuilder,
    SourceInput,
    WebSource,
)
from .transport import BaseTransport, RestTransport
from .types import JSON, AdvancedFilter, ConversationTurn, Filters, Metric, Vector, VectorId

PathLike = Union[str, Path]


class Dataset:
    """Dataset resource object returned by create/get/list calls.

    The object keeps the raw API payload while exposing instance methods that
    delegate to the existing service-style clients. It also implements the
    common mapping helpers so existing ``dataset["id"]`` style code keeps
    working.
    """

    def __init__(self, service: "DatasetsResource", raw_data: Mapping[str, Any]) -> None:
        self.service = service
        self.client = service.client
        self.raw_data: JSON = dict(raw_data)
        self.id = self._extract_id(self.raw_data)

    def search(
        self,
        query: Optional[Union[str, Sequence[float]]] = None,
        *,
        vector: Optional[Sequence[float]] = None,
        text: Optional[str] = None,
        search_text: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Filters] = None,
        advanced_filters: Optional[Sequence[AdvancedFilter]] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        nprobe_override: Optional[int] = None,
        rerank_depth_override: Optional[int] = None,
        hybrid: Optional[bool] = None,
        sparse_query: Optional[str] = None,
        alpha: Optional[float] = None,
        include_embeddings: Optional[bool] = None,
        include_documents: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        rerank: Optional[Union[bool, Mapping[str, Any]]] = None,
    ) -> JSON:
        """Search this dataset by text or vector.

        Args:
            query: Convenience query; ``str`` maps to text search and a float
                sequence maps to vector search.
            vector: Explicit vector query. Mutually exclusive with ``query`` and
                ``text``.
            text: Explicit text query. Mutually exclusive with ``query`` and
                ``vector``.
            search_text: Alias for ``text`` for single-field hybrid/BM25 UX.
            top_k: Maximum matches to return. Defaults to ``10``.
            filters: Exact-match metadata filters.
            advanced_filters: API-native advanced metadata filters.
            embedding_provider: Optional provider override for text queries.
            embedding_model: Optional model override for text queries.
            nprobe_override: Optional ANN probe override.
            rerank_depth_override: Optional rerank depth override.
            hybrid: Optional hybrid dense/sparse search toggle.
            sparse_query: Optional sparse query text for hybrid search.
            alpha: Optional dense/sparse weighting for hybrid search.
            include_embeddings: Whether result vectors include embeddings.
            include_documents: Whether result vectors include document text.
            include_metadata: Whether result vectors include metadata; defaults to
                API behavior when omitted.
            rerank: Enable semantic reranking. Use ``True`` or ``{"enabled": True}``;
                provider defaults to ``vectoramp`` and model to ``VectorAmp-Rerank-v1``.

        Returns:
            Search response JSON.
        """
        return self.service.search(
            self.id,
            query,
            vector=vector,
            text=text,
            search_text=search_text,
            top_k=top_k,
            filters=filters,
            advanced_filters=advanced_filters,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            nprobe_override=nprobe_override,
            rerank_depth_override=rerank_depth_override,
            hybrid=hybrid,
            sparse_query=sparse_query,
            alpha=alpha,
            include_embeddings=include_embeddings,
            include_documents=include_documents,
            include_metadata=include_metadata,
            rerank=rerank,
        )

    def insert(self, vectors: Sequence[Vector]) -> JSON:
        """Insert vectors into this dataset.

        Args:
            vectors: Vector records containing values plus optional id/metadata.

        Returns:
            Insert response JSON.
        """
        return self.service.insert_vectors(self.id, vectors)

    def insert_vectors(self, vectors: Sequence[Vector]) -> JSON:
        """Insert vectors into this dataset.

        Args:
            vectors: Vector records containing values plus optional id/metadata.

        Returns:
            Insert response JSON.
        """
        return self.service.insert_vectors(self.id, vectors)

    def delete_vectors(
        self,
        ids: Sequence[VectorId],
        *,
        write_concern: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Delete vectors from this dataset by id."""
        return self.service.delete_vectors(self.id, ids, write_concern=write_concern)

    def add_texts(
        self,
        texts: Union[str, Sequence[str]],
        *,
        ids: Optional[Sequence[VectorId]] = None,
        metadatas: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> JSON:
        """Embed text values with the dataset model and insert them.

        Args:
            texts: Single text or sequence of texts.
            ids: Optional vector ids (string or integer). Integer ids are kept as
                JSON numbers. Length must match ``texts`` when provided.
            metadatas: Optional per-text metadata. Length must match ``texts``.

        Returns:
            Insert response JSON.
        """
        return self.service.add_texts(self.id, texts, ids=ids, metadatas=metadatas)

    def delete(self) -> Any:
        """Delete this dataset and return the API response."""
        return self.service.delete(self.id)

    def list_documents(
        self,
        *,
        limit: int = 50,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
    ) -> JSON:
        """List source documents retained for this dataset.

        Args:
            limit: Maximum documents to return. Defaults to ``50``.
            cursor: Cursor from a previous page's ``next_cursor``.
            status: Optional document status filter.

        Returns:
            Document page JSON from ``/datasets/{id}/documents``.
        """
        return self.service.list_documents(self.id, limit=limit, cursor=cursor, status=status)

    def download_document(self, document_id: str) -> bytes:
        """Download a retained source document as raw bytes.

        Args:
            document_id: Source document identifier from :meth:`list_documents`.

        Returns:
            Raw document bytes.
        """
        return self.service.download_document(self.id, document_id)

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
        include_sources: bool = True,
    ) -> JSON:
        """Ask a non-streaming RAG question scoped to this dataset.

        Args:
            query: Natural-language question.
            top_k: Number of retrieved chunks to consider. Defaults to ``5``.
            conversation_history: Optional prior chat turns.
            include_sources: Whether to include source chunks. Defaults to ``True``.

        Returns:
            JSON response from ``/intelligence/query``.
        """
        if self.client is None:
            raise TypeError("Dataset.ask requires a Dataset created by a VectorAmp client.")
        return self.client.intelligence.query(
            query,
            dataset_id=self.id,
            top_k=top_k,
            conversation_history=conversation_history,
            include_sources=include_sources,
        )

    def ingest_files(
        self,
        paths: Sequence[PathLike],
        *,
        source_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> JSON:
        """Upload local files into this dataset.

        Creates a temporary ``file_upload`` source automatically. When
        ``source_name`` is omitted, the name is generated as
        ``file-upload-{first-file-stem}-{random-suffix}``.

        Args:
            paths: Local file paths to upload. Must contain at least one path.
            source_name: Optional source name for the auto-created source.
            description: Optional description for the source.

        Returns:
            Upload completion response JSON, including the created ingestion job.
        """
        if self.client is None:
            raise TypeError(
                "Dataset.ingest_files requires a Dataset created by a VectorAmp client."
            )
        return self.client.ingestion.ingest_files(
            dataset_id=self.id,
            paths=paths,
            source_name=source_name,
            description=description,
        )

    def ingest_source(self, source: SourceInput, *, pipeline_id: Optional[str] = None) -> JSON:
        """Start ingestion for an existing or newly-created source.

        Args:
            source: Existing source id, or a source builder that will be created
                before starting the job.
            pipeline_id: Optional pipeline override.

        Returns:
            Ingestion job response JSON.
        """
        if self.client is None:
            raise TypeError(
                "Dataset.ingest_source requires a Dataset created by a VectorAmp client."
            )
        source_id = self.client.sources.resolve_source_id(source)
        return self.client.ingestion.start_job(
            source_id=source_id,
            dataset_id=self.id,
            pipeline_id=pipeline_id,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Return a value from the raw dataset payload, or ``default``."""
        return self.raw_data.get(key, default)

    def keys(self) -> KeysView[str]:
        """Return keys from the raw dataset payload."""
        return self.raw_data.keys()

    def values(self) -> ValuesView[Any]:
        """Return values from the raw dataset payload."""
        return self.raw_data.values()

    def items(self) -> ItemsView[str, Any]:
        """Return items from the raw dataset payload."""
        return self.raw_data.items()

    def __getitem__(self, key: str) -> Any:
        return self.raw_data[key]

    def __contains__(self, key: object) -> bool:
        return key in self.raw_data

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw_data)

    def __len__(self) -> int:
        return len(self.raw_data)

    def __repr__(self) -> str:
        return f"Dataset(id={self.id!r})"

    @staticmethod
    def _extract_id(raw_data: Mapping[str, Any]) -> str:
        value = raw_data.get("id") or raw_data.get("dataset_id")
        if value is None:
            raise ValueError("Dataset response did not include id or dataset_id.")
        return str(value)


class DatasetsResource:
    """Dataset management, search, and vector insertion APIs."""

    def __init__(self, transport: BaseTransport, *, client: Optional[Any] = None) -> None:
        self._transport = transport
        self.client = client

    def list(self, *, limit: int = 50, offset: int = 0) -> JSON:
        """List datasets.

        Args:
            limit: Maximum datasets to return. Defaults to ``50``.
            offset: Pagination offset. Defaults to ``0``.

        Returns:
            Page JSON whose ``datasets`` entries are ``Dataset`` objects.
        """
        page = self._transport.request(
            "GET", "/datasets", params={"limit": limit, "offset": offset}
        )
        datasets = page.get("datasets")
        if isinstance(datasets, list):
            page["datasets"] = [self._to_dataset(dataset) for dataset in datasets]
        return page

    def get(self, dataset_id: str) -> Dataset:
        """Return one dataset by id.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            ``Dataset`` wrapper around the API payload.
        """
        return self._to_dataset(self._transport.request("GET", f"/datasets/{dataset_id}"))

    def create(
        self,
        *,
        name: str,
        dim: Optional[int] = None,
        metric: Metric = "cosine",
        embedding: Optional[Mapping[str, str]] = None,
        embedding_provider: str = "vectoramp",
        embedding_model: str = VECTORAMP_EMBEDDING_4B,
        hybrid: bool = False,
        filters: Optional[Mapping[str, Any]] = None,
        metadata_schema: Optional[Mapping[str, Any]] = None,
        tuning: Optional[Mapping[str, Any]] = None,
        openai_api_key: Optional[str] = None,
        openai_secret_ref: str = "emb:openai:api_key",
        validate_openai_key: bool = False,
    ) -> Dataset:
        """Create a SABLE dataset.

        ``index_type`` is intentionally not exposed; the SDK always requests
        SABLE. The only required argument is ``name``; everything else is
        inferred or defaulted (``VectorAmp-Embedding-4B`` at ``dim=2560``,
        ``metric="cosine"``).

        Args:
            name: Dataset name.
            dim: Vector dimension. Inferred for built-in embedding helpers when omitted.
            metric: Distance metric. Defaults to ``"cosine"``.
            embedding: Nested embedding config. Use ``openai("small")`` or
                ``openai("large")`` for OpenAI BYOM datasets.
            embedding_provider: Embedding provider. Defaults to ``"vectoramp"``.
            embedding_model: Embedding model. Defaults to
                ``"VectorAmp-Embedding-4B"``.
            hybrid: Enable hybrid dense + sparse indexing. Sends ``hybrid: true``
                in the create body when set. Defaults to ``False``.
            filters: Optional filter schema/configuration.
            metadata_schema: Optional metadata schema.
            tuning: Optional SABLE tuning parameters.
            openai_api_key: Optional OpenAI API key to store as an organization
                secret before creating the dataset. When provided, and no
                explicit embedding is passed, the dataset uses OpenAI
                ``text-embedding-3-small`` with ``secret_ref`` set to
                ``openai_secret_ref``.
            openai_secret_ref: Organization-secret name used for the OpenAI key.
            validate_openai_key: Ask the API to validate the OpenAI key before
                storing it. Defaults to ``False``.

        Returns:
            Created ``Dataset`` object.
        """
        if openai_api_key is not None:
            if self.client is None or not hasattr(self.client, "secrets"):
                raise TypeError("openai_api_key requires a DatasetResource owned by VectorAmp.")
            validation_model = str(
                (embedding or {}).get("model") or OPENAI_TEXT_EMBEDDING_3_SMALL
            )
            self.client.secrets.put(
                openai_secret_ref,
                openai_api_key,
                validate=validate_openai_key,
                model=validation_model,
            )
            if embedding is None and embedding_provider == "vectoramp":
                embedding = openai("small")

        embedding_config = {"provider": embedding_provider, "model": embedding_model}
        if embedding is not None:
            embedding_config.update(dict(embedding))
        if openai_api_key is not None and embedding_config.get("provider") == "openai":
            embedding_config["secret_ref"] = openai_secret_ref
        resolved_dim = dim or EMBEDDING_DIMENSIONS.get(str(embedding_config.get("model")))
        if resolved_dim is None:
            raise ValueError("dim is required for custom embedding models")

        body: JSON = {
            "name": name,
            "dim": resolved_dim,
            "metric": metric,
            "embedding": embedding_config,
            "index_type": "sable",
        }
        if hybrid:
            body["hybrid"] = True
        if filters is not None:
            body["filters"] = dict(filters)
        if metadata_schema is not None:
            body["metadata_schema"] = dict(metadata_schema)
        if tuning is not None:
            body["tuning"] = dict(tuning)
        return self._to_dataset(self._transport.request("POST", "/datasets", json_body=body))

    def create_openai(
        self,
        *,
        name: str,
        api_key: str,
        size: OpenAIEmbeddingSize = "small",
        secret_ref: str = "emb:openai:api_key",
        validate: bool = False,
        dim: Optional[int] = None,
        metric: Metric = "cosine",
        hybrid: bool = False,
        filters: Optional[Mapping[str, Any]] = None,
        metadata_schema: Optional[Mapping[str, Any]] = None,
        tuning: Optional[Mapping[str, Any]] = None,
    ) -> Dataset:
        """Store an OpenAI org secret, then create an OpenAI-backed dataset."""
        embedding_config = openai(size)
        embedding_config["secret_ref"] = secret_ref
        return self.create(
            name=name,
            dim=dim,
            metric=metric,
            embedding=embedding_config,
            hybrid=hybrid,
            filters=filters,
            metadata_schema=metadata_schema,
            tuning=tuning,
            openai_api_key=api_key,
            openai_secret_ref=secret_ref,
            validate_openai_key=validate,
        )

    def delete(self, dataset_id: str) -> Any:
        """Delete a dataset and return the API response.

        Args:
            dataset_id: Dataset identifier.
        """
        return self._transport.request("DELETE", f"/datasets/{dataset_id}")

    def stats(self, dataset_id: str) -> JSON:
        """Return dataset statistics.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            Stats response JSON.
        """
        return self._transport.request("GET", f"/datasets/{dataset_id}/stats")

    def list_documents(
        self,
        dataset_id: str,
        *,
        limit: int = 50,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
    ) -> JSON:
        """List source documents retained for a dataset.

        Args:
            dataset_id: Dataset identifier.
            limit: Maximum documents to return. Defaults to ``50``.
            cursor: Cursor from a previous page's ``next_cursor``.
            status: Optional document status filter.

        Returns:
            Document page JSON from ``/datasets/{dataset_id}/documents``.
        """
        return self._transport.request(
            "GET",
            f"/datasets/{dataset_id}/documents",
            params={"limit": limit, "cursor": cursor, "status": status},
        )

    def download_document(self, dataset_id: str, document_id: str) -> bytes:
        """Download a retained source document as raw bytes.

        Args:
            dataset_id: Dataset identifier.
            document_id: Source document identifier from :meth:`list_documents`.

        Returns:
            Raw document bytes.
        """
        return self._transport.download(
            "GET", f"/datasets/{dataset_id}/documents/{document_id}/download"
        )

    def search(
        self,
        dataset_id: str,
        query: Optional[Union[str, Sequence[float]]] = None,
        *,
        vector: Optional[Sequence[float]] = None,
        text: Optional[str] = None,
        search_text: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Filters] = None,
        advanced_filters: Optional[Sequence[AdvancedFilter]] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        nprobe_override: Optional[int] = None,
        rerank_depth_override: Optional[int] = None,
        hybrid: Optional[bool] = None,
        sparse_query: Optional[str] = None,
        alpha: Optional[float] = None,
        include_embeddings: Optional[bool] = None,
        include_documents: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        rerank: Optional[Union[bool, Mapping[str, Any]]] = None,
    ) -> JSON:
        """Search a dataset by text or vector.

        Args:
            dataset_id: Dataset identifier.
            query: Convenience query; ``str`` maps to text search and a float
                sequence maps to vector search.
            vector: Explicit vector query. Mutually exclusive with ``query`` and
                ``text``.
            text: Explicit text query. Mutually exclusive with ``query`` and
                ``vector``.
            search_text: Alias for ``text`` for single-field hybrid/BM25 UX.
            top_k: Maximum matches to return. Defaults to ``10``.
            filters: Exact-match metadata filters.
            advanced_filters: API-native advanced metadata filters.
            embedding_provider: Optional provider override for text queries.
            embedding_model: Optional model override for text queries.
            nprobe_override: Optional ANN probe override.
            rerank_depth_override: Optional rerank depth override.
            hybrid: Optional hybrid dense/sparse search toggle.
            sparse_query: Optional sparse query text for hybrid search.
            alpha: Optional dense/sparse weighting for hybrid search.
            include_embeddings: Whether result vectors include embeddings.
            include_documents: Whether result vectors include document text.
            include_metadata: Whether result vectors include metadata; defaults to
                API behavior when omitted.
            rerank: Enable semantic reranking. Use ``True`` or ``{"enabled": True}``;
                provider defaults to ``vectoramp`` and model to ``VectorAmp-Rerank-v1``.

        Returns:
            Search response JSON.
        """
        if search_text is not None:
            if text is not None:
                raise ValueError("Provide text or search_text, not both.")
            text = search_text
        if query is not None:
            if vector is not None or text is not None:
                raise ValueError("Provide query or vector/text/search_text, not both.")
            if isinstance(query, str):
                text = query
            else:
                vector = query
        if (vector is None) == (text is None):
            raise ValueError("Provide exactly one of vector or text/search_text.")
        body: JSON = {"top_k": top_k}
        if vector is not None:
            body["query"] = list(vector)
        if text is not None:
            body["query_text"] = text
        optional = {
            "filters": dict(filters) if filters is not None else None,
            "advanced_filters": list(advanced_filters) if advanced_filters is not None else None,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "nprobe_override": nprobe_override,
            "rerank_depth_override": rerank_depth_override,
            "hybrid": hybrid,
            "sparse_query": sparse_query,
            "alpha": alpha,
            "include_embeddings": include_embeddings,
            "include_documents": include_documents,
            "include_metadata": include_metadata,
            "rerank": dict(rerank) if isinstance(rerank, Mapping) else rerank,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return self._transport.request("POST", f"/datasets/{dataset_id}/search", json_body=body)

    def insert_vectors(self, dataset_id: str, vectors: Sequence[Vector]) -> JSON:
        """Insert vectors into a dataset.

        Args:
            dataset_id: Dataset identifier.
            vectors: Vector records containing values plus optional id/metadata.

        Returns:
            Insert response JSON.
        """
        return self._transport.request(
            "POST", f"/datasets/{dataset_id}/insert", json_body={"vectors": list(vectors)}
        )

    def insert(self, dataset_id: str, vectors: Sequence[Vector]) -> JSON:
        """Alias for ``insert_vectors``."""
        return self.insert_vectors(dataset_id, vectors)

    def delete_vectors(
        self,
        dataset_id: str,
        ids: Sequence[VectorId],
        *,
        write_concern: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Delete vectors from a dataset by id."""
        if not ids:
            raise ValueError("ids must contain at least one vector id.")
        body: JSON = {"ids": list(ids)}
        if write_concern is not None:
            body["write_concern"] = dict(write_concern)
        return self._transport.request(
            "DELETE", f"/api/v1/datasets/{dataset_id}/vectors", json_body=body
        )

    def embed(
        self,
        dataset_id: str,
        *,
        text: Optional[str] = None,
        texts: Optional[Sequence[str]] = None,
    ) -> JSON:
        """Embed one or more texts with the dataset embedding model.

        Args:
            dataset_id: Dataset identifier.
            text: Single text to embed.
            texts: Multiple texts to embed. Mutually exclusive with ``text``.

        Returns:
            Embedding response JSON.
        """
        if (text is None) == (texts is None):
            raise ValueError("Provide exactly one of text or texts.")
        body: JSON = {"text": text} if text is not None else {"texts": list(texts or [])}
        return self._transport.request("POST", f"/datasets/{dataset_id}/embed", json_body=body)

    def add_texts(
        self,
        dataset_id: str,
        texts: Union[str, Sequence[str]],
        *,
        ids: Optional[Sequence[VectorId]] = None,
        metadatas: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> JSON:
        """Embed text values with the dataset model and insert them.

        Args:
            dataset_id: Dataset identifier.
            texts: Single text or sequence of texts.
            ids: Optional vector ids (string or integer). Integer ids are kept as
                JSON numbers rather than coerced to strings. Length must match
                ``texts`` when provided.
            metadatas: Optional per-text metadata. Length must match ``texts``.

        Returns:
            Insert response JSON.
        """
        text_list = [texts] if isinstance(texts, str) else list(texts)
        if ids is not None and len(ids) != len(text_list):
            raise ValueError("ids length must match texts length.")
        if metadatas is not None and len(metadatas) != len(text_list):
            raise ValueError("metadatas length must match texts length.")
        embeddings = self.embed(dataset_id, texts=text_list).get("embeddings", [])
        if len(embeddings) != len(text_list):
            raise ValueError("Embedding response length did not match texts length.")
        vectors: list[Vector] = []
        for index, (text, values) in enumerate(zip(text_list, embeddings)):
            metadata = dict(metadatas[index]) if metadatas is not None else {}
            metadata.setdefault("text", text)
            # Provided ids pass through verbatim so integer ids stay JSON numbers;
            # only auto-generated ids are strings (UUIDs).
            vector_id: VectorId = ids[index] if ids is not None else str(uuid.uuid4())
            vectors.append(
                {
                    "id": vector_id,
                    "values": values,
                    "metadata": metadata,
                }
            )
        return self.insert_vectors(dataset_id, vectors)

    def ensure_engine(self, dataset_id: str) -> JSON:
        """Ensure a dataset engine is loaded and return the API response.

        Args:
            dataset_id: Dataset identifier.
        """
        return self._transport.request("POST", f"/datasets/{dataset_id}/ensure-engine")

    def _to_dataset(self, raw_data: Mapping[str, Any]) -> Dataset:
        return Dataset(self, raw_data)


class OrgSecretsResource:
    """Organization secret helpers."""

    def __init__(self, transport: BaseTransport) -> None:
        self._transport = transport

    def put(
        self,
        name: str,
        value: str,
        *,
        validate: bool = False,
        model: str = OPENAI_TEXT_EMBEDDING_3_SMALL,
    ) -> Any:
        """Create or update an organization secret.

        Stores provider credentials through the generic organization-secret
        REST endpoint. Plaintext is sent once and is never returned by reads.
        """
        if not name:
            raise ValueError("name is required.")
        if not value:
            raise ValueError("value is required.")
        return self._transport.request(
            "PUT",
            f"/org-secrets/{quote(name, safe='')}",
            json_body={"value": value},
        )

    def update(
        self,
        name: str,
        value: str,
        *,
        validate: bool = False,
        model: str = OPENAI_TEXT_EMBEDDING_3_SMALL,
    ) -> Any:
        """Alias for :meth:`put`; org secrets are upserted by name."""
        return self.put(name, value, validate=validate, model=model)

    def put_openai_api_key(
        self,
        api_key: str,
        *,
        secret_ref: str = "emb:openai:api_key",
        validate: bool = False,
        model: str = OPENAI_TEXT_EMBEDDING_3_SMALL,
    ) -> Any:
        """Create or update the OpenAI API key organization secret."""
        return self.put(secret_ref, api_key, validate=validate, model=model)


class IngestionResource:
    """Sources, upload sessions, and ingestion jobs."""

    def __init__(self, transport: BaseTransport) -> None:
        self._transport = transport

    def list_sources(self, *, limit: int = 50, offset: int = 0) -> JSON:
        """List ingestion sources.

        Args:
            limit: Maximum sources to return. Defaults to ``50``.
            offset: Pagination offset. Defaults to ``0``.

        Returns:
            Source page JSON.
        """
        return self._transport.request(
            "GET", "/ingestion/sources", params={"limit": limit, "offset": offset}
        )

    def get_source(self, source_id: str) -> JSON:
        """Return one ingestion source by id."""
        return self._transport.request("GET", f"/ingestion/sources/{source_id}")

    def delete_source(self, source_id: str, *, force: bool = False) -> Any:
        """Delete an ingestion source.

        Args:
            source_id: Source identifier.
            force: When ``True``, delete even if the source is still referenced by
                schedules or jobs (sends ``?force=true``). Defaults to ``False``.

        Returns:
            API response, or ``None`` for a ``204`` response.
        """
        params = {"force": "true"} if force else None
        return self._transport.request(
            "DELETE", f"/ingestion/sources/{source_id}", params=params
        )

    def list_unused_sources(self, *, limit: int = 50, offset: int = 0) -> JSON:
        """List sources not referenced by any schedule or job.

        Args:
            limit: Maximum sources to return. Defaults to ``50``.
            offset: Pagination offset. Defaults to ``0``.

        Returns:
            Source page JSON from ``/ingestion/sources/unused``.
        """
        return self._transport.request(
            "GET", "/ingestion/sources/unused", params={"limit": limit, "offset": offset}
        )

    def cleanup_unused_sources(self) -> JSON:
        """Delete every source not referenced by a schedule or job.

        Returns:
            ``{"deleted": [...], "count": int}`` describing the removed sources.
        """
        return self._transport.request("POST", "/ingestion/sources/cleanup")

    def get_source_references(self, source_id: str) -> JSON:
        """Return the schedules and jobs that reference a source.

        Args:
            source_id: Source identifier.

        Returns:
            Reference JSON from ``/ingestion/sources/{id}/references``.
        """
        return self._transport.request("GET", f"/ingestion/sources/{source_id}/references")

    def validate_source(
        self,
        source: Optional[SourceBuilder] = None,
        *,
        source_type: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Validate a source configuration without creating it.

        Accepts either a source builder (its ``source_type`` and ``config`` are
        derived from ``to_create_request``) or explicit ``source_type`` and
        ``config`` keyword arguments.

        Args:
            source: Optional source builder such as ``WebSource`` or ``S3Source``.
            source_type: Source type when not passing ``source``.
            config: Source config when not passing ``source``.

        Returns:
            Validation response JSON from ``/ingestion/sources/validate``.
        """
        if source is not None:
            request = source.to_create_request()
            body = {"source_type": request["source_type"], "config": request["config"]}
        else:
            if source_type is None or config is None:
                raise TypeError(
                    "validate_source requires source or both source_type and config."
                )
            body = {"source_type": source_type, "config": dict(config)}
        return self._transport.request(
            "POST", "/ingestion/sources/validate", json_body=body
        )

    def create_source(
        self,
        source: Optional[SourceBuilder] = None,
        *,
        name: Optional[str] = None,
        source_type: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create an ingestion source.

        Args:
            source: Optional source builder such as ``WebSource`` or ``S3Source``.
            name: Source name when not passing ``source``.
            source_type: Source type when not passing ``source``.
            config: Source config when not passing ``source``.
            description: Optional source description.
            metadata: Optional source metadata.

        Returns:
            Created source JSON.
        """
        if source is not None:
            body = source.to_create_request()
        else:
            if name is None or source_type is None or config is None:
                raise TypeError("create_source requires source or name, source_type, and config.")
            body = {
                "name": name,
                "source_type": source_type,
                "config": dict(config),
            }
            if description is not None:
                body["description"] = description
            if metadata is not None:
                body["metadata"] = dict(metadata)
        return self._transport.request("POST", "/ingestion/sources", json_body=body)

    def create(self, source: SourceBuilder) -> JSON:
        """Create an ingestion source from a source builder."""
        return self.create_source(source)

    def create_web(
        self,
        *,
        start_urls: Sequence[str],
        name: Optional[str] = None,
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
        allowed_domains: Optional[Sequence[str]] = None,
        include_patterns: Optional[Sequence[str]] = None,
        exclude_patterns: Optional[Sequence[str]] = None,
        crawl_delay_seconds: Optional[float] = None,
        include_assets: Optional[bool] = None,
        max_assets_per_page: Optional[int] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config_extra: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create a web crawler source.

        ``name`` defaults to ``web-{host-or-path}`` from the first start URL.
        Optional crawl settings are omitted from the request when ``None``.

        Returns:
            Created source JSON.
        """
        return self.create_source(
            WebSource(
                name=name,
                start_urls=start_urls,
                max_depth=max_depth,
                max_pages=max_pages,
                allowed_domains=allowed_domains,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                crawl_delay_seconds=crawl_delay_seconds,
                include_assets=include_assets,
                max_assets_per_page=max_assets_per_page,
                description=description,
                metadata=metadata,
                config_extra=config_extra,
            )
        )

    def create_s3(
        self,
        *,
        bucket: str,
        name: Optional[str] = None,
        region: str = "us-east-1",
        prefix: Optional[str] = None,
        sync_mode: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        role_arn: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        file_patterns: Optional[Sequence[str]] = None,
        max_file_size_mb: Optional[int] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config_extra: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create an Amazon S3 source.

        ``name`` defaults to ``s3-{bucket}``; ``region`` defaults to
        ``"us-east-1"``. ``sync_mode`` is omitted when ``None`` so the server
        applies its default (``"incremental"``). Optional credentials and file
        settings are omitted when ``None``.

        Returns:
            Created source JSON.
        """
        return self.create_source(
            S3Source(
                name=name,
                bucket=bucket,
                region=region,
                prefix=prefix,
                sync_mode=sync_mode,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                role_arn=role_arn,
                endpoint_url=endpoint_url,
                file_patterns=file_patterns,
                max_file_size_mb=max_file_size_mb,
                description=description,
                metadata=metadata,
                config_extra=config_extra,
            )
        )

    def create_gcs(
        self,
        *,
        bucket: str,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        project_id: Optional[str] = None,
        credentials_json: Optional[Mapping[str, Any]] = None,
        sync_mode: Optional[str] = None,
        file_patterns: Optional[Sequence[str]] = None,
        max_file_size_mb: Optional[int] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config_extra: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create a Google Cloud Storage source.

        ``sync_mode`` is omitted when ``None`` so the server applies its default
        (``"incremental"``).
        """
        return self.create_source(
            GCSSource(
                name=name,
                bucket=bucket,
                prefix=prefix,
                project_id=project_id,
                credentials_json=credentials_json,
                sync_mode=sync_mode,
                file_patterns=file_patterns,
                max_file_size_mb=max_file_size_mb,
                description=description,
                metadata=metadata,
                config_extra=config_extra,
            )
        )

    def create_jira(
        self,
        *,
        cloud_id: str,
        name: Optional[str] = None,
        access_token: Optional[str] = None,
        project_keys: Optional[Sequence[str]] = None,
        jql: Optional[str] = None,
        include_comments: bool = True,
        sync_mode: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config_extra: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create a Jira source.

        ``include_comments`` defaults to true. ``sync_mode`` is omitted when
        ``None`` so the server applies its default (``"incremental"``).
        """
        return self.create_source(
            JiraSource(
                name=name,
                cloud_id=cloud_id,
                access_token=access_token,
                project_keys=project_keys,
                jql=jql,
                include_comments=include_comments,
                sync_mode=sync_mode,
                description=description,
                metadata=metadata,
                config_extra=config_extra,
            )
        )

    def create_confluence(
        self,
        *,
        cloud_id: Optional[str] = None,
        base_url: Optional[str] = None,
        name: Optional[str] = None,
        auth_mode: str = "basic",
        username: Optional[str] = None,
        api_token: Optional[str] = None,
        oauth_credentials: Optional[Mapping[str, Any]] = None,
        spaces: Optional[Sequence[str]] = None,
        include_attachments: bool = False,
        sync_mode: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config_extra: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create a Confluence source.

        Provide either ``cloud_id`` or ``base_url``. ``auth_mode`` defaults to
        ``"basic"`` (set ``username``/``api_token``); use ``oauth_credentials``
        for OAuth. ``include_attachments`` defaults to ``False``. ``sync_mode``
        is omitted when ``None`` so the server applies its default
        (``"incremental"``).

        Returns:
            Created source JSON.
        """
        return self.create_source(
            ConfluenceSource(
                name=name,
                cloud_id=cloud_id,
                base_url=base_url,
                auth_mode=auth_mode,
                username=username,
                api_token=api_token,
                oauth_credentials=oauth_credentials,
                spaces=spaces,
                include_attachments=include_attachments,
                sync_mode=sync_mode,
                description=description,
                metadata=metadata,
                config_extra=config_extra,
            )
        )

    def create_google_drive(
        self,
        *,
        name: Optional[str] = None,
        folder_ids: Optional[Sequence[str]] = None,
        file_ids: Optional[Sequence[str]] = None,
        auth_mode: str = "oauth",
        oauth_credentials: Optional[Mapping[str, Any]] = None,
        include_shared_drives: Optional[bool] = None,
        sync_mode: Optional[str] = None,
        service_account_json: Optional[Mapping[str, Any]] = None,
        credentials_json: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config_extra: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create a Google Drive source.

        ``name`` defaults to ``gdrive-{first-folder-or-file-id}`` or ``gdrive``.
        ``auth_mode`` defaults to ``"oauth"``. ``sync_mode`` is omitted when
        ``None`` so the server applies its default (``"incremental"``). Optional
        auth/config values are omitted when ``None``.

        Returns:
            Created source JSON.
        """
        return self.create_source(
            GoogleDriveSource(
                name=name,
                folder_ids=folder_ids,
                file_ids=file_ids,
                auth_mode=auth_mode,
                oauth_credentials=oauth_credentials,
                include_shared_drives=include_shared_drives,
                sync_mode=sync_mode,
                service_account_json=service_account_json,
                credentials_json=credentials_json,
                description=description,
                metadata=metadata,
                config_extra=config_extra,
            )
        )

    def create_file_upload(
        self,
        *,
        name: str = "vectoramp-python-upload",
        storage_provider: str = "s3",
        sync_mode: str = "full",
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config_extra: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create a file-upload source record.

        Args:
            name: Source name. Defaults to ``"vectoramp-python-upload"``.
            storage_provider: Upload storage provider. Defaults to ``"s3"``.
            sync_mode: Source sync mode. Defaults to ``"full"``.
            description: Optional source description.
            metadata: Optional source metadata.
            config_extra: Optional extra config fields merged into the request.

        Returns:
            Created source JSON. For local file upload, prefer ``ingest_files``,
            which creates this source automatically and uploads the files.
        """
        return self.create_source(
            FileUploadSource(
                name=name,
                storage_provider=storage_provider,
                sync_mode=sync_mode,
                description=description,
                metadata=metadata,
                config_extra=config_extra,
            )
        )

    def resolve_source_id(self, source: SourceInput) -> str:
        """Return a source id, creating builder inputs when needed.

        Args:
            source: Existing source id or source builder.

        Returns:
            Source identifier string.
        """
        if isinstance(source, str):
            return source
        created = self.create_source(source)
        source_id = created.get("id") or created.get("source_id") or created.get("uuid")
        if source_id is None:
            raise ValueError("Source creation response did not include id or source_id.")
        return str(source_id)

    def start_job(
        self, *, source_id: str, dataset_id: str, pipeline_id: Optional[str] = None
    ) -> JSON:
        """Start an ingestion job.

        Args:
            source_id: Source identifier.
            dataset_id: Target dataset identifier.
            pipeline_id: Optional pipeline override.

        Returns:
            Ingestion job response JSON.
        """
        body: JSON = {"source_id": source_id, "dataset_id": dataset_id}
        if pipeline_id is not None:
            body["pipeline_id"] = pipeline_id
        return self._transport.request("POST", "/ingestion/jobs", json_body=body)

    def list_jobs(
        self, *, dataset_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> JSON:
        """List ingestion jobs.

        Args:
            dataset_id: Optional dataset filter.
            limit: Maximum jobs to return. Defaults to ``50``.
            offset: Pagination offset. Defaults to ``0``.

        Returns:
            Job page JSON.
        """
        return self._transport.request(
            "GET",
            "/ingestion/jobs",
            params={"dataset_id": dataset_id, "limit": limit, "offset": offset},
        )

    def get_job(self, job_id: str) -> JSON:
        """Return one ingestion job by id."""
        return self._transport.request("GET", f"/ingestion/jobs/{job_id}")

    def retry_job(self, job_id: str) -> JSON:
        """Queue a fresh full-rerun job from an eligible failed or cancelled job."""
        return self._transport.request("POST", f"/ingestion/jobs/{job_id}/retry")

    def init_upload(self, source_id: str, files: Sequence[Mapping[str, Any]]) -> JSON:
        """Initialize presigned uploads for a file-upload source.

        Args:
            source_id: File-upload source identifier.
            files: File descriptors with name, size, and content type.

        Returns:
            Upload session JSON, including upload URLs and job id.
        """
        return self._transport.request(
            "POST", f"/ingestion/sources/{source_id}/upload/init", json_body={"files": list(files)}
        )

    def complete_upload(self, source_id: str, *, job_id: str, file_ids: Sequence[str]) -> JSON:
        """Complete a file-upload session.

        Args:
            source_id: File-upload source identifier.
            job_id: Upload job identifier returned by ``init_upload``.
            file_ids: Uploaded file identifiers returned by ``init_upload``.

        Returns:
            Upload completion response JSON.
        """
        return self._transport.request(
            "POST",
            f"/ingestion/sources/{source_id}/upload/complete",
            json_body={"job_id": job_id, "file_ids": list(file_ids)},
        )

    def ingest_files(
        self,
        *,
        dataset_id: str,
        paths: Sequence[PathLike],
        source_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> JSON:
        """Create a file-upload source, upload files, and complete the flow.

        Creates a ``file_upload`` source automatically with config
        ``{"storage_provider": "s3", "sync_mode": "full"}`` and metadata
        containing ``dataset_id``. When ``source_name`` is omitted, the name is
        generated as ``file-upload-{first-file-stem}-{random-suffix}``.

        Args:
            dataset_id: Target dataset identifier.
            paths: Local file paths to upload. Must contain at least one path.
            source_name: Optional source name for the auto-created source.
            description: Optional description for the source.

        Returns:
            Upload completion response JSON, including the created ingestion job.
        """
        file_paths = [Path(path) for path in paths]
        if not file_paths:
            raise ValueError("ingest_files requires at least one path.")
        source = self.create_source(
            name=source_name or self._default_upload_source_name(file_paths),
            source_type="file_upload",
            description=description,
            config={"storage_provider": "s3", "sync_mode": "full"},
            metadata={"dataset_id": dataset_id},
        )
        source_id = str(source.get("id") or source.get("source_id"))
        files = [self._file_descriptor(path) for path in file_paths]
        upload = self.init_upload(source_id, files)
        uploads = upload.get("uploads", [])
        if len(uploads) != len(file_paths):
            raise ValueError("Upload init response did not match requested files.")
        if not isinstance(self._transport, RestTransport):
            raise TypeError(
                "ingest_files requires a REST transport with presigned URL upload support."
            )
        file_ids: list[str] = []
        for path, upload_info in zip(file_paths, uploads):
            file_ids.append(str(upload_info["file_id"]))
            self._transport.put_url(
                str(upload_info["upload_url"]),
                content=path.read_bytes(),
                content_type=self._guess_content_type(path),
            )
        job_id = str(upload["job_id"])
        complete = self.complete_upload(source_id, job_id=job_id, file_ids=file_ids)
        if isinstance(complete, dict):
            complete.setdefault("job_id", job_id)
        return complete

    @staticmethod
    def _default_upload_source_name(paths: Sequence[Path]) -> str:
        first_stem = paths[0].stem or "files"
        slug = "".join(char.lower() if char.isalnum() else "-" for char in first_stem)
        slug = "-".join(part for part in slug.split("-") if part)[:32] or "files"
        return f"file-upload-{slug}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _file_descriptor(path: Path) -> JSON:
        return {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "content_type": IngestionResource._guess_content_type(path),
        }

    @staticmethod
    def _guess_content_type(path: Path) -> str:
        return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


class SchedulesResource:
    """Ingestion schedule management.

    Recurring ingestion runs are defined as schedules. Each schedule pairs a
    source with a target dataset and a cron expression; the ingestion scheduler
    daemon polls for due schedules and creates jobs as they fire.
    """

    def __init__(self, transport: BaseTransport) -> None:
        self._transport = transport

    def list(self, *, limit: int = 50, offset: int = 0) -> JSON:
        """List schedules for the calling organization.

        Args:
            limit: Maximum schedules to return. Defaults to ``50``.
            offset: Pagination offset. Defaults to ``0``.

        Returns:
            ``{"schedules": [...], "total": int, "limit": int, "offset": int}``.
        """
        return self._transport.request(
            "GET", "/ingestion/schedules", params={"limit": limit, "offset": offset}
        )

    def get(self, schedule_id: str) -> JSON:
        """Return one schedule by id."""
        return self._transport.request("GET", f"/ingestion/schedules/{schedule_id}")

    def create(
        self,
        *,
        source_id: str,
        dataset_id: str,
        cron: str,
        timezone: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create a recurring ingestion schedule.

        Args:
            source_id: Ingestion source id to pull from on each run.
            dataset_id: Dataset id to ingest into.
            cron: 5-field cron expression (e.g. ``"0 * * * *"`` for hourly).
            timezone: Optional IANA timezone. Defaults to ``UTC`` server-side.
            pipeline_id: Optional pipeline id. Omit to use the default ingestion pipeline.
            enabled: Optional flag, defaults to ``True`` server-side.
            name: Optional human-readable name.
            metadata: Optional metadata blob attached to the schedule.

        Returns:
            Created schedule JSON.
        """
        body: dict[str, Any] = {
            "source_id": source_id,
            "dataset_id": dataset_id,
            "cron": cron,
        }
        if timezone is not None:
            body["timezone"] = timezone
        if pipeline_id is not None:
            body["pipeline_id"] = pipeline_id
        if enabled is not None:
            body["enabled"] = enabled
        if name is not None:
            body["name"] = name
        if metadata is not None:
            body["metadata"] = dict(metadata)
        return self._transport.request("POST", "/ingestion/schedules", json_body=body)

    def update(
        self,
        schedule_id: str,
        *,
        cron: Optional[str] = None,
        timezone: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        enabled: Optional[bool] = None,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Update a schedule. Pass only the fields you want to change.

        Returns:
            Updated schedule JSON.
        """
        body: dict[str, Any] = {}
        if cron is not None:
            body["cron"] = cron
        if timezone is not None:
            body["timezone"] = timezone
        if pipeline_id is not None:
            body["pipeline_id"] = pipeline_id
        if enabled is not None:
            body["enabled"] = enabled
        if name is not None:
            body["name"] = name
        if metadata is not None:
            body["metadata"] = dict(metadata)
        return self._transport.request(
            "PATCH", f"/ingestion/schedules/{schedule_id}", json_body=body
        )

    def delete(self, schedule_id: str) -> JSON:
        """Delete a schedule."""
        return self._transport.request("DELETE", f"/ingestion/schedules/{schedule_id}")

    def trigger(self, schedule_id: str) -> JSON:
        """Trigger an immediate run for a schedule, outside its cron cadence.

        Returns:
            ``{"job_id": "..."}`` for the newly created ingestion job.
        """
        return self._transport.request(
            "POST", f"/ingestion/schedules/{schedule_id}/trigger"
        )


class IntelligenceResource:
    """RAG query APIs."""

    def __init__(self, transport: BaseTransport) -> None:
        self._transport = transport

    def query(
        self,
        query: str,
        *,
        dataset_id: Optional[str] = None,
        top_k: int = 5,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
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
        body = self._body(
            query,
            dataset_id=dataset_id,
            top_k=top_k,
            conversation_history=conversation_history,
            include_sources=include_sources,
            stream=False,
        )
        return self._transport.request("POST", "/intelligence/query", json_body=body)

    def stream(
        self,
        query: str,
        *,
        dataset_id: Optional[str] = None,
        top_k: int = 5,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
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
        body = self._body(
            query,
            dataset_id=dataset_id,
            top_k=top_k,
            conversation_history=conversation_history,
            include_sources=include_sources,
            stream=True,
        )
        yield from self._transport.stream("POST", "/intelligence/query", json_body=body)


    def create_session(
        self,
        *,
        title: Optional[str] = None,
        workspace_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create a persistent Intelligence session."""
        body: JSON = {}
        if title is not None:
            body["title"] = title
        if workspace_id is not None:
            body["workspace_id"] = workspace_id
        if dataset_id is not None:
            body["dataset_id"] = dataset_id
        if metadata is not None:
            body["metadata"] = dict(metadata)
        return self._transport.request("POST", "/intelligence/sessions", json_body=body)

    def list_sessions(self, *, limit: int = 50) -> JSON:
        """List persistent Intelligence sessions."""
        return self._transport.request("GET", "/intelligence/sessions", params={"limit": limit})

    def get_session(self, session_id: str) -> JSON:
        """Fetch one persistent Intelligence session."""
        return self._transport.request("GET", f"/intelligence/sessions/{session_id}")

    def delete_session(self, session_id: str) -> JSON:
        """Delete a persistent Intelligence session."""
        return self._transport.request("DELETE", f"/intelligence/sessions/{session_id}")

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Append a message to a persistent Intelligence session."""
        body: JSON = {"role": role, "content": content}
        if metadata is not None:
            body["metadata"] = dict(metadata)
        return self._transport.request(
            "POST", f"/intelligence/sessions/{session_id}/messages", json_body=body
        )

    def list_messages(self, session_id: str, *, limit: int = 100) -> JSON:
        """List messages for a persistent Intelligence session."""
        return self._transport.request(
            "GET", f"/intelligence/sessions/{session_id}/messages", params={"limit": limit}
        )

    @staticmethod
    def _body(
        query: str,
        *,
        dataset_id: Optional[str],
        top_k: int,
        conversation_history: Optional[Sequence[ConversationTurn]],
        include_sources: bool,
        stream: bool,
    ) -> JSON:
        body: JSON = {
            "query": query,
            "top_k": top_k,
            "stream": stream,
            "include_sources": include_sources,
        }
        if dataset_id is not None:
            body["dataset_id"] = dataset_id
        if conversation_history is not None:
            body["conversation_history"] = list(conversation_history)
        return body
