"""Resource clients for the VectorAmp API."""

from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from .transport import BaseTransport, RestTransport
from .types import JSON, AdvancedFilter, ConversationTurn, Filters, Metric, Vector

PathLike = Union[str, Path]


class DatasetsResource:
    """Dataset management, search, and vector insertion APIs."""

    def __init__(self, transport: BaseTransport) -> None:
        self._transport = transport

    def list(self, *, limit: int = 50, offset: int = 0) -> JSON:
        return self._transport.request(
            "GET", "/datasets", params={"limit": limit, "offset": offset}
        )

    def get(self, dataset_id: str) -> JSON:
        return self._transport.request("GET", f"/datasets/{dataset_id}")

    def create(
        self,
        *,
        name: str,
        dim: int,
        metric: Metric = "cosine",
        embedding_provider: str = "vectoramp",
        embedding_model: str = "Qwen/Qwen3-Embedding-4B",
        filters: Optional[Mapping[str, Any]] = None,
        metadata_schema: Optional[Mapping[str, Any]] = None,
        tuning: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        """Create a SABLE dataset.

        `index_type` is intentionally not exposed; the SDK always requests SABLE.
        """
        body: JSON = {
            "name": name,
            "dim": dim,
            "metric": metric,
            "embedding": {"provider": embedding_provider, "model": embedding_model},
            "index_type": "sable",
        }
        if filters is not None:
            body["filters"] = dict(filters)
        if metadata_schema is not None:
            body["metadata_schema"] = dict(metadata_schema)
        if tuning is not None:
            body["tuning"] = dict(tuning)
        return self._transport.request("POST", "/datasets", json_body=body)

    def delete(self, dataset_id: str) -> Any:
        return self._transport.request("DELETE", f"/datasets/{dataset_id}")

    def stats(self, dataset_id: str) -> JSON:
        return self._transport.request("GET", f"/datasets/{dataset_id}/stats")

    def search(
        self,
        dataset_id: str,
        *,
        vector: Optional[Sequence[float]] = None,
        text: Optional[str] = None,
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
    ) -> JSON:
        if (vector is None) == (text is None):
            raise ValueError("Provide exactly one of vector or text.")
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
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return self._transport.request("POST", f"/datasets/{dataset_id}/search", json_body=body)

    def insert_vectors(self, dataset_id: str, vectors: Sequence[Vector]) -> JSON:
        return self._transport.request(
            "POST", f"/datasets/{dataset_id}/insert", json_body={"vectors": list(vectors)}
        )

    def embed(
        self,
        dataset_id: str,
        *,
        text: Optional[str] = None,
        texts: Optional[Sequence[str]] = None,
    ) -> JSON:
        if (text is None) == (texts is None):
            raise ValueError("Provide exactly one of text or texts.")
        body: JSON = {"text": text} if text is not None else {"texts": list(texts or [])}
        return self._transport.request("POST", f"/datasets/{dataset_id}/embed", json_body=body)

    def add_texts(
        self,
        dataset_id: str,
        texts: Sequence[str],
        *,
        ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> JSON:
        """Embed texts with the dataset model and insert them as vectors."""
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids length must match texts length.")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length.")
        embeddings = self.embed(dataset_id, texts=texts).get("embeddings", [])
        if len(embeddings) != len(texts):
            raise ValueError("Embedding response length did not match texts length.")
        vectors: list[Vector] = []
        for index, (text, values) in enumerate(zip(texts, embeddings)):
            metadata = dict(metadatas[index]) if metadatas is not None else {}
            metadata.setdefault("text", text)
            vectors.append(
                {
                    "id": ids[index] if ids is not None else str(uuid.uuid4()),
                    "values": values,
                    "metadata": metadata,
                }
            )
        return self.insert_vectors(dataset_id, vectors)

    def ensure_engine(self, dataset_id: str) -> JSON:
        return self._transport.request("POST", f"/datasets/{dataset_id}/ensure-engine")


class IngestionResource:
    """Sources, upload sessions, and ingestion jobs."""

    def __init__(self, transport: BaseTransport) -> None:
        self._transport = transport

    def list_sources(self, *, limit: int = 50, offset: int = 0) -> JSON:
        return self._transport.request(
            "GET", "/ingestion/sources", params={"limit": limit, "offset": offset}
        )

    def get_source(self, source_id: str) -> JSON:
        return self._transport.request("GET", f"/ingestion/sources/{source_id}")

    def create_source(
        self,
        *,
        name: str,
        source_type: str,
        config: Mapping[str, Any],
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        body: JSON = {
            "name": name,
            "source_type": source_type,
            "config": dict(config),
        }
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = dict(metadata)
        return self._transport.request("POST", "/v1/sources", json_body=body)

    def start_job(
        self, *, source_id: str, dataset_id: str, pipeline_id: Optional[str] = None
    ) -> JSON:
        body: JSON = {"source_id": source_id, "dataset_id": dataset_id}
        if pipeline_id is not None:
            body["pipeline_id"] = pipeline_id
        return self._transport.request("POST", "/ingestion/jobs", json_body=body)

    def list_jobs(
        self, *, dataset_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> JSON:
        return self._transport.request(
            "GET",
            "/ingestion/jobs",
            params={"dataset_id": dataset_id, "limit": limit, "offset": offset},
        )

    def get_job(self, job_id: str) -> JSON:
        return self._transport.request("GET", f"/ingestion/jobs/{job_id}")

    def init_upload(self, source_id: str, files: Sequence[Mapping[str, Any]]) -> JSON:
        return self._transport.request(
            "POST", f"/v1/sources/{source_id}/upload/init", json_body={"files": list(files)}
        )

    def complete_upload(self, source_id: str, *, job_id: str, file_ids: Sequence[str]) -> JSON:
        return self._transport.request(
            "POST",
            f"/v1/sources/{source_id}/upload/complete",
            json_body={"job_id": job_id, "file_ids": list(file_ids)},
        )

    def ingest_files(
        self,
        *,
        dataset_id: str,
        paths: Sequence[PathLike],
        source_name: str = "vectoramp-python-upload",
        description: Optional[str] = None,
    ) -> JSON:
        """Create a file-upload source, upload local files, and complete the upload flow."""
        source = self.create_source(
            name=source_name,
            source_type="file_upload",
            description=description,
            config={"storage_provider": "s3", "sync_mode": "full"},
            metadata={"dataset_id": dataset_id},
        )
        source_id = str(source.get("id") or source.get("source_id"))
        file_paths = [Path(path) for path in paths]
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
        return self.complete_upload(source_id, job_id=str(upload["job_id"]), file_ids=file_ids)

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
        body = self._body(
            query,
            dataset_id=dataset_id,
            top_k=top_k,
            conversation_history=conversation_history,
            include_sources=include_sources,
            stream=True,
        )
        yield from self._transport.stream("POST", "/intelligence/query", json_body=body)

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
