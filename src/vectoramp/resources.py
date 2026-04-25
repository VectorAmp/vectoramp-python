"""Resource clients for the VectorAmp API."""

from __future__ import annotations

import mimetypes
import uuid
from collections.abc import ItemsView, KeysView, ValuesView
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from .sources import (
    FileUploadSource,
    GoogleDriveSource,
    S3Source,
    SourceBuilder,
    SourceInput,
    WebSource,
)
from .transport import BaseTransport, RestTransport
from .types import JSON, AdvancedFilter, ConversationTurn, Filters, Metric, Vector

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
        return self.service.search(
            self.id,
            vector=vector,
            text=text,
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
        )

    def insert(self, vectors: Sequence[Vector]) -> JSON:
        return self.service.insert_vectors(self.id, vectors)

    def insert_vectors(self, vectors: Sequence[Vector]) -> JSON:
        return self.service.insert_vectors(self.id, vectors)

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> JSON:
        return self.service.add_texts(self.id, texts, ids=ids, metadatas=metadatas)

    def delete(self) -> Any:
        return self.service.delete(self.id)

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        conversation_history: Optional[Sequence[ConversationTurn]] = None,
        include_sources: bool = True,
    ) -> JSON:
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
        source_name: str = "vectoramp-python-upload",
        description: Optional[str] = None,
    ) -> JSON:
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
        return self.raw_data.get(key, default)

    def keys(self) -> KeysView[str]:
        return self.raw_data.keys()

    def values(self) -> ValuesView[Any]:
        return self.raw_data.values()

    def items(self) -> ItemsView[str, Any]:
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
        page = self._transport.request(
            "GET", "/datasets", params={"limit": limit, "offset": offset}
        )
        datasets = page.get("datasets")
        if isinstance(datasets, list):
            page["datasets"] = [self._to_dataset(dataset) for dataset in datasets]
        return page

    def get(self, dataset_id: str) -> Dataset:
        return self._to_dataset(self._transport.request("GET", f"/datasets/{dataset_id}"))

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
    ) -> Dataset:
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
        return self._to_dataset(self._transport.request("POST", "/datasets", json_body=body))

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

    def insert(self, dataset_id: str, vectors: Sequence[Vector]) -> JSON:
        return self.insert_vectors(dataset_id, vectors)

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

    def _to_dataset(self, raw_data: Mapping[str, Any]) -> Dataset:
        return Dataset(self, raw_data)


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
        source: Optional[SourceBuilder] = None,
        *,
        name: Optional[str] = None,
        source_type: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
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
        return self._transport.request("POST", "/v1/sources", json_body=body)

    def create(self, source: SourceBuilder) -> JSON:
        return self.create_source(source)

    def create_web(
        self,
        *,
        name: str,
        start_urls: Sequence[str],
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
        allowed_domains: Optional[Sequence[str]] = None,
        include_patterns: Optional[Sequence[str]] = None,
        exclude_patterns: Optional[Sequence[str]] = None,
        crawl_delay_seconds: Optional[float] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config_extra: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
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
                description=description,
                metadata=metadata,
                config_extra=config_extra,
            )
        )

    def create_s3(
        self,
        *,
        name: str,
        bucket: str,
        region: str,
        prefix: Optional[str] = None,
        sync_mode: str = "full",
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

    def create_google_drive(
        self,
        *,
        name: str,
        folder_ids: Optional[Sequence[str]] = None,
        file_ids: Optional[Sequence[str]] = None,
        auth_mode: str = "oauth",
        oauth_credentials: Optional[Mapping[str, Any]] = None,
        include_shared_drives: Optional[bool] = None,
        sync_mode: str = "full",
        service_account_json: Optional[Mapping[str, Any]] = None,
        credentials_json: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        config_extra: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
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
