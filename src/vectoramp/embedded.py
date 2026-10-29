"""Embedded SABLE transport for the public VectorAmp client surface.

The native runtime is imported lazily so normal hosted installations do not
depend on a platform wheel. Embedded mode intentionally implements the portable
dataset/vector subset; hosted-only ingestion and Intelligence routes fail with a
clear capability error.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Union

from .exceptions import APIError
from .transport import BaseTransport

_DATASET_PATH = re.compile(r"^/datasets/([^/]+)(?:/(search|insert|stats|optimize|index-status))?$")


class EmbeddedTransport(BaseTransport):
    """Route the SDK dataset API to local, in-process SABLE datasets."""

    def __init__(self, root: Union[str, os.PathLike[str]]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._catalog_path = self.root / "catalog.json"
        self._lock = threading.RLock()
        self._datasets: Dict[str, Any] = {}
        self._catalog = self._load_catalog()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Any:
        del headers
        method = method.upper()
        if path == "/datasets":
            if method == "POST":
                return self._create(dict(json_body or {}))
            if method == "GET":
                values = list(self._catalog["datasets"].values())
                offset = int((params or {}).get("offset") or 0)
                limit = int((params or {}).get("limit") or 50)
                return {
                    "datasets": values[offset : offset + limit],
                    "total": len(values),
                    "limit": limit,
                    "offset": offset,
                }

        match = _DATASET_PATH.match(path)
        if not match:
            raise APIError(501, f"{path} is not available in embedded mode")
        dataset_id, operation = match.groups()
        dataset = self._dataset(dataset_id)
        if operation is None and method == "GET":
            return dict(dataset)
        if operation is None and method == "DELETE":
            return self._delete(dataset_id)
        if operation == "stats" and method == "GET":
            status = self._runtime_dataset(dataset_id).status()
            return {"vector_count": status.vector_count, "index_state": status.state}
        if operation == "index-status" and method == "GET":
            status = self._runtime_dataset(dataset_id).status()
            return {
                "state": status.state,
                "vector_count": status.vector_count,
                "training_threshold": status.training_threshold,
                "training_error": status.training_error,
            }
        if operation == "optimize" and method == "POST":
            self._runtime_dataset(dataset_id).optimize()
            return self.request("GET", f"/datasets/{dataset_id}/index-status")
        if operation == "insert" and method == "POST":
            return self._insert(dataset_id, dict(json_body or {}))
        if operation == "search" and method == "POST":
            return self._search(dataset_id, dict(json_body or {}))
        raise APIError(405, f"{method} is not allowed for {path}")

    def stream(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        del method, json_body, headers
        raise APIError(501, f"streaming route {path} is not available in embedded mode")

    def close(self) -> None:
        with self._lock:
            self._datasets.clear()

    def _create(self, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from vectoramp_sable import SableDataset
        except ImportError as error:  # pragma: no cover - platform packaging gate
            raise RuntimeError("embedded mode requires vectoramp-sable-embedded-runtime") from error
        dimension = int(body.get("dim") or body.get("dimension") or 0)
        if dimension <= 0:
            raise APIError(400, "dim is required in embedded mode")
        dataset_id = str(uuid.uuid4())
        metadata_schema = self._metadata_schema(body.get("schema") or body.get("metadata_schema"))
        tuning = body.get("tuning") or {}
        dataset = {
            "id": dataset_id,
            "name": body.get("name") or dataset_id,
            "dim": dimension,
            "dimension": dimension,
            "metric": body.get("metric") or "cosine",
            "metadata_schema": metadata_schema,
            "index_type": "sable",
            "mode": "embedded",
        }
        runtime_dataset = SableDataset.create(
            str(self.root / "datasets" / dataset_id),
            dimension,
            metric=dataset["metric"],
            metadata_schema=metadata_schema,
            l0_capacity=int(tuning.get("l0_capacity", 10_000)),
            training_threshold=int(tuning.get("training_threshold", 10_000)),
        )
        with self._lock:
            self._catalog["datasets"][dataset_id] = dataset
            self._datasets[dataset_id] = runtime_dataset
            self._save_catalog()
        return dict(dataset)

    @staticmethod
    def _metadata_schema(raw_schema: Any) -> Dict[str, str]:
        if not raw_schema:
            return {}
        if isinstance(raw_schema, Mapping):
            return {
                str(name): str(config if isinstance(config, str) else config.get("type"))
                for name, config in raw_schema.items()
            }
        return {
            str(field["name"]): str(field["type"])
            for field in raw_schema
            if isinstance(field, Mapping) and "name" in field and "type" in field
        }

    def _insert(self, dataset_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        vectors = list(body.get("vectors") or [])
        ids = []
        values = []
        metadata = []
        for record in vectors:
            vector_id = record.get("id")
            if not isinstance(vector_id, int) or isinstance(vector_id, bool) or vector_id < 0:
                raise APIError(400, "embedded vector ids must be non-negative integers")
            ids.append(vector_id)
            values.append(list(record.get("values") or []))
            metadata.append(
                {key: str(value) for key, value in (record.get("metadata") or {}).items()}
            )
        self._runtime_dataset(dataset_id).insert(ids, values, metadata)
        return {"inserted": len(vectors)}

    def _search(self, dataset_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if "query" not in body:
            raise APIError(400, "embedded mode requires a vector query")
        from vectoramp_sable import SableFilter

        filter_value = None
        equality = body.get("filters") or {}
        advanced = body.get("advanced_filters") or []
        if equality or advanced:
            filter_value = SableFilter()
            for key, value in equality.items():
                filter_value.eq(key, str(value))
            for item in advanced:
                self._add_advanced_filter(
                    filter_value,
                    item,
                    self._dataset(dataset_id).get("metadata_schema") or {},
                )
        hits = self._runtime_dataset(dataset_id).search(
            list(body["query"]), int(body.get("top_k") or 10), filter_value
        )
        include_metadata = body.get("include_metadata") is not False
        return {
            "results": [
                {
                    "id": hit.id,
                    "distance": hit.distance,
                    "score": hit.distance,
                    **({"metadata": dict(hit.metadata)} if include_metadata else {}),
                }
                for hit in hits
            ]
        }

    @staticmethod
    def _add_advanced_filter(
        filter_value: Any,
        item: Mapping[str, Any],
        metadata_schema: Mapping[str, Any],
    ) -> None:
        field = str(item.get("field") or item.get("key") or "")
        if not field or field not in metadata_schema:
            raise APIError(400, f"advanced filter field {field!r} was not declared")
        declared_kind = str(metadata_schema[field]).lower()
        kind = str(item.get("kind") or item.get("type") or declared_kind).lower()
        if kind != declared_kind:
            raise APIError(
                400,
                f"advanced filter for {field!r} uses {kind!r}, but the field is {declared_kind!r}",
            )

        if "op" in item:
            op = str(item.get("op") or "").lower()
            if "value" not in item or item.get("value") is None:
                raise APIError(400, f"advanced filter {field!r} requires a value")
            if kind == "string":
                if op != "eq":
                    raise APIError(400, f"string advanced filter {op!r} is unsupported")
                filter_value.eq(field, str(item["value"]))
                return
            method = getattr(filter_value, f"compare_{kind}", None)
            if method is None or op not in {"eq", "gt", "ge", "gte", "lt", "le", "lte"}:
                raise APIError(400, f"unsupported embedded advanced filter {item!r}")
            method(field, op, item["value"])
            return

        minimum = item.get("min_inclusive", item.get("gte"))
        maximum = item.get("max_inclusive", item.get("lte"))
        method = getattr(filter_value, f"range_{kind}", None)
        if method is None or (minimum is None and maximum is None):
            raise APIError(400, f"unsupported embedded range filter {item!r}")
        method(field, minimum, maximum)

    def _dataset(self, dataset_id: str) -> Dict[str, Any]:
        try:
            return self._catalog["datasets"][dataset_id]
        except KeyError as error:
            raise APIError(404, f"dataset {dataset_id} was not found") from error

    def _runtime_dataset(self, dataset_id: str) -> Any:
        self._dataset(dataset_id)
        with self._lock:
            runtime_dataset = self._datasets.get(dataset_id)
            if runtime_dataset is None:
                from vectoramp_sable import SableDataset

                runtime_dataset = SableDataset.open(str(self.root / "datasets" / dataset_id))
                self._datasets[dataset_id] = runtime_dataset
            return runtime_dataset

    def _delete(self, dataset_id: str) -> None:
        self._dataset(dataset_id)
        with self._lock:
            self._datasets.pop(dataset_id, None)
            # Catalog removal is still required when storage was already
            # removed by an interrupted or externally completed delete.
            with suppress(FileNotFoundError):
                shutil.rmtree(self.root / "datasets" / dataset_id)
            del self._catalog["datasets"][dataset_id]
            self._save_catalog()
        return None

    def _load_catalog(self) -> Dict[str, Any]:
        if not self._catalog_path.exists():
            return {"format_version": 1, "datasets": {}}
        catalog = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        if catalog.get("format_version") != 1 or not isinstance(catalog.get("datasets"), dict):
            raise RuntimeError("unsupported embedded VectorAmp catalog")
        return catalog

    def _save_catalog(self) -> None:
        temporary = self._catalog_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self._catalog, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self._catalog_path)
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
