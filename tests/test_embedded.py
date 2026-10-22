from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from vectoramp import VectorAmp
from vectoramp.embedded import EmbeddedTransport
from vectoramp.exceptions import APIError


class FakeFilter:
    def __init__(self) -> None:
        self.equalities = {}
        self.ranges = []
        self.predicates = []

    def eq(self, key: str, value: str) -> None:
        self.equalities[key] = value

    def range_i32(self, key: str, minimum: int, maximum: int) -> None:
        self.ranges.append((key, minimum, maximum))

    def compare_i32(self, key: str, op: str, value: int) -> None:
        self.predicates.append((key, op, value))


class FakeCollection:
    saved = {}

    def __init__(self, path: str, dimension: int) -> None:
        self.path = path
        self.dimension = dimension
        self.records = []
        self.indexed = False

    @classmethod
    def create(cls, path: str, dimension: int, **_: object) -> "FakeCollection":
        value = cls(path, dimension)
        cls.saved[path] = value
        return value

    @classmethod
    def open(cls, path: str) -> "FakeCollection":
        return cls.saved[path]

    def insert(self, ids, vectors, metadata) -> None:
        self.records.extend(zip(ids, vectors, metadata))

    def search(self, query, top_k, filter_value=None):
        records = self.records
        if filter_value is not None:
            records = [
                record
                for record in records
                if all(
                    record[2].get(key) == value for key, value in filter_value.equalities.items()
                )
            ]
            for key, op, expected in filter_value.predicates:
                records = [
                    record
                    for record in records
                    if {
                        "eq": int(record[2][key]) == expected,
                        "gt": int(record[2][key]) > expected,
                        "ge": int(record[2][key]) >= expected,
                        "gte": int(record[2][key]) >= expected,
                        "lt": int(record[2][key]) < expected,
                        "le": int(record[2][key]) <= expected,
                        "lte": int(record[2][key]) <= expected,
                    }[op]
                ]
        return [
            SimpleNamespace(id=record[0], distance=float(index), metadata=record[2])
            for index, record in enumerate(records[:top_k])
        ]

    def optimize(self) -> None:
        self.indexed = True

    def status(self):
        return SimpleNamespace(
            state="indexed" if self.indexed else "exact",
            vector_count=len(self.records),
            training_threshold=10_000,
            training_error=None,
        )


def test_embedded_client_uses_the_normal_dataset_api(tmp_path, monkeypatch) -> None:
    FakeCollection.saved.clear()
    monkeypatch.setitem(
        sys.modules,
        "vectoramp_sable",
        SimpleNamespace(SableCollection=FakeCollection, SableFilter=FakeFilter),
    )
    client = VectorAmp.embedded(tmp_path)
    dataset = client.datasets.create(
        name="docs",
        dim=3,
        metadata_schema={"tenant": "string"},
        tuning={"training_threshold": 256},
    )
    assert dataset.insert(
        [
            {"id": 1, "values": [1.0, 0.0, 0.0], "metadata": {"tenant": "acme"}},
            {"id": 2, "values": [0.0, 1.0, 0.0], "metadata": {"tenant": "other"}},
        ]
    ) == {"inserted": 2}
    result = dataset.search([1.0, 0.0, 0.0], filters={"tenant": "acme"})["results"]
    assert [hit["id"] for hit in result] == [1]
    assert result[0]["metadata"] == {"tenant": "acme"}
    without_metadata = dataset.search(
        [1.0, 0.0, 0.0], filters={"tenant": "acme"}, include_metadata=False
    )["results"]
    assert "metadata" not in without_metadata[0]
    ranked = dataset.search([1.0, 0.0, 0.0], top_k=2)["results"]
    assert ranked[1]["score"] == ranked[1]["distance"] == 1.0
    assert dataset.optimize()["state"] == "indexed"
    assert dataset.index_status()["vector_count"] == 2
    client.close()

    reopened = VectorAmp.embedded(tmp_path)
    assert reopened.datasets.get(dataset.id).raw_data["name"] == "docs"
    assert reopened.datasets.stats(dataset.id)["vector_count"] == 2


def test_embedded_transport_catalog_filters_and_failures(tmp_path, monkeypatch) -> None:
    FakeCollection.saved.clear()
    monkeypatch.setitem(
        sys.modules,
        "vectoramp_sable",
        SimpleNamespace(SableCollection=FakeCollection, SableFilter=FakeFilter),
    )
    transport = EmbeddedTransport(tmp_path)

    with pytest.raises(APIError, match="dim is required"):
        transport.request("POST", "/datasets", json_body={"name": "bad"})
    created = transport.request(
        "POST",
        "/datasets",
        json_body={
            "name": "docs",
            "dimension": 3,
            "metric": "l2",
            "schema": [{"name": "tenant", "type": "string"}, {"name": "year", "type": "i32"}],
        },
    )
    dataset_id = created["id"]
    assert transport.request("GET", "/datasets", params={"offset": 0, "limit": 1})["total"] == 1
    assert transport.request("GET", f"/datasets/{dataset_id}")["metric"] == "l2"

    with pytest.raises(APIError, match="non-negative integers"):
        transport.request(
            "POST",
            f"/datasets/{dataset_id}/insert",
            json_body={"vectors": [{"id": True, "values": [1.0, 0.0, 0.0]}]},
        )
    transport.request(
        "POST",
        f"/datasets/{dataset_id}/insert",
        json_body={
            "vectors": [
                {
                    "id": 9,
                    "values": [1.0, 0.0, 0.0],
                    "metadata": {"tenant": "acme", "year": 2026},
                }
            ]
        },
    )
    result = transport.request(
        "POST",
        f"/datasets/{dataset_id}/search",
        json_body={
            "query": [1.0, 0.0, 0.0],
            "filters": {"tenant": "acme"},
            "advanced_filters": [{"field": "year", "kind": "i32", "gte": 2026, "lte": 2026}],
        },
    )
    assert [hit["id"] for hit in result["results"]] == [9]
    assert result["results"][0]["metadata"] == {"tenant": "acme", "year": "2026"}
    strict = transport.request(
        "POST",
        f"/datasets/{dataset_id}/search",
        json_body={
            "query": [1.0, 0.0, 0.0],
            "filters": {"tenant": "acme"},
            "advanced_filters": [{"field": "year", "op": "lt", "value": 2027}],
        },
    )
    assert [hit["id"] for hit in strict["results"]] == [9]
    with pytest.raises(APIError, match="requires a value"):
        transport.request(
            "POST",
            f"/datasets/{dataset_id}/search",
            json_body={
                "query": [1.0, 0.0, 0.0],
                "advanced_filters": [{"field": "year", "op": "lt"}],
            },
        )
    with pytest.raises(APIError, match="unsupported embedded advanced filter"):
        transport.request(
            "POST",
            f"/datasets/{dataset_id}/search",
            json_body={
                "query": [1.0, 0.0, 0.0],
                "advanced_filters": [{"field": "year", "op": "in", "value": [2026]}],
            },
        )
    with pytest.raises(APIError, match="vector query"):
        transport.request("POST", f"/datasets/{dataset_id}/search", json_body={})
    with pytest.raises(APIError, match="uses 'bogus'.*field is 'i32'"):
        transport.request(
            "POST",
            f"/datasets/{dataset_id}/search",
            json_body={
                "query": [1.0, 0.0, 0.0],
                "advanced_filters": [{"field": "year", "kind": "bogus"}],
            },
        )
    with pytest.raises(APIError, match="not available in embedded mode"):
        transport.request("GET", "/sources")
    with pytest.raises(APIError, match="streaming route"):
        next(transport.stream("POST", "/intelligence/ask"))
    transport.request("DELETE", f"/datasets/{dataset_id}")
    with pytest.raises(APIError, match="was not found"):
        transport.request("GET", f"/datasets/{dataset_id}")

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "catalog.json").write_text(json.dumps({"format_version": 99}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported embedded VectorAmp catalog"):
        EmbeddedTransport(corrupt)


def test_embedded_delete_retains_catalog_when_storage_removal_fails(tmp_path, monkeypatch) -> None:
    FakeCollection.saved.clear()
    monkeypatch.setitem(
        sys.modules,
        "vectoramp_sable",
        SimpleNamespace(SableCollection=FakeCollection, SableFilter=FakeFilter),
    )
    transport = EmbeddedTransport(tmp_path)
    dataset = transport.request(
        "POST", "/datasets", json_body={"name": "docs", "dimension": 3}
    )

    def fail_removal(_path) -> None:
        raise PermissionError("storage is read-only")

    monkeypatch.setattr("vectoramp.embedded.shutil.rmtree", fail_removal)
    with pytest.raises(PermissionError, match="read-only"):
        transport.request("DELETE", f"/datasets/{dataset['id']}")
    assert transport.request("GET", f"/datasets/{dataset['id']}")["name"] == "docs"
