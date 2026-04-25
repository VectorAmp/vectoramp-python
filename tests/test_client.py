from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from vectoramp import APIError, AuthenticationError, VectorAmp


def make_client(handler):
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return VectorAmp(api_key="test-key", base_url="https://api.test", http_client=http_client)


def json_response(data: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=data, headers={"content-type": "application/json"})


def test_requires_api_key() -> None:
    with pytest.raises(AuthenticationError):
        VectorAmp(api_key="")


def test_dataset_create_forces_sable_and_auth_header() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return json_response({"id": "ds_1", "index_type": "sable"}, 201)

    client = make_client(handler)
    result = client.datasets.create(name="docs", dim=2560)

    assert result["id"] == "ds_1"
    assert seen["url"] == "https://api.test/datasets"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["body"]["index_type"] == "sable"
    assert seen["body"]["embedding"] == {
        "provider": "vectoramp",
        "model": "Qwen/Qwen3-Embedding-4B",
    }


def test_list_get_delete_and_stats() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.method == "GET" and request.url.path == "/datasets":
            return json_response({"datasets": [], "total": 0, "limit": 25, "offset": 5})
        if request.method == "GET" and request.url.path == "/datasets/ds_1/stats":
            return json_response({"vector_count": 2})
        if request.method == "GET":
            return json_response({"id": "ds_1"})
        return httpx.Response(204)

    client = make_client(handler)
    assert client.datasets.list(limit=25, offset=5)["limit"] == 25
    assert client.datasets.get("ds_1")["id"] == "ds_1"
    assert client.datasets.stats("ds_1")["vector_count"] == 2
    assert client.datasets.delete("ds_1") is None
    assert calls[0] == ("GET", "/datasets", {"limit": "25", "offset": "5"})


def test_search_requires_exactly_one_query() -> None:
    client = make_client(lambda request: json_response({}))
    with pytest.raises(ValueError):
        client.datasets.search("ds_1")
    with pytest.raises(ValueError):
        client.datasets.search("ds_1", vector=[0.1], text="hello")


def test_search_text_payload() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return json_response({"results": []})

    client = make_client(handler)
    client.datasets.search(
        "ds_1",
        text="hello",
        top_k=3,
        filters={"kind": "doc"},
        advanced_filters=[{"field": "price", "op": "gt", "value": 5}],
        hybrid=True,
        sparse_query="hello",
        alpha=0.7,
        include_documents=True,
    )
    assert seen["body"] == {
        "top_k": 3,
        "query_text": "hello",
        "filters": {"kind": "doc"},
        "advanced_filters": [{"field": "price", "op": "gt", "value": 5}],
        "hybrid": True,
        "sparse_query": "hello",
        "alpha": 0.7,
        "include_documents": True,
    }


def test_insert_vectors_and_add_texts() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("/embed"):
            return json_response({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
        return json_response({"inserted": 2})

    client = make_client(handler)
    result = client.datasets.add_texts(
        "ds_1",
        ["one", "two"],
        ids=["one-id", "two-id"],
        metadatas=[{"a": 1}, {"b": 2}],
    )
    assert result == {"inserted": 2}
    assert calls[0] == ("/datasets/ds_1/embed", {"texts": ["one", "two"]})
    assert calls[1][0] == "/datasets/ds_1/insert"
    assert calls[1][1]["vectors"][0] == {
        "id": "one-id",
        "values": [0.1, 0.2],
        "metadata": {"a": 1, "text": "one"},
    }


def test_add_texts_validates_lengths() -> None:
    client = make_client(lambda request: json_response({"embeddings": []}))
    with pytest.raises(ValueError):
        client.datasets.add_texts("ds", ["a"], ids=[])
    with pytest.raises(ValueError):
        client.datasets.add_texts("ds", ["a"], metadatas=[])
    with pytest.raises(ValueError):
        client.datasets.add_texts("ds", ["a"])


def test_intelligence_query_and_top_level_ask() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return json_response({"answer": "42"})

    client = make_client(handler)
    assert client.ask("why?", dataset_id="all", top_k=2)["answer"] == "42"
    assert seen["path"] == "/intelligence/query"
    assert seen["body"] == {
        "query": "why?",
        "top_k": 2,
        "stream": False,
        "include_sources": True,
        "dataset_id": "all",
    }


def test_intelligence_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "text/event-stream"
        assert json.loads(request.content)["stream"] is True
        body = 'data: {"chunk_type":"text","content":"hi","metadata":{}}\n\n'
        body += 'data: {"chunk_type":"done","content":"","metadata":{}}\n\n'
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = make_client(handler)
    assert list(client.ask_stream("hello")) == [
        {"chunk_type": "text", "content": "hi", "metadata": {}},
        {"chunk_type": "done", "content": "", "metadata": {}},
    ]


def test_api_error_decodes_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"detail": "bad news"}, 400)

    client = make_client(handler)
    with pytest.raises(APIError) as exc:
        client.datasets.get("missing")
    assert exc.value.status_code == 400
    assert "bad news" in str(exc.value)


def test_ingestion_sources_and_jobs() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path == "/ingestion/sources":
            return json_response({"sources": [], "total": 0, "limit": 10, "offset": 1})
        if request.url.path == "/ingestion/jobs" and request.method == "GET":
            return json_response({"jobs": [], "total": 0, "limit": 10, "offset": 0})
        if request.url.path == "/ingestion/jobs" and request.method == "POST":
            return json_response({"job_id": "job_1"})
        return json_response({"id": "source_1"})

    client = make_client(handler)
    assert client.ingestion.list_sources(limit=10, offset=1)["sources"] == []
    assert client.ingestion.get_source("source_1")["id"] == "source_1"
    assert client.ingestion.start_job(source_id="source_1", dataset_id="ds_1")["job_id"] == "job_1"
    assert client.ingestion.list_jobs(dataset_id="ds_1", limit=10)["jobs"] == []
    assert client.ingestion.get_job("job_1")["id"] == "source_1"
    assert calls[0] == ("GET", "/ingestion/sources", {"limit": "10", "offset": "1"})


def test_ingest_files_upload_flow(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello")
    uploaded = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "upload.test":
            uploaded["content"] = request.content
            uploaded["content_type"] = request.headers["content-type"]
            return httpx.Response(200)
        if request.url.path == "/v1/sources":
            assert json.loads(request.content)["metadata"] == {"dataset_id": "ds_1"}
            return json_response({"id": "source_1"})
        if request.url.path == "/v1/sources/source_1/upload/init":
            assert json.loads(request.content)["files"][0]["name"] == "sample.txt"
            return json_response(
                {
                    "job_id": "job_1",
                    "uploads": [
                        {
                            "file_id": "file_1",
                            "file_name": "sample.txt",
                            "upload_url": "https://upload.test/file_1",
                        }
                    ],
                }
            )
        if request.url.path == "/v1/sources/source_1/upload/complete":
            assert json.loads(request.content) == {"job_id": "job_1", "file_ids": ["file_1"]}
            return json_response({"job_id": "job_1", "status": "pending"})
        raise AssertionError(str(request.url))

    client = make_client(handler)
    result = client.ingestion.ingest_files(dataset_id="ds_1", paths=[sample])
    assert result["status"] == "pending"
    assert uploaded == {"content": b"hello", "content_type": "text/plain"}


def test_context_manager_closes() -> None:
    closed = {"value": False}

    class Transport:
        def request(self, *args, **kwargs):
            return None

        def stream(self, *args, **kwargs):
            yield {}

        def close(self):
            closed["value"] = True

    with VectorAmp(transport=Transport()):
        pass
    assert closed["value"] is True
