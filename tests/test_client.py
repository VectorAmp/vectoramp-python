from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from vectoramp import (
    APIError,
    AuthenticationError,
    Dataset,
    FileUploadSource,
    GCSSource,
    GenericSource,
    GoogleDriveSource,
    JiraSource,
    S3Source,
    VectorAmp,
    WebSource,
)


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
    result = client.datasets.create(
        name="docs",
        dim=2560,
        filters={"category": "string"},
        metadata_schema={"title": {"type": "string"}},
        tuning={"replicas": 1},
    )

    assert result["id"] == "ds_1"
    assert seen["url"] == "https://api.test/datasets"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["body"]["index_type"] == "sable"
    assert seen["body"]["embedding"] == {
        "provider": "vectoramp",
        "model": "VectorAmp-Embedding-2560",
    }
    assert seen["body"]["filters"] == {"category": "string"}
    assert seen["body"]["metadata_schema"] == {"title": {"type": "string"}}
    assert seen["body"]["tuning"] == {"replicas": 1}


def test_list_get_delete_and_stats() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.method == "GET" and request.url.path == "/datasets":
            return json_response(
                {"datasets": [{"id": "ds_1", "name": "docs"}], "total": 1, "limit": 25, "offset": 5}
            )
        if request.method == "GET" and request.url.path == "/datasets/ds_1/stats":
            return json_response({"vector_count": 2})
        if request.method == "GET":
            return json_response({"id": "ds_1"})
        return httpx.Response(204)

    client = make_client(handler)
    page = client.datasets.list(limit=25, offset=5)
    assert page["limit"] == 25
    assert isinstance(page["datasets"][0], Dataset)
    assert page["datasets"][0].id == "ds_1"
    assert page["datasets"][0].get("name") == "docs"
    assert list(page["datasets"][0].keys()) == ["id", "name"]
    assert list(page["datasets"][0].values()) == ["ds_1", "docs"]
    assert list(page["datasets"][0].items()) == [("id", "ds_1"), ("name", "docs")]
    assert "name" in page["datasets"][0]
    assert list(iter(page["datasets"][0])) == ["id", "name"]
    assert len(page["datasets"][0]) == 2
    assert repr(page["datasets"][0]) == "Dataset(id='ds_1')"
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
    with pytest.raises(ValueError):
        client.datasets.search("ds_1", "hello", text="hello")


def test_search_text_payload() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return json_response({"results": []})

    client = make_client(handler)
    client.datasets.search(
        "ds_1",
        "hello",
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


def test_search_text_alias_payload() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return json_response({"results": []})

    client = make_client(handler)
    client.datasets.search("ds_1", search_text="hybrid query", top_k=3)
    assert seen["body"] == {"top_k": 3, "query_text": "hybrid query"}


def test_dataset_documents_list_and_download() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/datasets/ds_1/documents":
            assert dict(request.url.params) == {"limit": "10", "cursor": "doc_0", "status": "ready"}
            return json_response(
                {"documents": [{"id": "doc_1", "file_name": "a.md"}], "next_cursor": None}
            )
        if (
            request.method == "GET"
            and request.url.path == "/datasets/ds_1/documents/doc_1/download"
        ):
            return httpx.Response(
                307,
                headers={"location": "https://download.test/doc_1"},
            )
        if request.method == "GET" and str(request.url) == "https://download.test/doc_1":
            return httpx.Response(200, content=b"hello", headers={"content-type": "text/markdown"})
        return json_response({"detail": "unexpected"}, 404)

    client = make_client(handler)
    page = client.datasets.list_documents("ds_1", limit=10, cursor="doc_0", status="ready")
    assert page["documents"][0]["id"] == "doc_1"
    assert client.datasets.download_document("ds_1", "doc_1") == b"hello"

    dataset = Dataset(client.datasets, {"id": "ds_1"})
    assert dataset.list_documents(limit=10, cursor="doc_0", status="ready")["next_cursor"] is None
    assert dataset.download_document("doc_1") == b"hello"


def test_dataset_resource_instance_methods_delegate_to_services(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "upload.test":
            return httpx.Response(200)
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/datasets/ds_1/search":
            return json_response({"results": []})
        if request.url.path == "/datasets/ds_1/insert":
            return json_response({"inserted": 1})
        if request.url.path == "/datasets/ds_1/embed":
            return json_response({"embeddings": [[0.1, 0.2]]})
        if request.url.path == "/intelligence/query":
            return json_response({"answer": "yes"})
        if request.url.path == "/ingestion/jobs":
            return json_response({"job_id": "job_1"})
        if request.url.path == "/v1/sources":
            return json_response({"id": "source_2"})
        if request.url.path == "/v1/sources/source_2/upload/init":
            return json_response(
                {
                    "job_id": "job_2",
                    "uploads": [
                        {
                            "file_id": "file_1",
                            "file_name": "sample.txt",
                            "upload_url": "https://upload.test/file_1",
                        }
                    ],
                }
            )
        if request.url.path == "/v1/sources/source_2/upload/complete":
            return json_response({"job_id": "job_2", "status": "pending"})
        if request.method == "DELETE" and request.url.path == "/datasets/ds_1":
            return httpx.Response(204)
        raise AssertionError(str(request.url))

    client = make_client(handler)
    dataset = Dataset(client.datasets, {"id": "ds_1", "name": "docs"})

    assert dataset.raw_data["name"] == "docs"
    assert dataset.client is client
    assert dataset.search(text="hello", top_k=2) == {"results": []}
    assert dataset.insert([{"id": "v1", "values": [0.1], "metadata": {}}]) == {"inserted": 1}
    assert dataset.insert_vectors([{"id": "v2", "values": [0.2], "metadata": {}}]) == {
        "inserted": 1
    }
    assert dataset.add_texts(["hello"], ids=["text_1"])["inserted"] == 1
    assert dataset.ask("question?")["answer"] == "yes"
    assert dataset.ingest_source("source_1", pipeline_id="pipe_1")["job_id"] == "job_1"
    assert dataset.ingest_files([sample], source_name="upload")["status"] == "pending"
    assert dataset.delete() is None

    assert ("POST", "/datasets/ds_1/search", {"top_k": 2, "query_text": "hello"}) in calls
    assert (
        "POST",
        "/intelligence/query",
        {
            "query": "question?",
            "top_k": 5,
            "stream": False,
            "include_sources": True,
            "dataset_id": "ds_1",
        },
    ) in calls


def test_insert_vectors_and_add_texts() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("/embed"):
            body = json.loads(request.content)
            if body.get("texts") == ["single"]:
                return json_response({"embeddings": [[0.9, 1.0]]})
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
    assert client.datasets.insert("ds_1", [{"id": "three", "values": [0.5]}]) == {"inserted": 2}
    assert client.datasets.add_texts("ds_1", "single")["inserted"] == 2
    assert client.datasets.embed("ds_1", text="one") == {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    assert calls[0] == ("/datasets/ds_1/embed", {"texts": ["one", "two"]})
    assert calls[1][0] == "/datasets/ds_1/insert"
    assert calls[1][1]["vectors"][0] == {
        "id": "one-id",
        "values": [0.1, 0.2],
        "metadata": {"a": 1, "text": "one"},
    }
    assert calls[3] == ("/datasets/ds_1/embed", {"texts": ["single"]})


def test_dataset_requires_identifier() -> None:
    client = make_client(lambda request: json_response({}))
    with pytest.raises(ValueError):
        Dataset(client.datasets, {"name": "missing-id"})


def test_dataset_client_methods_require_client() -> None:
    dataset = Dataset(VectorAmp(transport=DummyTransport()).datasets, {"id": "ds_1"})
    dataset.client = None
    with pytest.raises(TypeError):
        dataset.ask("hello")
    with pytest.raises(TypeError):
        dataset.ingest_files(["a.txt"])
    with pytest.raises(TypeError):
        dataset.ingest_source("source_1")


class DummyTransport:
    def request(self, *args, **kwargs):
        return None

    def stream(self, *args, **kwargs):
        yield {}

    def close(self):
        pass


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


def test_typed_source_builders_use_ingestion_field_names() -> None:
    assert WebSource(
        name="docs",
        start_urls=["https://docs.example.com"],
        max_depth=2,
        max_pages=50,
        allowed_domains=["docs.example.com"],
        include_assets=True,
        max_assets_per_page=3,
        metadata={"dataset_id": "ds_1"},
    ).to_create_request() == {
        "name": "docs",
        "source_type": "web",
        "config": {
            "start_urls": ["https://docs.example.com"],
            "sync_mode": "full",
            "max_depth": 2,
            "max_pages": 50,
            "allowed_domains": ["docs.example.com"],
            "include_assets": True,
            "max_assets_per_page": 3,
        },
        "metadata": {"dataset_id": "ds_1"},
    }
    assert (
        S3Source(
            name="bucket",
            bucket="docs-bucket",
            region="us-east-1",
            prefix="docs/",
            file_patterns=["*.pdf"],
            max_file_size_mb=100,
        ).to_create_request()["source_type"]
        == "s3"
    )
    assert (
        WebSource(start_urls=["https://docs.example.com/path"]).to_create_request()["name"]
        == "web-docs-example-com"
    )
    assert S3Source(bucket="docs-bucket").to_create_request() == {
        "name": "s3-docs-bucket",
        "source_type": "s3",
        "config": {"bucket": "docs-bucket", "region": "us-east-1", "sync_mode": "full"},
    }
    assert GCSSource(bucket="docs-bucket", prefix="docs/").to_create_request() == {
        "name": "gcs-docs-bucket",
        "source_type": "gcs",
        "config": {"bucket": "docs-bucket", "sync_mode": "full", "prefix": "docs/"},
    }
    assert GoogleDriveSource(
        name="drive",
        folder_ids=["folder_1"],
        oauth_credentials={"token": "secret"},
        include_shared_drives=True,
    ).to_create_request() == {
        "name": "drive",
        "source_type": "gdrive",
        "config": {
            "auth_mode": "oauth",
            "sync_mode": "full",
            "folder_ids": ["folder_1"],
            "include_shared_drives": True,
            "oauth_credentials": {"token": "secret"},
        },
    }
    assert JiraSource(cloud_id="cloud_1", project_keys=["ENG"]).to_create_request() == {
        "name": "jira-eng",
        "source_type": "jira",
        "config": {
            "cloud_id": "cloud_1",
            "include_comments": True,
            "sync_mode": "full",
            "project_keys": ["ENG"],
        },
    }
    assert FileUploadSource(name="upload").to_create_request() == {
        "name": "upload",
        "source_type": "file_upload",
        "config": {"storage_provider": "s3", "sync_mode": "full"},
    }
    assert GenericSource(
        name="custom", source_type="future", config={"field": "value"}
    ).to_create_request() == {
        "name": "custom",
        "source_type": "future",
        "config": {"field": "value"},
    }
    assert (
        GenericSource(source_type="future", config={"field": "value"}).to_create_request()["name"]
        == "future"
    )


def test_source_create_helpers_and_dataset_typed_ingest() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/v1/sources":
            return json_response({"uuid": "source_new"})
        if request.url.path == "/ingestion/jobs":
            return json_response({"job_id": "job_new"})
        raise AssertionError(str(request.url))

    client = make_client(handler)
    assert client.sources is client.ingestion
    assert (
        client.sources.create_web(start_urls=["https://example.com"], max_depth=1)["uuid"]
        == "source_new"
    )
    assert client.sources.create_s3(bucket="bucket", role_arn="arn")["uuid"] == "source_new"
    assert client.sources.create_gcs(bucket="gcs-bucket", prefix="docs/")["uuid"] == "source_new"
    assert (
        client.sources.create_jira(cloud_id="cloud", project_keys=["ENG"])["uuid"] == "source_new"
    )
    assert (
        client.sources.create_google_drive(folder_ids=["folder"], include_shared_drives=True)[
            "uuid"
        ]
        == "source_new"
    )
    assert client.sources.create_file_upload(name="upload")["uuid"] == "source_new"

    dataset = Dataset(client.datasets, {"id": "ds_1"})
    assert dataset.ingest_source(
        WebSource(name="docs", start_urls=["https://docs.example.com"]), pipeline_id="pipe_1"
    ) == {"job_id": "job_new"}

    assert calls[0][2] == {
        "name": "web-example-com",
        "source_type": "web",
        "config": {
            "start_urls": ["https://example.com"],
            "sync_mode": "full",
            "max_depth": 1,
        },
    }
    assert calls[1][2]["source_type"] == "s3"
    assert calls[1][2]["name"] == "s3-bucket"
    assert calls[1][2]["config"]["region"] == "us-east-1"
    assert calls[2][2]["source_type"] == "gcs"
    assert calls[2][2]["name"] == "gcs-gcs-bucket"
    assert calls[3][2]["source_type"] == "jira"
    assert calls[3][2]["config"]["include_comments"] is True
    assert calls[4][2]["source_type"] == "gdrive"
    assert calls[4][2]["name"] == "gdrive-folder"
    assert calls[5][2]["source_type"] == "file_upload"
    assert calls[-2][2]["source_type"] == "web"
    assert calls[-1] == (
        "POST",
        "/ingestion/jobs",
        {"source_id": "source_new", "dataset_id": "ds_1", "pipeline_id": "pipe_1"},
    )


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
        if request.url.path == "/ingestion/jobs/job_1/retry" and request.method == "POST":
            return json_response({"job_id": "job_2", "status": "pending"})
        return json_response({"id": "source_1"})

    client = make_client(handler)
    assert client.ingestion.list_sources(limit=10, offset=1)["sources"] == []
    assert client.ingestion.get_source("source_1")["id"] == "source_1"
    assert client.ingestion.start_job(source_id="source_1", dataset_id="ds_1")["job_id"] == "job_1"
    assert client.ingestion.list_jobs(dataset_id="ds_1", limit=10)["jobs"] == []
    assert client.ingestion.get_job("job_1")["id"] == "source_1"
    assert client.ingestion.retry_job("job_1")["job_id"] == "job_2"
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
            body = json.loads(request.content)
            assert body["metadata"] == {"dataset_id": "ds_1"}
            assert body["name"].startswith("file-upload-sample-")
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


def test_schedules_crud_and_trigger() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, dict(request.url.params), request.content))
        if request.url.path == "/ingestion/schedules" and request.method == "GET":
            return json_response(
                {
                    "schedules": [{"id": "sch_1", "cron": "0 * * * *", "enabled": True}],
                    "total": 1,
                    "limit": 10,
                    "offset": 0,
                }
            )
        if request.url.path == "/ingestion/schedules" and request.method == "POST":
            return json_response({"id": "sch_2", "cron": "0 0 * * *"}, status_code=201)
        if request.url.path == "/ingestion/schedules/sch_1" and request.method == "GET":
            return json_response({"id": "sch_1", "cron": "0 * * * *"})
        if request.url.path == "/ingestion/schedules/sch_2" and request.method == "PATCH":
            return json_response({"id": "sch_2", "enabled": False})
        if request.url.path == "/ingestion/schedules/sch_2" and request.method == "DELETE":
            return json_response({"deleted": True})
        if request.url.path == "/ingestion/schedules/sch_1/trigger" and request.method == "POST":
            return json_response({"job_id": "job_42"}, status_code=202)
        raise AssertionError(f"Unexpected {request.method} {request.url}")

    client = make_client(handler)

    page = client.schedules.list(limit=10, offset=0)
    assert page["total"] == 1
    assert page["schedules"][0]["id"] == "sch_1"

    assert client.schedules.get("sch_1")["id"] == "sch_1"

    created = client.schedules.create(
        source_id="src_1",
        dataset_id="ds_1",
        cron="0 0 * * *",
        timezone="UTC",
    )
    assert created["id"] == "sch_2"
    create_body = json.loads(calls[2][3])
    assert create_body == {
        "source_id": "src_1",
        "dataset_id": "ds_1",
        "cron": "0 0 * * *",
        "timezone": "UTC",
    }

    updated = client.schedules.update("sch_2", enabled=False)
    assert updated["enabled"] is False
    update_body = json.loads(calls[3][3])
    assert update_body == {"enabled": False}

    assert client.schedules.delete("sch_2") == {"deleted": True}
    assert client.schedules.trigger("sch_1") == {"job_id": "job_42"}


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
