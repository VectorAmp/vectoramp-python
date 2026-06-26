from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from vectoramp import (
    ConfluenceSource,
    GCSSource,
    GoogleDriveSource,
    JiraSource,
    VectorAmp,
    WebSource,
)


def make_client(handler):
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return VectorAmp(api_key="test-key", base_url="https://api.test", http_client=http_client)


def json_response(data: Any, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=data, headers={"content-type": "application/json"})


# --------------------------------------------------------------------------- #
# Source management
# --------------------------------------------------------------------------- #


def test_source_management_methods() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, dict(request.url.params), body))
        path = request.url.path
        if request.method == "DELETE" and path == "/ingestion/sources/src_1":
            return httpx.Response(204)
        if request.method == "GET" and path == "/ingestion/sources/unused":
            return json_response({"sources": [{"id": "src_2"}], "total": 1})
        if request.method == "POST" and path == "/ingestion/sources/cleanup":
            return json_response({"deleted": ["src_2", "src_3"], "count": 2})
        if request.method == "GET" and path == "/ingestion/sources/src_1/references":
            return json_response({"schedules": ["sch_1"], "jobs": []})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    client = make_client(handler)

    # sources and ingestion are the same object; methods are reachable on both.
    assert client.sources is client.ingestion

    assert client.ingestion.delete_source("src_1") is None
    assert client.sources.delete_source("src_1", force=True) is None

    unused = client.sources.list_unused_sources(limit=10, offset=2)
    assert unused["total"] == 1

    cleanup = client.sources.cleanup_unused_sources()
    assert cleanup == {"deleted": ["src_2", "src_3"], "count": 2}

    refs = client.sources.get_source_references("src_1")
    assert refs["schedules"] == ["sch_1"]

    assert calls[0] == ("DELETE", "/ingestion/sources/src_1", {}, None)
    assert calls[1] == ("DELETE", "/ingestion/sources/src_1", {"force": "true"}, None)
    assert calls[2] == ("GET", "/ingestion/sources/unused", {"limit": "10", "offset": "2"}, None)
    assert calls[3] == ("POST", "/ingestion/sources/cleanup", {}, None)
    assert calls[4] == ("GET", "/ingestion/sources/src_1/references", {}, None)


def test_validate_source_with_builder_and_explicit() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return json_response({"valid": True})

    client = make_client(handler)

    assert client.sources.validate_source(
        WebSource(start_urls=["https://example.com"], max_depth=1)
    ) == {"valid": True}
    # Builder body carries only source_type + config (no name).
    assert calls[0] == {
        "source_type": "web",
        "config": {"start_urls": ["https://example.com"], "max_depth": 1},
    }

    assert client.sources.validate_source(
        source_type="s3", config={"bucket": "b"}
    ) == {"valid": True}
    assert calls[1] == {"source_type": "s3", "config": {"bucket": "b"}}


def test_validate_source_requires_builder_or_kwargs() -> None:
    client = make_client(lambda request: json_response({}))
    with pytest.raises(TypeError):
        client.sources.validate_source()
    with pytest.raises(TypeError):
        client.sources.validate_source(source_type="s3")


# --------------------------------------------------------------------------- #
# connection_id serialization on source builders
# --------------------------------------------------------------------------- #


def test_connection_id_serializes_into_config() -> None:
    assert (
        GoogleDriveSource(folder_ids=["f1"], connection_id="conn_gd").to_create_request()[
            "config"
        ]["connection_id"]
        == "conn_gd"
    )
    assert (
        GCSSource(bucket="b", connection_id="conn_gcs").to_create_request()["config"][
            "connection_id"
        ]
        == "conn_gcs"
    )
    assert (
        JiraSource(cloud_id="c", connection_id="conn_jira").to_create_request()["config"][
            "connection_id"
        ]
        == "conn_jira"
    )
    assert (
        ConfluenceSource(cloud_id="c", connection_id="conn_conf").to_create_request()["config"][
            "connection_id"
        ]
        == "conn_conf"
    )


def test_connection_id_omitted_when_unset() -> None:
    # When connection_id is not provided it must not appear in config.
    for source in (
        GoogleDriveSource(folder_ids=["f1"]),
        GCSSource(bucket="b"),
        JiraSource(cloud_id="c"),
        ConfluenceSource(cloud_id="c"),
    ):
        assert "connection_id" not in source.to_create_request()["config"]


# --------------------------------------------------------------------------- #
# Connections resource
# --------------------------------------------------------------------------- #


def test_connections_crud() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, dict(request.url.params), body))
        path = request.url.path
        if request.method == "GET" and path == "/connections":
            return json_response({"connections": [{"id": "conn_1"}]})
        if request.method == "POST" and path == "/connections":
            return json_response(
                {
                    "id": "conn_1",
                    "provider": "google_drive",
                    "status": "pending",
                    "authorization_url": "https://auth.test/go",
                },
                201,
            )
        if request.method == "GET" and path == "/connections/conn_1":
            return json_response({"id": "conn_1", "status": "connected"})
        if request.method == "DELETE" and path == "/connections/conn_1":
            return json_response({"deleted": True})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    client = make_client(handler)

    assert client.connections.list()["connections"][0]["id"] == "conn_1"
    assert client.connections.list(provider="google_drive")["connections"][0]["id"] == "conn_1"

    created = client.connections.create("google_drive", source_type="gdrive")
    assert created["authorization_url"] == "https://auth.test/go"

    assert client.connections.get("conn_1")["status"] == "connected"
    assert client.connections.delete("conn_1") == {"deleted": True}

    # provider param only present when supplied.
    assert calls[0] == ("GET", "/connections", {}, None)
    assert calls[1] == ("GET", "/connections", {"provider": "google_drive"}, None)
    assert calls[2] == (
        "POST",
        "/connections",
        {},
        {"provider": "google_drive", "source_type": "gdrive"},
    )


def test_connections_create_without_source_type() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return json_response({"id": "conn_1", "status": "pending"})

    client = make_client(handler)
    client.connections.create("atlassian")
    assert seen["body"] == {"provider": "atlassian"}


def test_connect_polls_until_connected_with_custom_on_url() -> None:
    captured = {}
    state = {"polls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/connections":
            return json_response(
                {
                    "id": "conn_9",
                    "provider": "google_drive",
                    "status": "pending",
                    "authorization_url": "https://auth.test/conn_9",
                }
            )
        if request.method == "GET" and request.url.path == "/connections/conn_9":
            state["polls"] += 1
            status = "connected" if state["polls"] >= 2 else "pending"
            return json_response({"id": "conn_9", "status": status})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    client = make_client(handler)
    result = client.connections.connect(
        "google_drive",
        on_url=lambda url: captured.setdefault("url", url),
        poll_interval=0,
        timeout=5,
    )
    assert result == {"id": "conn_9", "status": "connected"}
    assert captured["url"] == "https://auth.test/conn_9"
    assert state["polls"] == 2


def test_connect_default_on_url_prints(capsys) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/connections":
            return json_response(
                {
                    "id": "conn_p",
                    "status": "pending",
                    "authorization_url": "https://auth.test/conn_p",
                }
            )
        if request.method == "GET" and request.url.path == "/connections/conn_p":
            return json_response({"id": "conn_p", "status": "connected"})
        raise AssertionError(str(request.url))

    client = make_client(handler)
    client.connections.connect("google_drive", poll_interval=0, timeout=5)
    out = capsys.readouterr().out
    assert "https://auth.test/conn_p" in out


def test_connect_skips_on_url_when_absent() -> None:
    flag = {"called": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/connections":
            # No authorization_url and already connected.
            return json_response({"id": "conn_x", "status": "connected"})
        if request.method == "GET" and request.url.path == "/connections/conn_x":
            return json_response({"id": "conn_x", "status": "connected"})
        raise AssertionError(str(request.url))

    client = make_client(handler)
    client.connections.connect(
        "google_drive",
        on_url=lambda url: flag.__setitem__("called", True),
        poll_interval=0,
        timeout=5,
    )
    assert flag["called"] is False


def test_connect_times_out() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/connections":
            return json_response(
                {"id": "conn_t", "status": "pending", "authorization_url": "https://auth.test/t"}
            )
        if request.method == "GET" and request.url.path == "/connections/conn_t":
            return json_response({"id": "conn_t", "status": "pending"})
        raise AssertionError(str(request.url))

    client = make_client(handler)
    with pytest.raises(TimeoutError):
        client.connections.connect(
            "google_drive", on_url=lambda url: None, poll_interval=0, timeout=0
        )


def test_connect_requires_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response({"provider": "google_drive", "status": "pending"})

    client = make_client(handler)
    with pytest.raises(ValueError):
        client.connections.connect("google_drive", poll_interval=0, timeout=5)
