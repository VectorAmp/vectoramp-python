from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from vectoramp import (
    ConfluenceSource,
    GCSSource,
    GitHubSource,
    GitLabSource,
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


# --------------------------------------------------------------------------- #
# Source-control source builders (GitHub / GitLab)
# --------------------------------------------------------------------------- #


def test_github_source_builds_minimal_config() -> None:
    body = GitHubSource(installation_id=42, repositories=["octo/hello-world"]).to_create_request()

    assert body == {
        "name": "github-octo-hello-world",
        "source_type": "github",
        "config": {"installation_id": 42, "repositories": ["octo/hello-world"]},
    }


def test_github_source_serializes_all_options() -> None:
    body = GitHubSource(
        name="Platform repos",
        installation_id=7,
        repositories=["acme/api", "acme/web"],
        ref_mode="explicit",
        refs=["main", "release"],
        excluded_refs=["wip"],
        active_branch_days=30,
        include_pull_requests=False,
        include_review_threads=False,
        include_direct_commits=False,
        include_globs=["docs/**"],
        exclude_globs=["**/*.lock"],
        max_file_size_bytes=2_000_000,
        sync_mode="full",
        description="Platform code",
        metadata={"team": "platform"},
        config_extra={"experimental": True},
    ).to_create_request()

    assert body == {
        "name": "Platform repos",
        "source_type": "github",
        "config": {
            "installation_id": 7,
            "repositories": ["acme/api", "acme/web"],
            "ref_mode": "explicit",
            "refs": ["main", "release"],
            "excluded_refs": ["wip"],
            "active_branch_days": 30,
            "include_pull_requests": False,
            "include_review_threads": False,
            "include_direct_commits": False,
            "include_globs": ["docs/**"],
            "exclude_globs": ["**/*.lock"],
            "max_file_size_bytes": 2_000_000,
            "sync_mode": "full",
            "experimental": True,
        },
        "description": "Platform code",
        "metadata": {"team": "platform"},
    }


@pytest.mark.parametrize(
    "source",
    [
        GitHubSource(repositories=["octo/hello-world"]),
        GitHubSource(installation_id=-1, repositories=["octo/hello-world"]),
        GitHubSource(installation_id=42),
    ],
)
def test_github_source_requires_installation_and_repositories(source: GitHubSource) -> None:
    with pytest.raises(ValueError):
        source.to_create_request()


def test_gitlab_source_builds_minimal_config() -> None:
    body = GitLabSource(projects=["mygroup/myproject"]).to_create_request()

    assert body == {
        "name": "gitlab-mygroup-myproject",
        "source_type": "gitlab",
        "config": {
            "auth_mode": "oauth",
            "gitlab_url": "https://gitlab.com",
            "projects": ["mygroup/myproject"],
        },
    }


def test_gitlab_source_names_from_group_when_no_project() -> None:
    body = GitLabSource(groups=["mygroup"]).to_create_request()

    assert body["name"] == "gitlab-mygroup"
    assert body["config"]["groups"] == ["mygroup"]
    assert "projects" not in body["config"]


def test_gitlab_source_serializes_token_auth_and_options() -> None:
    body = GitLabSource(
        name="Self-managed",
        groups=["platform"],
        projects=["platform/api"],
        auth_mode="token",
        gitlab_url="https://gitlab.example.com",
        access_token="glpat-secret",
        ref_mode="active",
        refs=["main"],
        excluded_refs=["scratch"],
        active_branch_days=14,
        include_merge_requests=False,
        include_review_threads=False,
        include_direct_commits=False,
        include_globs=["src/**"],
        exclude_globs=["**/*.png"],
        max_file_size_bytes=500_000,
        sync_mode="full",
    ).to_create_request()

    assert body["config"] == {
        "auth_mode": "token",
        "gitlab_url": "https://gitlab.example.com",
        "groups": ["platform"],
        "projects": ["platform/api"],
        "access_token": "glpat-secret",
        "ref_mode": "active",
        "refs": ["main"],
        "excluded_refs": ["scratch"],
        "active_branch_days": 14,
        "include_merge_requests": False,
        "include_review_threads": False,
        "include_direct_commits": False,
        "include_globs": ["src/**"],
        "exclude_globs": ["**/*.png"],
        "max_file_size_bytes": 500_000,
        "sync_mode": "full",
    }


def test_gitlab_source_supports_connection_id() -> None:
    config = GitLabSource(projects=["g/p"], connection_id="conn_gl").to_create_request()["config"]

    assert config["connection_id"] == "conn_gl"
    assert "access_token" not in config


def test_gitlab_source_requires_group_or_project() -> None:
    with pytest.raises(ValueError):
        GitLabSource().to_create_request()


def test_create_github_and_gitlab_post_to_sources() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content)))
        return json_response({"id": "src_new"}, status_code=201)

    client = make_client(handler)

    assert client.sources.create_github(
        installation_id=42, repositories=["octo/hello-world"]
    ) == {"id": "src_new"}
    assert client.sources.create_gitlab(
        projects=["mygroup/myproject"], auth_mode="token", access_token="glpat-secret"
    ) == {"id": "src_new"}

    assert calls[0] == (
        "/ingestion/sources",
        {
            "name": "github-octo-hello-world",
            "source_type": "github",
            "config": {"installation_id": 42, "repositories": ["octo/hello-world"]},
        },
    )
    assert calls[1] == (
        "/ingestion/sources",
        {
            "name": "gitlab-mygroup-myproject",
            "source_type": "gitlab",
            "config": {
                "auth_mode": "token",
                "gitlab_url": "https://gitlab.com",
                "projects": ["mygroup/myproject"],
                "access_token": "glpat-secret",
            },
        },
    )


def test_validate_source_accepts_source_control_builders() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return json_response({"valid": True})

    client = make_client(handler)

    client.sources.validate_source(GitHubSource(installation_id=1, repositories=["o/r"]))
    client.sources.validate_source(GitLabSource(groups=["g"]))

    # Validation bodies carry source_type + config only (no name).
    assert calls[0] == {
        "source_type": "github",
        "config": {"installation_id": 1, "repositories": ["o/r"]},
    }
    assert calls[1] == {
        "source_type": "gitlab",
        "config": {
            "auth_mode": "oauth",
            "gitlab_url": "https://gitlab.com",
            "groups": ["g"],
        },
    }
