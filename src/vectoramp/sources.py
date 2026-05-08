"""Typed ingestion source builders for the VectorAmp SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable
from urllib.parse import urlparse

from .types import JSON

KnownSourceType = Literal["s3", "web", "gcs", "gdrive", "file_upload", "jira"]


@runtime_checkable
class SourceBuilder(Protocol):
    """Object that can be converted to a source-create request."""

    def to_create_request(self) -> JSON:
        """Return the body fields accepted by the source-create API."""


@dataclass(frozen=True)
class GenericSource:
    """Escape hatch for source types not yet modeled by the SDK.

    Args:
        name: Source name. Defaults to the normalized ``source_type``.
        source_type: API source type. Required.
        config: Source-specific configuration.
        description: Optional source description.
        metadata: Optional source metadata.
    """

    name: Optional[str] = None
    source_type: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None

    def to_create_request(self) -> JSON:
        """Return source-create request fields for this generic source."""
        if not self.source_type:
            raise ValueError("GenericSource requires source_type.")
        return _source_body(
            name=self.name or _default_source_name(self.source_type),
            source_type=self.source_type,
            config=self.config,
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class WebSource:
    """Web crawler ingestion source.

    Args:
        name: Source name. Defaults to ``web-{host-or-path}`` from the first URL.
        start_urls: Crawl entry URLs. At least one URL is required.
        max_depth: Optional crawl depth limit.
        max_pages: Optional page limit.
        allowed_domains: Optional domain allow-list.
        include_patterns: Optional URL include patterns.
        exclude_patterns: Optional URL exclude patterns.
        crawl_delay_seconds: Optional delay between requests.
        sync_mode: Sync mode. Defaults to ``"full"``.
        description: Optional source description.
        metadata: Optional source metadata.
        config_extra: Optional extra config fields merged into the request.
    """

    name: Optional[str] = None
    start_urls: Sequence[str] = ()
    max_depth: Optional[int] = None
    max_pages: Optional[int] = None
    allowed_domains: Optional[Sequence[str]] = None
    include_patterns: Optional[Sequence[str]] = None
    exclude_patterns: Optional[Sequence[str]] = None
    crawl_delay_seconds: Optional[float] = None
    include_assets: Optional[bool] = None
    max_assets_per_page: Optional[int] = None
    sync_mode: str = "full"
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    config_extra: Optional[Mapping[str, Any]] = None

    def to_create_request(self) -> JSON:
        """Return source-create request fields for this web source."""
        if not self.start_urls:
            raise ValueError("WebSource requires at least one start URL.")
        config: JSON = {"start_urls": list(self.start_urls), "sync_mode": self.sync_mode}
        _set_optional(config, "max_depth", self.max_depth)
        _set_optional(config, "max_pages", self.max_pages)
        _set_optional_sequence(config, "allowed_domains", self.allowed_domains)
        _set_optional_sequence(config, "include_patterns", self.include_patterns)
        _set_optional_sequence(config, "exclude_patterns", self.exclude_patterns)
        _set_optional(config, "crawl_delay_seconds", self.crawl_delay_seconds)
        _set_optional(config, "include_assets", self.include_assets)
        _set_optional(config, "max_assets_per_page", self.max_assets_per_page)
        _merge_extra(config, self.config_extra)
        return _source_body(
            name=self.name or _default_web_source_name(self.start_urls),
            source_type="web",
            config=config,
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class S3Source:
    """Amazon S3 ingestion source.

    Args:
        name: Source name. Defaults to ``s3-{bucket}``.
        bucket: S3 bucket name. Required.
        region: AWS region. Defaults to ``"us-east-1"``.
        prefix: Optional key prefix.
        sync_mode: Sync mode. Defaults to ``"full"``.
        access_key_id: Optional AWS access key id.
        secret_access_key: Optional AWS secret access key.
        role_arn: Optional role ARN.
        endpoint_url: Optional S3-compatible endpoint URL.
        file_patterns: Optional file pattern filters.
        max_file_size_mb: Optional max file size in MB.
        description: Optional source description.
        metadata: Optional source metadata.
        config_extra: Optional extra config fields merged into the request.
    """

    name: Optional[str] = None
    bucket: str = ""
    region: str = "us-east-1"
    prefix: Optional[str] = None
    sync_mode: str = "full"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    role_arn: Optional[str] = None
    endpoint_url: Optional[str] = None
    file_patterns: Optional[Sequence[str]] = None
    max_file_size_mb: Optional[int] = None
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    config_extra: Optional[Mapping[str, Any]] = None

    def to_create_request(self) -> JSON:
        """Return source-create request fields for this S3 source."""
        if not self.bucket:
            raise ValueError("S3Source requires bucket.")
        config: JSON = {"bucket": self.bucket, "region": self.region, "sync_mode": self.sync_mode}
        _set_optional(config, "prefix", self.prefix)
        _set_optional(config, "access_key_id", self.access_key_id)
        _set_optional(config, "secret_access_key", self.secret_access_key)
        _set_optional(config, "role_arn", self.role_arn)
        _set_optional(config, "endpoint_url", self.endpoint_url)
        _set_optional_sequence(config, "file_patterns", self.file_patterns)
        _set_optional(config, "max_file_size_mb", self.max_file_size_mb)
        _merge_extra(config, self.config_extra)
        return _source_body(
            name=self.name or _default_source_name("s3", self.bucket),
            source_type="s3",
            config=config,
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class GCSSource:
    """Google Cloud Storage ingestion source."""

    name: Optional[str] = None
    bucket: str = ""
    prefix: Optional[str] = None
    project_id: Optional[str] = None
    credentials_json: Optional[Mapping[str, Any]] = None
    sync_mode: str = "full"
    file_patterns: Optional[Sequence[str]] = None
    max_file_size_mb: Optional[int] = None
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    config_extra: Optional[Mapping[str, Any]] = None

    def to_create_request(self) -> JSON:
        if not self.bucket:
            raise ValueError("GCSSource requires bucket.")
        config: JSON = {"bucket": self.bucket, "sync_mode": self.sync_mode}
        _set_optional(config, "prefix", self.prefix)
        _set_optional(config, "project_id", self.project_id)
        if self.credentials_json is not None:
            config["credentials_json"] = dict(self.credentials_json)
        _set_optional_sequence(config, "file_patterns", self.file_patterns)
        _set_optional(config, "max_file_size_mb", self.max_file_size_mb)
        _merge_extra(config, self.config_extra)
        return _source_body(
            name=self.name or _default_source_name("gcs", self.bucket),
            source_type="gcs",
            config=config,
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class GoogleDriveSource:
    """Google Drive ingestion source.

    Args:
        name: Source name. Defaults to ``gdrive-{first-folder-or-file-id}`` or
            ``gdrive``.
        folder_ids: Optional folder ids to ingest.
        file_ids: Optional file ids to ingest.
        auth_mode: Auth mode. Defaults to ``"oauth"``.
        oauth_credentials: Optional OAuth credential payload.
        include_shared_drives: Optional shared-drive toggle.
        sync_mode: Sync mode. Defaults to ``"full"``.
        service_account_json: Optional service-account credential payload.
        credentials_json: Optional generic credential payload.
        description: Optional source description.
        metadata: Optional source metadata.
        config_extra: Optional extra config fields merged into the request.
    """

    name: Optional[str] = None
    folder_ids: Optional[Sequence[str]] = None
    file_ids: Optional[Sequence[str]] = None
    auth_mode: str = "oauth"
    oauth_credentials: Optional[Mapping[str, Any]] = None
    include_shared_drives: Optional[bool] = None
    sync_mode: str = "full"
    service_account_json: Optional[Mapping[str, Any]] = None
    credentials_json: Optional[Mapping[str, Any]] = None
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    config_extra: Optional[Mapping[str, Any]] = None

    def to_create_request(self) -> JSON:
        """Return source-create request fields for this Google Drive source."""
        config: JSON = {"auth_mode": self.auth_mode, "sync_mode": self.sync_mode}
        _set_optional_sequence(config, "folder_ids", self.folder_ids)
        _set_optional_sequence(config, "file_ids", self.file_ids)
        _set_optional(config, "include_shared_drives", self.include_shared_drives)
        if self.oauth_credentials is not None:
            config["oauth_credentials"] = dict(self.oauth_credentials)
        if self.service_account_json is not None:
            config["service_account_json"] = dict(self.service_account_json)
        if self.credentials_json is not None:
            config["credentials_json"] = dict(self.credentials_json)
        _merge_extra(config, self.config_extra)
        return _source_body(
            name=self.name or _default_google_drive_source_name(self.folder_ids, self.file_ids),
            source_type="gdrive",
            config=config,
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class FileUploadSource:
    """File-upload ingestion source.

    This models the source record only. Use ``dataset.ingest_files`` or
    ``client.ingestion.ingest_files`` for the local presigned-upload flow; those
    helpers auto-create a ``file_upload`` source when uploading files.

    Args:
        name: Source name. Defaults to ``"vectoramp-python-upload"``.
        storage_provider: Upload storage provider. Defaults to ``"s3"``.
        sync_mode: Sync mode. Defaults to ``"full"``.
        description: Optional source description.
        metadata: Optional source metadata.
        config_extra: Optional extra config fields merged into the request.
    """

    name: str = "vectoramp-python-upload"
    storage_provider: str = "s3"
    sync_mode: str = "full"
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    config_extra: Optional[Mapping[str, Any]] = None

    def to_create_request(self) -> JSON:
        """Return source-create request fields for this file-upload source."""
        config: JSON = {
            "storage_provider": self.storage_provider,
            "sync_mode": self.sync_mode,
        }
        _merge_extra(config, self.config_extra)
        return _source_body(
            name=self.name,
            source_type="file_upload",
            config=config,
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class JiraSource:
    """Jira ingestion source. ``include_comments`` defaults to true."""

    name: Optional[str] = None
    cloud_id: str = ""
    access_token: Optional[str] = None
    project_keys: Optional[Sequence[str]] = None
    jql: Optional[str] = None
    include_comments: bool = True
    sync_mode: str = "full"
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    config_extra: Optional[Mapping[str, Any]] = None

    def to_create_request(self) -> JSON:
        if not self.cloud_id:
            raise ValueError("JiraSource requires cloud_id.")
        config: JSON = {
            "cloud_id": self.cloud_id,
            "include_comments": self.include_comments,
            "sync_mode": self.sync_mode,
        }
        _set_optional(config, "access_token", self.access_token)
        _set_optional_sequence(config, "project_keys", self.project_keys)
        _set_optional(config, "jql", self.jql)
        _merge_extra(config, self.config_extra)
        hint = self.project_keys[0] if self.project_keys else self.cloud_id
        return _source_body(
            name=self.name or _default_source_name("jira", hint),
            source_type="jira",
            config=config,
            description=self.description,
            metadata=self.metadata,
        )


TypedSource = Union[
    GenericSource, WebSource, S3Source, GCSSource, GoogleDriveSource, FileUploadSource, JiraSource
]
SourceInput = Union[str, TypedSource]


def _source_body(
    *,
    name: str,
    source_type: str,
    config: Mapping[str, Any],
    description: Optional[str],
    metadata: Optional[Mapping[str, Any]],
) -> JSON:
    body: JSON = {"name": name, "source_type": source_type, "config": dict(config)}
    if description is not None:
        body["description"] = description
    if metadata is not None:
        body["metadata"] = dict(metadata)
    return body


def _default_web_source_name(start_urls: Sequence[str]) -> str:
    first_url = start_urls[0]
    parsed = urlparse(first_url)
    hint = parsed.netloc or parsed.path or first_url
    return _default_source_name("web", hint)


def _default_google_drive_source_name(
    folder_ids: Optional[Sequence[str]], file_ids: Optional[Sequence[str]]
) -> str:
    hint = None
    if folder_ids:
        hint = folder_ids[0]
    elif file_ids:
        hint = file_ids[0]
    return _default_source_name("gdrive", hint)


def _default_source_name(source_type: str, hint: Optional[str] = None) -> str:
    parts = [source_type.replace("_", "-")]
    if hint:
        slug = _slugify(str(hint))
        if slug:
            parts.append(slug)
    return "-".join(parts)


def _slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in slug.split("-") if part)[:48]


def _set_optional(config: JSON, key: str, value: Any) -> None:
    if value is not None:
        config[key] = value


def _set_optional_sequence(config: JSON, key: str, value: Optional[Sequence[str]]) -> None:
    if value is not None:
        config[key] = list(value)


def _merge_extra(config: JSON, extra: Optional[Mapping[str, Any]]) -> None:
    if extra is not None:
        config.update(dict(extra))
