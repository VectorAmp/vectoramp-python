"""Typed ingestion source builders for the VectorAmp SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable

from .types import JSON

KnownSourceType = Literal["s3", "web", "gdrive", "file_upload"]


@runtime_checkable
class SourceBuilder(Protocol):
    """Object that can be converted to a source-create request."""

    def to_create_request(self) -> JSON:
        """Return the body fields accepted by the source-create API."""


@dataclass(frozen=True)
class GenericSource:
    """Escape hatch for source types not yet modeled by the SDK."""

    name: str
    source_type: str
    config: Mapping[str, Any]
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None

    def to_create_request(self) -> JSON:
        return _source_body(
            name=self.name,
            source_type=self.source_type,
            config=self.config,
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class WebSource:
    """Web crawler ingestion source."""

    name: str
    start_urls: Sequence[str]
    max_depth: Optional[int] = None
    max_pages: Optional[int] = None
    allowed_domains: Optional[Sequence[str]] = None
    include_patterns: Optional[Sequence[str]] = None
    exclude_patterns: Optional[Sequence[str]] = None
    crawl_delay_seconds: Optional[float] = None
    sync_mode: str = "full"
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    config_extra: Optional[Mapping[str, Any]] = None

    def to_create_request(self) -> JSON:
        config: JSON = {"start_urls": list(self.start_urls), "sync_mode": self.sync_mode}
        _set_optional(config, "max_depth", self.max_depth)
        _set_optional(config, "max_pages", self.max_pages)
        _set_optional_sequence(config, "allowed_domains", self.allowed_domains)
        _set_optional_sequence(config, "include_patterns", self.include_patterns)
        _set_optional_sequence(config, "exclude_patterns", self.exclude_patterns)
        _set_optional(config, "crawl_delay_seconds", self.crawl_delay_seconds)
        _merge_extra(config, self.config_extra)
        return _source_body(
            name=self.name,
            source_type="web",
            config=config,
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class S3Source:
    """Amazon S3 ingestion source."""

    name: str
    bucket: str
    region: str
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
            name=self.name,
            source_type="s3",
            config=config,
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class GoogleDriveSource:
    """Google Drive ingestion source."""

    name: str
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
            name=self.name,
            source_type="gdrive",
            config=config,
            description=self.description,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class FileUploadSource:
    """File-upload ingestion source.

    This models the source record only. Use ``dataset.ingest_files`` or
    ``client.ingestion.ingest_files`` for the local presigned-upload flow.
    """

    name: str = "vectoramp-python-upload"
    storage_provider: str = "s3"
    sync_mode: str = "full"
    description: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None
    config_extra: Optional[Mapping[str, Any]] = None

    def to_create_request(self) -> JSON:
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


TypedSource = Union[GenericSource, WebSource, S3Source, GoogleDriveSource, FileUploadSource]
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


def _set_optional(config: JSON, key: str, value: Any) -> None:
    if value is not None:
        config[key] = value


def _set_optional_sequence(config: JSON, key: str, value: Optional[Sequence[str]]) -> None:
    if value is not None:
        config[key] = list(value)


def _merge_extra(config: JSON, extra: Optional[Mapping[str, Any]]) -> None:
    if extra is not None:
        config.update(dict(extra))
