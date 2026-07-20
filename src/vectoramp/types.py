"""Public type aliases used by the VectorAmp SDK."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, TypedDict, Union

JSON = Dict[str, Any]
Metadata = Mapping[str, Any]
Metric = Literal["cosine", "dot", "euclidean"]
MetadataFieldType = Literal["string", "u32", "i32", "i64", "f32", "f64"]


class MetadataSchemaField(TypedDict):
    """A typed metadata field declared on a dataset."""

    name: str
    type: MetadataFieldType


MetadataSchema = Sequence[MetadataSchemaField]


class EmbeddingConfig(TypedDict):
    provider: str
    model: str


# A vector record id may be a string or an integer. Integer ids are serialized
# as JSON numbers (not coerced to strings) so the API preserves them verbatim.
VectorId = Union[str, int]


class Vector(TypedDict, total=False):
    id: VectorId
    values: Sequence[float]
    metadata: Metadata


class ConversationTurn(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class Page(TypedDict):
    total: int
    limit: int
    offset: int


class DatasetPage(Page, total=False):
    datasets: List[JSON]


class SourcePage(Page, total=False):
    sources: List[JSON]


class JobPage(Page, total=False):
    jobs: List[JSON]


FilterValue = Union[str, int, float, bool, None]
Filters = Mapping[str, FilterValue]
AdvancedFilter = Mapping[str, Any]
OptionalJSON = Optional[Mapping[str, Any]]



class SourceChunk(TypedDict, total=False):
    chunk_id: str
    score: float
    text: str
    chunk_index: int
    sheet_name: str
    row_start: int
    row_end: int
    column_names: List[str]


class SourceCitation(TypedDict, total=False):
    name: str
    path: str
    url: str
    dataset_id: str
    dataset_document_id: str
    source_type: str
    content_type: str
    relevance: float
    pages: List[int]
    sheet_names: List[str]
    chunk_count: int
    preview: str
    chunks: List[SourceChunk]
    timestamp_start: Union[int, float, str]
    timestamp_end: Union[int, float, str]
    file_id: str
    thumbnail_url: str
    preview_ref: str


class RAGChunk(TypedDict, total=False):
    id: str
    text: str
    score: float
    source: str
    source_url: str
    page: Union[int, str]
    metadata: JSON


class QueryResponse(TypedDict, total=False):
    answer: str
    sources: List[SourceCitation]
    chunks: List[RAGChunk]
    message: Optional[str]
    metadata: JSON


class SessionCreateRequest(TypedDict, total=False):
    title: str
    workspace_id: str
    dataset_id: str
    metadata: JSON


class IntelligenceSession(TypedDict, total=False):
    id: str
    title: str
    workspace_id: str
    dataset_id: str
    metadata: JSON
    created_at: str
    updated_at: str


class SessionMessageCreateRequest(TypedDict, total=False):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    metadata: JSON


class SessionMessage(TypedDict, total=False):
    id: str
    session_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    metadata: JSON
    created_at: str


class SessionList(TypedDict, total=False):
    sessions: List[IntelligenceSession]


class MessageList(TypedDict, total=False):
    messages: List[SessionMessage]
