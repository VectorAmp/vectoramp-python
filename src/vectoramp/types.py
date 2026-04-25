"""Public type aliases used by the VectorAmp SDK."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, TypedDict, Union

JSON = Dict[str, Any]
Metadata = Mapping[str, Any]
Metric = Literal["cosine", "dot", "euclidean"]


class EmbeddingConfig(TypedDict):
    provider: str
    model: str


class Vector(TypedDict, total=False):
    id: str
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
