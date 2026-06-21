"""VectorAmp Python SDK."""

from .client import VectorAmp
from .embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDINGS,
    OPENAI_TEXT_EMBEDDING_3_LARGE,
    OPENAI_TEXT_EMBEDDING_3_SMALL,
    VECTORAMP_EMBEDDING_4B,
    openai,
)
from .exceptions import APIError, AuthenticationError, VectorAmpError
from .resources import Dataset
from .sources import (
    ConfluenceSource,
    FileUploadSource,
    GCSSource,
    GenericSource,
    GoogleDriveSource,
    JiraSource,
    S3Source,
    WebSource,
)

__all__ = [
    "APIError",
    "AuthenticationError",
    "ConfluenceSource",
    "Dataset",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDINGS",
    "FileUploadSource",
    "GCSSource",
    "GenericSource",
    "GoogleDriveSource",
    "JiraSource",
    "OPENAI_TEXT_EMBEDDING_3_LARGE",
    "OPENAI_TEXT_EMBEDDING_3_SMALL",
    "S3Source",
    "VectorAmp",
    "VECTORAMP_EMBEDDING_4B",
    "VectorAmpError",
    "WebSource",
    "openai",
]
