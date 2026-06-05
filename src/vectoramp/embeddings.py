"""Embedding helpers for dataset creation."""

from __future__ import annotations

from typing import Dict, Literal, Mapping

EmbeddingConfig = Dict[str, str]
OpenAIEmbeddingSize = Literal["small", "large"]

VECTORAMP_EMBEDDING_4B = "VectorAmp-Embedding-4B"
OPENAI_TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
OPENAI_TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

EMBEDDING_DIMENSIONS: Mapping[str, int] = {
    VECTORAMP_EMBEDDING_4B: 2560,
    OPENAI_TEXT_EMBEDDING_3_SMALL: 1536,
    OPENAI_TEXT_EMBEDDING_3_LARGE: 3072,
}


def openai(size: OpenAIEmbeddingSize = "small") -> EmbeddingConfig:
    """Return an OpenAI BYOM embedding config for dataset creation."""

    return {
        "provider": "openai",
        "model": (
            OPENAI_TEXT_EMBEDDING_3_LARGE
            if size == "large"
            else OPENAI_TEXT_EMBEDDING_3_SMALL
        ),
        "secret_ref": "emb:openai:api_key",
    }


EMBEDDINGS = {
    "vectoramp_4b": {"provider": "vectoramp", "model": VECTORAMP_EMBEDDING_4B},
    "openai_small": openai("small"),
    "openai_large": openai("large"),
}
