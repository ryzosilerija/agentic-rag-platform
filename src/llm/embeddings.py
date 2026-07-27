"""Embedding provider factory — companion to factory.py (chat).

Uses a SEPARATE provider (settings.embedding_provider, default "local") so
chat can stay on Gemini's free tier without competing for the stricter
embedding quota (100 RPM on free tier).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import AzureOpenAI, OpenAI

from src.config import settings

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Local model: BGE base, 768-dim, ~440MB one-time download, fast on CPU.
LOCAL_MODEL_NAME = "BAAI/bge-base-en-v1.5"

_EMBED_MODEL_BY_PROVIDER: dict[str, str] = {
    "gemini": "gemini-embedding-001",
    "azure": "text-embedding-3-large",
    "ollama": "nomic-embed-text",
    "local": LOCAL_MODEL_NAME,
}

_EMBED_DIM_BY_PROVIDER: dict[str, int] = {
    "gemini": 768,
    "azure": 3072,
    "ollama": 768,
    "local": 768,
}


@lru_cache(maxsize=1)
def _get_local_model() -> Any:
    """Lazy-load sentence-transformers model (downloads on first call)."""
    from sentence_transformers import SentenceTransformer
    print(f"  loading local embedding model {LOCAL_MODEL_NAME}...")
    return SentenceTransformer(LOCAL_MODEL_NAME)


def get_embedding_client() -> OpenAI:
    """Return an OpenAI-compatible client for API-based providers."""
    provider = settings.embedding_provider
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is empty; set it in .env")
        return OpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_OPENAI_BASE_URL)
    if provider == "azure":
        return AzureOpenAI(
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    if provider == "ollama":
        return OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    raise ValueError("get_embedding_client() is not valid for local provider")


def get_embedding_model() -> str:
    return _EMBED_MODEL_BY_PROVIDER[settings.embedding_provider]


def get_embedding_dim() -> int:
    return _EMBED_DIM_BY_PROVIDER[settings.embedding_provider]


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of texts using the configured embedding provider."""
    if not texts:
        return []
    provider = settings.embedding_provider

    if provider == "local":
        model = _get_local_model()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    client = get_embedding_client()
    model_name = get_embedding_model()
    dim = get_embedding_dim()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        kwargs: dict = {"model": model_name, "input": batch}
        if provider in ("gemini", "azure"):
            kwargs["dimensions"] = dim
        resp = client.embeddings.create(**kwargs)
        out.extend([d.embedding for d in resp.data])
    return out