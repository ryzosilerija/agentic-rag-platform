"""Runtime configuration loaded from environment variables / .env file."""
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False,
    )
    provider: Literal["gemini", "azure", "ollama"] = "gemini"
    # Embeddings default to LOCAL — keeps Gemini quota for chat, no rate limits.
    embedding_provider: Literal["gemini", "azure", "ollama", "local"] = "local"

    gemini_api_key: str = ""
    azure_openai_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_deployment_name: str = ""
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    qdrant_collection: str = "documents"
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "agentic-rag-platform"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "agentic-rag-platform"


settings = Settings()