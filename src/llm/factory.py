"""LLM provider factory: swap Gemini / Azure / Ollama via one env var."""
from __future__ import annotations
from openai import AzureOpenAI, OpenAI
from src.config import settings

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

_MODEL_BY_PROVIDER = {"gemini": "gemini-flash-latest", "azure": "gpt-4o-mini", "ollama": "qwen2.5:7b"}
_JUDGE_MODEL_BY_PROVIDER = {"gemini": "gemini-flash-latest", "azure": "gpt-4o-mini", "ollama": "qwen2.5:7b"}


def get_chat_client() -> OpenAI:
    if settings.provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is empty; set it in .env")
        return OpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_OPENAI_BASE_URL)
    if settings.provider == "azure":
        if not (settings.azure_openai_key and settings.azure_openai_endpoint):
            raise RuntimeError("Azure OpenAI env vars are not fully set")
        return AzureOpenAI(
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    if settings.provider == "ollama":
        return OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    raise ValueError(f"Unknown provider: {settings.provider}")


def get_model_name() -> str:
    if settings.provider == "azure" and settings.azure_deployment_name:
        return settings.azure_deployment_name
    return _MODEL_BY_PROVIDER[settings.provider]


def get_judge_model_name() -> str:
    if settings.provider == "azure" and settings.azure_deployment_name:
        return settings.azure_deployment_name
    return _JUDGE_MODEL_BY_PROVIDER[settings.provider]
