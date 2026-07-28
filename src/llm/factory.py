"""LLM provider factory (chat).

When LANGSMITH_TRACING is enabled, the returned client is wrapped with
langsmith.wrappers.wrap_openai so all chat.completions.create calls
auto-appear as traces in the LangSmith project.
"""

from __future__ import annotations

from openai import AzureOpenAI, OpenAI

from src.config import settings

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

_MODEL_BY_PROVIDER: dict[str, str] = {
    "gemini": "gemini-flash-latest",
    "azure": "gpt-4o-mini",
    "ollama": "qwen2.5:7b",
}

_JUDGE_MODEL_BY_PROVIDER: dict[str, str] = {
    "gemini": "gemini-flash-latest",
    "azure": "gpt-4o-mini",
    "ollama": "qwen2.5:7b",
}


def _maybe_wrap(client: OpenAI) -> OpenAI:
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        return client
    try:
        from langsmith.wrappers import wrap_openai
        return wrap_openai(client)
    except Exception as e:
        print(f"  ! LangSmith wrap_openai failed ({e}); continuing without wrap.")
        return client


def get_chat_client() -> OpenAI:
    if settings.provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is empty; set it in .env")
        client: OpenAI = OpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_OPENAI_BASE_URL)
    elif settings.provider == "azure":
        if not (settings.azure_openai_key and settings.azure_openai_endpoint):
            raise RuntimeError("Azure OpenAI env vars are not fully set")
        client = AzureOpenAI(
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    elif settings.provider == "ollama":
        client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError(f"Unknown provider: {settings.provider}")

    return _maybe_wrap(client)


def get_model_name() -> str:
    if settings.provider == "azure" and settings.azure_deployment_name:
        return settings.azure_deployment_name
    return _MODEL_BY_PROVIDER[settings.provider]


def get_judge_model_name() -> str:
    if settings.provider == "azure" and settings.azure_deployment_name:
        return settings.azure_deployment_name
    return _JUDGE_MODEL_BY_PROVIDER[settings.provider]