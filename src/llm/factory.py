"""LLM provider factory (chat).

When LANGSMITH_TRACING is enabled, the returned client is wrapped with
langsmith.wrappers.wrap_openai so all chat.completions.create calls
auto-appear as traces in the LangSmith project.

gpt-5 family models reject `max_tokens` (require `max_completion_tokens`)
and only accept the default temperature. A thin wrapper normalizes those
so the rest of the codebase can stay provider-agnostic.
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


def _needs_gpt5_compat(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


class _CompatChatCompletions:
    """Wraps chat.completions to normalize params for gpt-5 / o-series models."""

    def __init__(self, inner):
        self._inner = inner

    def create(self, **kwargs):
        model = kwargs.get("model", "")
        if _needs_gpt5_compat(model):
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            # These models only allow the default temperature (1).
            if "temperature" in kwargs and kwargs["temperature"] != 1:
                kwargs.pop("temperature")
        return self._inner.create(**kwargs)


class _CompatChat:
    def __init__(self, inner):
        self.completions = _CompatChatCompletions(inner.completions)


class _CompatClient:
    """Thin proxy exposing .chat.completions.create with gpt-5 normalization."""

    def __init__(self, inner):
        self._inner = inner
        self.chat = _CompatChat(inner.chat)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _maybe_wrap(client):
    if settings.langsmith_tracing and settings.langsmith_api_key:
        try:
            from langsmith.wrappers import wrap_openai
            client = wrap_openai(client)
        except Exception as e:
            print(f"  ! LangSmith wrap_openai failed ({e}); continuing without wrap.")
    return client


def get_chat_client():
    if settings.provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is empty; set it in .env")
        client = OpenAI(api_key=settings.gemini_api_key, base_url=GEMINI_OPENAI_BASE_URL)
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

    client = _maybe_wrap(client)
    # Apply gpt-5 param normalization for Azure gpt-5 / o-series deployments.
    if settings.provider == "azure" and _needs_gpt5_compat(get_model_name()):
        client = _CompatClient(client)
    return client


def get_model_name() -> str:
    if settings.provider == "azure" and settings.azure_deployment_name:
        return settings.azure_deployment_name
    return _MODEL_BY_PROVIDER[settings.provider]


def get_judge_model_name() -> str:
    if settings.provider == "azure" and settings.azure_deployment_name:
        return settings.azure_deployment_name
    return _JUDGE_MODEL_BY_PROVIDER[settings.provider]