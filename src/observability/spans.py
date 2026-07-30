"""Span helper — a context manager and a decorator for adding spans with attributes."""

from __future__ import annotations

import functools
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def span(name: str, **attrs: Any) -> Generator[trace.Span, None, None]:
    """Start a named span, attach attributes, set error status on exception."""
    tracer = trace.get_tracer("agentic-rag-platform")
    with tracer.start_as_current_span(name) as sp:
        for k, v in attrs.items():
            if v is None:
                continue
            try:
                sp.set_attribute(k, v)
            except Exception:
                sp.set_attribute(k, str(v))
        try:
            yield sp
        except Exception as e:
            sp.set_status(Status(StatusCode.ERROR, str(e)))
            sp.record_exception(e)
            raise


def traced(name: str) -> Callable[[F], F]:
    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(name):
                return fn(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return deco


def set_llm_usage(sp: trace.Span, usage: Any) -> None:
    """Attach OpenAI-style usage stats to a span."""
    if usage is None:
        return
    try:
        sp.set_attribute("llm.prompt_tokens", int(usage.prompt_tokens or 0))
        sp.set_attribute("llm.completion_tokens", int(usage.completion_tokens or 0))
        sp.set_attribute("llm.total_tokens", int(usage.total_tokens or 0))
        from src.observability.cost import COST
        model = ""
        try:
            model = sp.attributes.get("llm.model", "") if hasattr(sp, "attributes") else ""
        except Exception:
            model = ""
        cost = COST.record_llm(model, int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0))
        sp.set_attribute("llm.cost_usd", cost)
    except Exception:
        pass