"""Central tracing setup — LangSmith (LLM-native) + OpenTelemetry (spans)."""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.config import settings

_INITIALIZED = False


def init_tracing() -> None:
    """Initialize OTel + LangSmith. Idempotent."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        print(f"  LangSmith tracing -> project={settings.langsmith_project}")

    resource = Resource.create({SERVICE_NAME: settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    try:
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        print(f"  OTel spans -> {settings.otel_exporter_otlp_endpoint}")
    except Exception as e:
        print(f"  ! OTel exporter setup failed ({e}); spans will be dropped.")

    trace.set_tracer_provider(provider)
    _INITIALIZED = True


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("agentic-rag-platform")