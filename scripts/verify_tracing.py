"""M5 verification: run one query with tracing on, print where to look.

Usage:
    python -m scripts.verify_tracing
"""

from __future__ import annotations

import time

from src.agents.rag_agent import RAGAgent
from src.config import settings
from src.observability.tracing import init_tracing


def main() -> int:
    init_tracing()
    agent = RAGAgent()

    query = "What is the CVSS severity of CVE-2021-44228?"
    print(f"\nQuery: {query!r}\n")
    t0 = time.perf_counter()
    resp = agent.run(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Answer:\n{resp.answer}\n")
    print(f"Latency:        {elapsed_ms:.0f} ms")
    print(f"Retrieved docs: {resp.metadata.get('num_retrieved')}")
    print(f"Tools called:   {resp.metadata.get('tools_called')}")

    # Force any batched spans to flush before we exit.
    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=5000)
    except Exception:
        pass

    print("\n" + "=" * 66)
    print(" M5 verification complete.")
    print("=" * 66)
    print("\n[ Jaeger UI (OTel spans) ]")
    print("  Open: http://localhost:16686")
    print(f"  Service: {settings.otel_service_name}")
    print("  Look for: agent.run trace with nested rewrite/retrieve/rerank/call_tools/synthesize spans")
    if settings.langsmith_tracing and settings.langsmith_api_key:
        print("\n[ LangSmith (LLM traces) ]")
        print(f"  Project: {settings.langsmith_project}")
        print("  Open: https://smith.langchain.com/")
    else:
        print("\n[ LangSmith ]")
        print("  Not enabled. Add these to .env to see LLM-native traces:")
        print("    LANGSMITH_TRACING=true")
        print("    LANGSMITH_API_KEY=<your_key>")
        print("  Get a free key at https://smith.langchain.com/settings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())