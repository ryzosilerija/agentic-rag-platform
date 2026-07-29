"""FastAPI app: /chat (JSON) and /chat/stream (SSE) endpoints.

Routes through the Supervisor, which classifies each query and dispatches to
the RAG agent (conceptual/how-to) or SQL agent (quantitative/aggregate).
The response metadata includes `routed_to` so clients see which agent answered.

Run:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agents.base import Citation
from src.orchestrator.supervisor import Supervisor
from src.observability.tracing import init_tracing


_supervisor: Supervisor | None = None


def get_supervisor() -> Supervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor()
    return _supervisor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_tracing()
    _ = get_supervisor()
    yield


app = FastAPI(title="Agentic RAG Platform", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)


class ChatRequest(BaseModel):
    query: str


def _citation_dict(c: Citation) -> dict[str, str]:
    return {"source_id": c.source_id, "chunk_id": c.chunk_id, "section": c.section, "text": c.text}


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "name": "agentic-rag-platform"}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    resp = get_supervisor().run(req.query)
    return {
        "answer": resp.answer,
        "citations": [_citation_dict(c) for c in resp.citations],
        "metadata": resp.metadata,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    supervisor = get_supervisor()

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        for event in supervisor.stream(req.query):
            for node_name, node_state in event.items():
                payload = {"node": node_name, "state": _safe_state(node_state)}
                yield {"event": node_name, "data": json.dumps(payload)}
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(event_stream())


def _safe_state(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k == "response":
            # AgentResponse object — surface answer + routing only.
            out["answer"] = getattr(v, "answer", "")
            out["routed_to"] = getattr(v, "metadata", {}).get("routed_to", "")
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
    return out