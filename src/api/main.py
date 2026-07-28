"""FastAPI app: /chat (JSON) and /chat/stream (SSE) endpoints.

Run:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agents.base import Citation
from src.agents.rag_agent import RAGAgent


_agent: RAGAgent | None = None


def get_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        _agent = RAGAgent()
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _ = get_agent()
    yield


app = FastAPI(title="Agentic RAG Platform", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str


def _citation_dict(c: Citation) -> dict[str, str]:
    return {"source_id": c.source_id, "chunk_id": c.chunk_id, "section": c.section, "text": c.text}


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "name": "agentic-rag-platform"}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    resp = get_agent().run(req.query)
    return {
        "answer": resp.answer,
        "citations": [_citation_dict(c) for c in resp.citations],
        "metadata": resp.metadata,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> EventSourceResponse:
    agent = get_agent()

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        for event in agent.stream(req.query):
            for node_name, node_state in event.items():
                payload = {"node": node_name, "state": _safe_state(node_state)}
                yield {"event": node_name, "data": json.dumps(payload)}
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(event_stream())


def _safe_state(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k == "retrieved" and isinstance(v, list):
            out[k] = [
                {"source": p.get("source_id"), "section": p.get("section"), "score": s}
                for _pid, s, p in v[:5]
            ]
        elif k == "citations" and isinstance(v, list):
            out[k] = [_citation_dict(c) for c in v]
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
    return out