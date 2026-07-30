"""Agent-to-Agent (A2A) protocol — typed message envelopes.

A minimal, typed message layer that lets a coordinating agent delegate work to
specialist agents and receive structured results back. Each message is an
envelope with a stable schema so agents can interoperate. Deliberately small:
the value is the contract, not the transport (same envelopes could be
serialized over HTTP/queue for a distributed deployment).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class A2AStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    REFUSED = "refused"


@dataclass
class A2ATask:
    """A request from a coordinator to a specialist agent."""

    recipient: str
    query: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    sender: str = "planner"
    depends_on: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class A2AResult:
    """A specialist agent's structured response to an A2ATask."""

    task_id: str
    sender: str
    status: A2AStatus
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


def make_result_from_agent_response(task: A2ATask, resp: Any, elapsed_ms: float) -> A2AResult:
    """Adapt a specialist Agent's AgentResponse into an A2AResult envelope."""
    meta = getattr(resp, "metadata", {}) or {}
    refused = bool(meta.get("rejected") or meta.get("blocked"))
    status = A2AStatus.REFUSED if refused else A2AStatus.OK
    cites = [
        {"source_id": c.source_id, "section": c.section}
        for c in getattr(resp, "citations", []) or []
    ]
    return A2AResult(
        task_id=task.task_id,
        sender=task.recipient,
        status=status,
        answer=getattr(resp, "answer", ""),
        citations=cites,
        metadata=meta,
        elapsed_ms=elapsed_ms,
    )