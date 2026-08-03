"""Agent interface — all agents (RAG, SQL, API) implement this same shape.

Phase 2/3 will add SQLAgent and APIAgent as concrete subclasses without
touching the orchestrator, retrieval, or API layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Citation:
    source_id: str
    chunk_id: str
    section: str
    text: str
    page: int | None = None


@dataclass
class AgentResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Base class for all agents."""

    name: str

    @abstractmethod
    def run(self, query: str) -> AgentResponse:
        """Run agent to completion, return the full response."""

    @abstractmethod
    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        """Stream node-level state updates as the graph executes."""