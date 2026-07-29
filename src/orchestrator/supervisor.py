"""Supervisor orchestrator — routes a query to the right agent.

Graph: START -> route -> (rag | sql | api) -> END

Router picks:
  - "rag": conceptual / how-to / definitional questions from OWASP+NIST docs.
  - "sql": quantitative / aggregate / list questions over the KEV catalog.
  - "api": questions needing a LIVE external REST API call (fetching current
    data from a public API not in our corpus or database).

All agents share the Agent interface — adding one = register + one prompt line.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import APIError, InternalServerError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.api_agent import APIAgent
from src.agents.base import Agent, AgentResponse
from src.agents.rag_agent import RAGAgent
from src.agents.sql_agent import SQLAgent
from src.llm.factory import get_chat_client, get_model_name
from src.observability.spans import set_llm_usage, span

_RETRY_EXCEPTIONS = (RateLimitError, InternalServerError, APIError)

AgentName = Literal["rag", "sql", "api"]

ROUTER_PROMPT = """You are a router that decides which specialist agent should answer a cybersecurity question.

Agents:
- "rag": conceptual, how-to, definitional, or explanatory questions using a knowledge base of OWASP and NIST security documentation. Examples: "How do I prevent SQL injection?", "What is broken access control?", "What is the CVSS severity of CVE-2021-44228?"
- "sql": quantitative questions that require counting, aggregating, filtering, or listing entries from a structured database of the CISA Known Exploited Vulnerabilities (KEV) catalog. Examples: "How many Microsoft vulnerabilities are in the catalog?", "Which vendor has the most exploited vulnerabilities?"
- "api": questions that need a LIVE call to an external public REST API to fetch current data NOT in our docs or database — e.g. live GitHub repository info, real-time data from a public web API. Examples: "How many open issues does the langchain GitHub repo have?", "Fetch the latest release of a public project."

Rules of thumb:
- NUMBER/COUNT/RANKING/LIST from the KEV catalog -> "sql"
- HOW/WHY/WHAT IS / guidance / a specific CVE's details -> "rag"
- Needs fresh data from an external live web API -> "api"

Respond with exactly one word: rag OR sql OR api

Question: {query}
Agent:"""


class SupervisorState(TypedDict, total=False):
    query: str
    route: AgentName
    response: AgentResponse


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
)
def _chat(**kwargs: Any) -> Any:
    with span("llm.chat", **{"llm.model": kwargs.get("model", "")}) as sp:
        resp = get_chat_client().chat.completions.create(**kwargs)
        set_llm_usage(sp, getattr(resp, "usage", None))
        return resp


class Supervisor(Agent):
    """Routes queries to the RAG, SQL, or API agent."""

    name = "supervisor"

    def __init__(
        self,
        rag_agent: RAGAgent | None = None,
        sql_agent: SQLAgent | None = None,
        api_agent: APIAgent | None = None,
    ) -> None:
        self.agents: dict[str, Agent] = {
            "rag": rag_agent or RAGAgent(),
            "sql": sql_agent or SQLAgent(),
            "api": api_agent or APIAgent(),
        }
        self.graph = self._build_graph()

    def classify(self, query: str) -> AgentName:
        """LLM router: return 'rag', 'sql', or 'api'."""
        with span("supervisor.route") as sp:
            try:
                resp = _chat(
                    model=get_model_name(),
                    messages=[{"role": "user", "content": ROUTER_PROMPT.format(query=query)}],
                    temperature=0.0,
                    max_tokens=200,
                )
                raw = (resp.choices[0].message.content or "").strip().lower()
            except _RETRY_EXCEPTIONS:
                raw = "rag"
            if "sql" in raw:
                route: AgentName = "sql"
            elif "api" in raw:
                route = "api"
            else:
                route = "rag"
            sp.set_attribute("supervisor.route", route)
            return route

    def _build_graph(self):
        g = StateGraph(SupervisorState)
        g.add_node("route", self._route_node)
        g.add_node("rag", self._rag_node)
        g.add_node("sql", self._sql_node)
        g.add_node("api", self._api_node)
        g.add_edge(START, "route")
        g.add_conditional_edges(
            "route", lambda s: s["route"],
            {"rag": "rag", "sql": "sql", "api": "api"},
        )
        g.add_edge("rag", END)
        g.add_edge("sql", END)
        g.add_edge("api", END)
        return g.compile()

    def _route_node(self, state: SupervisorState) -> dict[str, Any]:
        return {"route": self.classify(state["query"])}

    def _rag_node(self, state: SupervisorState) -> dict[str, Any]:
        return {"response": self.agents["rag"].run(state["query"])}

    def _sql_node(self, state: SupervisorState) -> dict[str, Any]:
        return {"response": self.agents["sql"].run(state["query"])}

    def _api_node(self, state: SupervisorState) -> dict[str, Any]:
        return {"response": self.agents["api"].run(state["query"])}

    def run(self, query: str) -> AgentResponse:
        with span("supervisor.run", **{"query": query[:200]}):
            result = self.graph.invoke({"query": query})
        resp: AgentResponse = result["response"]
        resp.metadata["routed_to"] = result.get("route", "")
        return resp

    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        yield from self.graph.stream({"query": query})