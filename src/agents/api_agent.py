"""API agent — answers questions by planning and making a guarded REST call.

Graph: START -> plan -> execute (guarded) -> summarize -> END

The LLM plans which url+method to call; the guard enforces SSRF/scheme/port/
method safety before any network call. General REST agent (any PUBLIC API) but
cannot reach internal/metadata endpoints. Implements the Agent interface.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import APIError, InternalServerError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.base import Agent, AgentResponse, Citation
from src.llm.factory import get_chat_client, get_model_name
from src.observability.spans import set_llm_usage, span
from src.tools.http_exec import guarded_fetch

_RETRY_EXCEPTIONS = (RateLimitError, InternalServerError, APIError)

PLAN_PROMPT = """You plan a single HTTP GET request to a PUBLIC REST API to help answer the user's question.

Known useful public endpoints (you may use these or another public API):
- NVD CVE API: https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=CVE-XXXX-XXXX
- GitHub REST API: https://api.github.com/... (e.g. /repos/{{owner}}/{{repo}})
- CISA / public advisory JSON endpoints

Rules:
- Only GET requests to public https URLs. Never target localhost, private IPs, or cloud metadata.
- Respond as strict JSON, no markdown: {{"url": "<full url>", "method": "GET", "reason": "<why>"}}
- If no public API call would help, respond: {{"url": "", "method": "GET", "reason": "no suitable API"}}

Question: {question}
JSON:"""

SUMMARIZE_PROMPT = """Answer the user's question using the API response below. Be concise and cite concrete values.

Question: {question}

Called: {method} {url}

API response (may be truncated):
{body}

Answer:"""


class APIState(TypedDict, total=False):
    query: str
    url: str
    method: str
    plan_reason: str
    result: dict[str, Any]
    answer: str


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


class APIAgent(Agent):
    """General REST agent with SSRF-guarded fetch."""

    name = "api"

    def __init__(self) -> None:
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(APIState)
        g.add_node("plan", self._plan)
        g.add_node("execute", self._execute)
        g.add_node("summarize", self._summarize)
        g.add_edge(START, "plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "summarize")
        g.add_edge("summarize", END)
        return g.compile()

    def _plan(self, state: APIState) -> dict[str, Any]:
        with span("api.plan"):
            resp = _chat(
                model=get_model_name(),
                messages=[{"role": "user", "content": PLAN_PROMPT.format(question=state["query"])}],
                temperature=0.0,
                max_tokens=400,
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            try:
                plan = json.loads(raw)
            except json.JSONDecodeError:
                plan = {"url": "", "method": "GET", "reason": "planner output unparseable"}
            return {
                "url": plan.get("url", ""),
                "method": plan.get("method", "GET"),
                "plan_reason": plan.get("reason", ""),
            }

    def _execute(self, state: APIState) -> dict[str, Any]:
        url = state.get("url", "")
        if not url:
            return {"result": {"error": "no URL planned", "blocked": False}}
        with span("api.execute", **{"api.url": url, "api.method": state.get("method", "GET")}) as sp:
            result = guarded_fetch(url, state.get("method", "GET"))
            sp.set_attribute("api.blocked", bool(result.get("blocked")))
            sp.set_attribute("api.status", result.get("status_code", 0))
            return {"result": result}

    def _summarize(self, state: APIState) -> dict[str, Any]:
        result = state.get("result", {})
        if result.get("blocked"):
            return {"answer": f"I couldn't make that request safely. ({result.get('error')})"}
        if result.get("error"):
            return {"answer": f"The API call didn't succeed: {result.get('error')}"}

        body = result.get("body", "")
        with span("api.summarize"):
            resp = _chat(
                model=get_model_name(),
                messages=[
                    {
                        "role": "user",
                        "content": SUMMARIZE_PROMPT.format(
                            question=state["query"],
                            method=state.get("method", "GET"),
                            url=state.get("url", ""),
                            body=body[:8000],
                        ),
                    }
                ],
                temperature=0.1,
            )
            return {"answer": (resp.choices[0].message.content or "").strip()}

    def run(self, query: str) -> AgentResponse:
        with span("agent.run", **{"agent.name": self.name, "query": query[:200]}):
            result = self.graph.invoke({"query": query})
        url = result.get("url", "")
        res = result.get("result", {})
        citations = []
        if url:
            citations.append(
                Citation(
                    source_id=f"api:{url}",
                    chunk_id="",
                    section=f"{result.get('method', 'GET')} {url}",
                    text=(res.get("body", "") or res.get("error", ""))[:400],
                )
            )
        return AgentResponse(
            answer=result.get("answer", ""),
            citations=citations,
            metadata={
                "url": url,
                "method": result.get("method", "GET"),
                "plan_reason": result.get("plan_reason", ""),
                "blocked": bool(res.get("blocked")),
                "status_code": res.get("status_code", 0),
            },
        )

    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        yield from self.graph.stream({"query": query})