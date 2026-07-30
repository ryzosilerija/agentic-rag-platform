"""Planner-orchestrator — decomposes a question and delegates over A2A.

Graph: START -> plan -> dispatch -> compose -> END

- plan:     an LLM decomposes the question into 1..N sub-tasks, each assigned
            to a specialist agent (rag | sql | api).
- dispatch: each sub-task is sent as an A2ATask; the agent''s response comes
            back as an A2AResult. Dependent tasks run after their deps, with
            the dependency''s answer injected into the dependent query.
- compose:  an LLM composes the final answer from all A2AResults.

Genuine multi-agent coordination: one agent''s output feeds another''s input,
mediated by the typed A2A protocol.
"""

from __future__ import annotations

import json
import re
import time
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

from src.agents.api_agent import APIAgent
from src.agents.base import Agent, AgentResponse, Citation
from src.agents.rag_agent import RAGAgent
from src.agents.sql_agent import SQLAgent
from src.llm.factory import get_chat_client, get_model_name
from src.observability.spans import set_llm_usage, span
from src.orchestrator.a2a import (
    A2AResult,
    A2AStatus,
    A2ATask,
    make_result_from_agent_response,
)

_RETRY_EXCEPTIONS = (RateLimitError, InternalServerError, APIError)

PLAN_PROMPT = """You are a planning orchestrator for a multi-agent cybersecurity system.

Specialist agents you can delegate to:
- "rag": conceptual / how-to / definitional questions from OWASP & NIST docs, and specific CVE explanations.
- "sql": counting / aggregating / ranking / listing over the CISA Known Exploited Vulnerabilities (KEV) catalog.
- "api": live data from a public REST API (e.g. current GitHub repo stats).

Decompose the user''s question into an ordered list of sub-tasks.

DECOMPOSITION RULE: If answering requires BOTH (a) a count/ranking/lookup from the KEV catalog AND (b) an explanation or live-data step, you MUST create separate sub-tasks — one per agent. A single agent cannot both query the database and explain concepts.

Trigger words that signal MULTIPLE tasks: "which ... and what", "the most/top ... and explain/describe", "find ... then". When you see a question that first identifies something via data, then asks about it, split it.

Worked example:
Question: "Which vendor has the most known exploited vulnerabilities, and what is that vendor known for?"
Correct plan: task 1 = sql "Which vendor has the most entries in the KEV catalog?"; task 2 = rag "What is {{task1}} known for in cybersecurity?" (depends_on [1]).

Only use ONE sub-task when the question is genuinely answerable by a single agent.

For dependent sub-tasks, reference an earlier task''s result in the query with {{taskN}} (1-indexed), and list the dependency in "depends_on".

Respond as strict JSON, no markdown:
{{"tasks": [{{"id": 1, "agent": "sql", "query": "...", "depends_on": []}}, {{"id": 2, "agent": "rag", "query": "explain {{task1}}", "depends_on": [1]}}]}}

Question: {question}
JSON:"""

COMPOSE_PROMPT = """Compose a single, coherent answer to the user''s question from the specialist agents'' results below. Be concise and preserve concrete facts/numbers. Do not mention the internal agents or task structure.

Question: {question}

Specialist results:
{results}

Final answer:"""


class PlannerState(TypedDict, total=False):
    query: str
    tasks: list[dict[str, Any]]
    results: list[A2AResult]
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


class PlannerOrchestrator(Agent):
    """Decomposes questions and delegates to specialist agents over A2A."""

    name = "planner"

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

    def _build_graph(self):
        g = StateGraph(PlannerState)
        g.add_node("plan", self._plan)
        g.add_node("dispatch", self._dispatch)
        g.add_node("compose", self._compose)
        g.add_edge(START, "plan")
        g.add_edge("plan", "dispatch")
        g.add_edge("dispatch", "compose")
        g.add_edge("compose", END)
        return g.compile()

    def _plan(self, state: PlannerState) -> dict[str, Any]:
        with span("planner.plan") as sp:
            resp = _chat(
                model=get_model_name(),
                messages=[{"role": "user", "content": PLAN_PROMPT.format(question=state["query"])}],
                temperature=0.0,
                max_tokens=600,
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(raw)
                tasks = parsed.get("tasks", [])
            except json.JSONDecodeError:
                tasks = [{"id": 1, "agent": "rag", "query": state["query"], "depends_on": []}]
            if not tasks:
                tasks = [{"id": 1, "agent": "rag", "query": state["query"], "depends_on": []}]
            sp.set_attribute("planner.num_tasks", len(tasks))
            sp.set_attribute("planner.agents", ",".join(t.get("agent", "?") for t in tasks))
            return {"tasks": tasks}

    def _dispatch(self, state: PlannerState) -> dict[str, Any]:
        tasks = state.get("tasks", [])
        results_by_id: dict[int, A2AResult] = {}
        answers_by_id: dict[int, str] = {}

        for t in sorted(tasks, key=lambda x: x.get("id", 0)):
            agent_name = t.get("agent", "rag")
            if agent_name not in self.agents:
                agent_name = "rag"
            query = t.get("query", state["query"])

            for dep_id in t.get("depends_on", []):
                token = "{{task%d}}" % dep_id
                if token in query and dep_id in answers_by_id:
                    query = query.replace(token, answers_by_id[dep_id])
            query = re.sub(r"\{\{task\d+\}\}", "", query).strip()

            task = A2ATask(recipient=agent_name, query=query)
            with span("a2a.dispatch", **{"a2a.recipient": agent_name, "a2a.task_id": task.task_id}):
                t0 = time.perf_counter()
                try:
                    resp: AgentResponse = self.agents[agent_name].run(task.query)
                    elapsed = (time.perf_counter() - t0) * 1000
                    result = make_result_from_agent_response(task, resp, elapsed)
                except Exception as e:
                    elapsed = (time.perf_counter() - t0) * 1000
                    result = A2AResult(
                        task_id=task.task_id, sender=agent_name,
                        status=A2AStatus.ERROR, answer=f"(agent error: {e})",
                        elapsed_ms=elapsed,
                    )
            results_by_id[t.get("id", 0)] = result
            answers_by_id[t.get("id", 0)] = result.answer

        return {"results": list(results_by_id.values())}

    def _compose(self, state: PlannerState) -> dict[str, Any]:
        results = state.get("results", [])
        if len(results) == 1 and results[0].status == A2AStatus.OK:
            return {"answer": results[0].answer}

        with span("planner.compose", **{"planner.num_results": len(results)}):
            results_str = "\n\n".join(
                f"[{r.sender} agent | {r.status.value}] {r.answer}" for r in results
            )
            resp = _chat(
                model=get_model_name(),
                messages=[
                    {"role": "user", "content": COMPOSE_PROMPT.format(
                        question=state["query"], results=results_str,
                    )}
                ],
                temperature=0.1,
            )
            return {"answer": (resp.choices[0].message.content or "").strip()}

    def run(self, query: str) -> AgentResponse:
        with span("planner.run", **{"query": query[:200]}):
            result = self.graph.invoke({"query": query})
        results = result.get("results", [])
        citations = [
            Citation(source_id=c.get("source_id", ""), chunk_id="", section=c.get("section", ""), text="")
            for r in results for c in r.citations
        ]
        return AgentResponse(
            answer=result.get("answer", ""),
            citations=citations,
            metadata={
                "num_tasks": len(result.get("tasks", [])),
                "plan": [
                    {"agent": t.get("agent"), "query": t.get("query", "")[:80], "depends_on": t.get("depends_on", [])}
                    for t in result.get("tasks", [])
                ],
                "agents_used": [r.sender for r in results],
            },
        )

    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        yield from self.graph.stream({"query": query})