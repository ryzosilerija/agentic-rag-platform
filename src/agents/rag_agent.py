"""RAG agent built on LangGraph.

Graph: START -> rewrite -> retrieve -> maybe_call_tools -> synthesize -> END

M5 additions:
- Every graph node runs inside an OTel span with useful attributes.
- Every LLM call gets its own child span with token counts.
- Every tool invocation gets its own child span with name + args.
- LangSmith auto-traces the full graph when LANGSMITH_TRACING=true.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
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
from src.agents.prompts import (
    QUERY_REWRITE_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_TEMPLATE,
    TOOL_DECIDER_SYSTEM_PROMPT,
)
from src.llm.factory import get_chat_client, get_model_name
from src.mcp_server.registry import get_tool_schemas, invoke_tool
from src.observability.spans import set_llm_usage, span
from src.retrieval.hybrid import RetrievalConfig, retrieve as retrieve_docs
from src.retrieval.rerank import rerank as rerank_fn


class AgentState(TypedDict, total=False):
    query: str
    rewritten_query: str
    retrieved: list[tuple[str, float, dict[str, Any]]]
    tool_results: list[dict[str, Any]]
    answer: str
    citations: list[Citation]


_RETRY_EXCEPTIONS = (RateLimitError, InternalServerError, APIError)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
)
def _chat_with_retry(**kwargs: Any) -> Any:
    """OpenAI chat.completions.create wrapped with retry + OTel span."""
    with span("llm.chat", **{
        "llm.model": kwargs.get("model", ""),
        "llm.temperature": kwargs.get("temperature"),
        "llm.has_tools": bool(kwargs.get("tools")),
    }) as sp:
        resp = get_chat_client().chat.completions.create(**kwargs)
        set_llm_usage(sp, getattr(resp, "usage", None))
        return resp


def _format_context(
    retrieved: list[tuple[str, float, dict[str, Any]]],
    tool_results: list[dict[str, Any]],
) -> str:
    lines = []
    idx = 1
    for _pid, _score, payload in retrieved:
        src = payload.get("source_id", "?")
        section = payload.get("section") or "-"
        page = payload.get("page")
        loc = f"{src}, p.{page}" if page is not None else src
        text = (payload.get("text") or "").strip()
        lines.append(f"[{idx}] Source: {loc} | Section: {section}\n{text}\n")
        idx += 1
    for tr in tool_results:
        name = tr.get("name", "?")
        args = json.dumps(tr.get("args", {}), separators=(",", ":"))
        result_json = json.dumps(tr.get("result", {}), indent=2, default=str)
        if len(result_json) > 2000:
            result_json = result_json[:2000] + "\n...(truncated)"
        lines.append(f"[{idx}] Tool: {name} | Args: {args}\n{result_json}\n")
        idx += 1
    return "\n".join(lines)


def _extract_citations(
    retrieved: list[tuple[str, float, dict[str, Any]]],
    tool_results: list[dict[str, Any]],
) -> list[Citation]:
    cites: list[Citation] = []
    for _pid, _score, p in retrieved:
        cites.append(Citation(
            source_id=p.get("source_id", "?"),
            chunk_id=p.get("chunk_id", ""),
            section=p.get("section") or "",
            text=(p.get("text") or "")[:400],
            page=p.get("page"),
        ))
    for tr in tool_results:
        cites.append(Citation(
            source_id=f"tool:{tr.get('name', '?')}",
            chunk_id="",
            section=json.dumps(tr.get("args", {}), separators=(",", ":")),
            text=json.dumps(tr.get("result", {}), default=str)[:400],
        ))
    return cites


class RAGAgent(Agent):
    """Hybrid-retrieval RAG agent with MCP tool calling and grounded citations."""

    name = "rag"

    def __init__(self, retrieval_config: RetrievalConfig | None = None) -> None:
        self.retrieval_config = retrieval_config or RetrievalConfig(rerank_top_k=5)
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("rewrite", self._rewrite_query)
        g.add_node("retrieve", self._retrieve)
        g.add_node("call_tools", self._maybe_call_tools)
        g.add_node("synthesize", self._synthesize)
        g.add_edge(START, "rewrite")
        g.add_edge("rewrite", "retrieve")
        g.add_edge("retrieve", "call_tools")
        g.add_edge("call_tools", "synthesize")
        g.add_edge("synthesize", END)
        return g.compile()

    def _rewrite_query(self, state: AgentState) -> dict[str, Any]:
        with span("agent.rewrite", **{"query.length": len(state["query"])}) as sp:
            try:
                resp = _chat_with_retry(
                    model=get_model_name(),
                    messages=[
                        {"role": "user", "content": QUERY_REWRITE_PROMPT.format(query=state["query"])},
                    ],
                    max_tokens=512,
                    temperature=0.0,
                )
                rewritten = (resp.choices[0].message.content or state["query"]).strip()
                rewritten = rewritten.strip("\"'` ").rstrip(":")
            except _RETRY_EXCEPTIONS:
                rewritten = state["query"]
            if len(rewritten.split()) < 3:
                rewritten = state["query"]
            sp.set_attribute("rewritten.length", len(rewritten))
            return {"rewritten_query": rewritten or state["query"]}

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        original = state["query"]
        rewritten = state.get("rewritten_query") or original

        with span("agent.retrieve", **{
            "retrieval.use_bm25": self.retrieval_config.use_bm25,
            "retrieval.use_rerank": self.retrieval_config.use_rerank,
            "retrieval.multi_query": rewritten != original,
        }) as sp:
            cfg_pool = replace(
                self.retrieval_config,
                use_rerank=False,
                rerank_top_k=self.retrieval_config.fusion_k,
            )
            candidates: dict[str, tuple[str, float, dict[str, Any]]] = {}
            queries = [original] + ([rewritten] if rewritten and rewritten != original else [])
            for q in queries:
                for hit in retrieve_docs(q, cfg_pool):
                    pid = hit[0]
                    if pid not in candidates or hit[1] > candidates[pid][1]:
                        candidates[pid] = hit
            pool = list(candidates.values())

            if self.retrieval_config.use_rerank:
                with span("agent.rerank", **{"rerank.pool_size": len(pool)}):
                    final = rerank_fn(original, pool, top_k=self.retrieval_config.rerank_top_k)
            else:
                final = sorted(pool, key=lambda x: -x[1])[: self.retrieval_config.rerank_top_k]

            sp.set_attribute("retrieval.candidates", len(pool))
            sp.set_attribute("retrieval.returned", len(final))
            return {"retrieved": final}

    def _maybe_call_tools(self, state: AgentState) -> dict[str, Any]:
        with span("agent.call_tools") as parent_sp:
            try:
                resp = _chat_with_retry(
                    model=get_model_name(),
                    messages=[
                        {"role": "system", "content": TOOL_DECIDER_SYSTEM_PROMPT},
                        {"role": "user", "content": state["query"]},
                    ],
                    tools=get_tool_schemas(),
                    tool_choice="auto",
                    temperature=0.0,
                )
            except _RETRY_EXCEPTIONS:
                parent_sp.set_attribute("tools.count", 0)
                return {"tool_results": []}

            tool_calls = resp.choices[0].message.tool_calls or []
            parent_sp.set_attribute("tools.count", len(tool_calls))

            results: list[dict[str, Any]] = []
            for call in tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                with span(f"tool.{call.function.name}", **{"tool.args": json.dumps(args)}):
                    result = invoke_tool(call.function.name, args)
                results.append({"name": call.function.name, "args": args, "result": result})
            return {"tool_results": results}

    def _synthesize(self, state: AgentState) -> dict[str, Any]:
        retrieved = state.get("retrieved") or []
        tool_results = state.get("tool_results") or []
        with span("agent.synthesize", **{
            "synth.num_docs": len(retrieved),
            "synth.num_tools": len(tool_results),
        }) as sp:
            if not retrieved and not tool_results:
                sp.set_attribute("synth.fallback", True)
                return {
                    "answer": "I don't have enough information in the provided documents to answer that.",
                    "citations": [],
                }
            context = _format_context(retrieved, tool_results)
            prompt = SYNTHESIS_USER_TEMPLATE.format(context=context, query=state["query"])
            resp = _chat_with_retry(
                model=get_model_name(),
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            answer = (resp.choices[0].message.content or "").strip()
            sp.set_attribute("synth.answer_length", len(answer))
            return {"answer": answer, "citations": _extract_citations(retrieved, tool_results)}

    def run(self, query: str) -> AgentResponse:
        with span("agent.run", **{"agent.name": self.name, "query": query[:200]}):
            result = self.graph.invoke({"query": query})
        return AgentResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            metadata={
                "rewritten_query": result.get("rewritten_query", ""),
                "num_retrieved": len(result.get("retrieved", [])),
                "num_tools_called": len(result.get("tool_results", [])),
                "tools_called": [t.get("name") for t in result.get("tool_results", [])],
            },
        )

    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        yield from self.graph.stream({"query": query})