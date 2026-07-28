"""RAG agent built on LangGraph.

Graph: START -> rewrite_query -> retrieve -> synthesize -> END

- Multi-query retrieval: runs hybrid retrieval on both original + rewritten
  queries, merges candidates, reranks against the ORIGINAL query.
- LLM calls wrapped with exponential-backoff retry (handles Gemini 503 / 429).
"""

from __future__ import annotations

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
)
from src.llm.factory import get_chat_client, get_model_name
from src.retrieval.hybrid import RetrievalConfig, retrieve as retrieve_docs
from src.retrieval.rerank import rerank as rerank_fn


class AgentState(TypedDict, total=False):
    query: str
    rewritten_query: str
    retrieved: list[tuple[str, float, dict[str, Any]]]
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
    """Wrap the OpenAI SDK chat call with exponential-backoff retry."""
    client = get_chat_client()
    return client.chat.completions.create(**kwargs)


def _format_context(retrieved: list[tuple[str, float, dict[str, Any]]]) -> str:
    lines = []
    for i, (_pid, _score, payload) in enumerate(retrieved, 1):
        src = payload.get("source_id", "?")
        section = payload.get("section", "") or "-"
        text = (payload.get("text") or "").strip()
        lines.append(f"[{i}] Source: {src} | Section: {section}\n{text}\n")
    return "\n".join(lines)


def _extract_citations(
    retrieved: list[tuple[str, float, dict[str, Any]]],
) -> list[Citation]:
    return [
        Citation(
            source_id=p.get("source_id", "?"),
            chunk_id=p.get("chunk_id", ""),
            section=p.get("section") or "",
            text=(p.get("text") or "")[:400],
        )
        for _pid, _score, p in retrieved
    ]


class RAGAgent(Agent):
    """Hybrid-retrieval RAG agent with grounded citations."""

    name = "rag"

    def __init__(self, retrieval_config: RetrievalConfig | None = None) -> None:
        self.retrieval_config = retrieval_config or RetrievalConfig(rerank_top_k=5)
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("rewrite", self._rewrite_query)
        g.add_node("retrieve", self._retrieve)
        g.add_node("synthesize", self._synthesize)
        g.add_edge(START, "rewrite")
        g.add_edge("rewrite", "retrieve")
        g.add_edge("retrieve", "synthesize")
        g.add_edge("synthesize", END)
        return g.compile()

    def _rewrite_query(self, state: AgentState) -> dict[str, Any]:
        try:
            resp = _chat_with_retry(
                model=get_model_name(),
                messages=[
                    {"role": "user", "content": QUERY_REWRITE_PROMPT.format(query=state["query"])},
                ],
                max_tokens=100,
                temperature=0.0,
            )
            rewritten = (resp.choices[0].message.content or state["query"]).strip()
            rewritten = rewritten.strip("\"'` ").rstrip(":")
        except _RETRY_EXCEPTIONS:
            # If even retries can't get a rewrite, fall back to the original query.
            rewritten = state["query"]
        # Safety net: overly-short rewrites lose too much intent.
        if len(rewritten.split()) < 3:
            rewritten = state["query"]
        return {"rewritten_query": rewritten or state["query"]}

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        original = state["query"]
        rewritten = state.get("rewritten_query") or original

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
            final = rerank_fn(original, pool, top_k=self.retrieval_config.rerank_top_k)
        else:
            final = sorted(pool, key=lambda x: -x[1])[: self.retrieval_config.rerank_top_k]

        return {"retrieved": final}

    def _synthesize(self, state: AgentState) -> dict[str, Any]:
        retrieved = state.get("retrieved") or []
        if not retrieved:
            return {
                "answer": "I don't have enough information in the provided documents to answer that.",
                "citations": [],
            }
        context = _format_context(retrieved)
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
        return {"answer": answer, "citations": _extract_citations(retrieved)}

    def run(self, query: str) -> AgentResponse:
        result = self.graph.invoke({"query": query})
        return AgentResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            metadata={
                "rewritten_query": result.get("rewritten_query", ""),
                "num_retrieved": len(result.get("retrieved", [])),
            },
        )

    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        yield from self.graph.stream({"query": query})