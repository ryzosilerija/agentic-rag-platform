"""SQL agent — answers questions by querying the KEV database.

Graph: START -> generate_sql -> validate -> execute -> answer -> END

Defense-in-depth: (1) read-only connection, (2) SQL parsed & restricted to a
single SELECT, (3) schema allowlist. On validation failure the agent never
touches the database. Implements the Agent interface so the supervisor can
route to it.
"""

from __future__ import annotations

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
from src.db.database import get_readonly_connection, get_schema_description
from src.db.sql_guard import SQLValidationError, enforce_limit, validate_sql
from src.llm.factory import get_chat_client, get_model_name
from src.observability.spans import set_llm_usage, span

_RETRY_EXCEPTIONS = (RateLimitError, InternalServerError, APIError)

SQL_GEN_PROMPT = """You are a SQLite expert. Write ONE read-only SELECT query that answers the user's question using ONLY the schema below.

{schema}

Rules:
- Output ONLY the SQL query. No markdown fences, no explanation, no trailing semicolon.
- SELECT statements only. Never write INSERT/UPDATE/DELETE/DROP/etc.
- Reference only the table(s) in the schema.
- Use LIKE '%term%' for partial vendor/product matching (use LOWER() for case-insensitivity).

Question: {question}
SQL:"""

ANSWER_PROMPT = """Answer the user's question using the SQL query results below. Be concise and specific. Cite concrete numbers from the results.

Question: {question}

SQL executed:
{sql}

Results (up to 100 rows):
{results}

Answer:"""


class SQLState(TypedDict, total=False):
    query: str
    sql: str
    validated_sql: str
    rows: list[dict[str, Any]]
    error: str
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


class SQLAgent(Agent):
    """Text-to-SQL agent over the read-only KEV database."""

    name = "sql"

    def __init__(self, max_rows: int = 100) -> None:
        self.max_rows = max_rows
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(SQLState)
        g.add_node("generate_sql", self._generate_sql)
        g.add_node("validate", self._validate)
        g.add_node("execute", self._execute)
        g.add_node("answer", self._answer)
        g.add_edge(START, "generate_sql")
        g.add_edge("generate_sql", "validate")
        g.add_edge("validate", "execute")
        g.add_edge("execute", "answer")
        g.add_edge("answer", END)
        return g.compile()

    def _generate_sql(self, state: SQLState) -> dict[str, Any]:
        with span("sql.generate"):
            resp = _chat(
                model=get_model_name(),
                messages=[
                    {
                        "role": "user",
                        "content": SQL_GEN_PROMPT.format(
                            schema=get_schema_description(), question=state["query"]
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=800,
            )
            sql = (resp.choices[0].message.content or "").strip()
            sql = sql.replace("```sql", "").replace("```", "").strip().rstrip(";")
            return {"sql": sql}

    def _validate(self, state: SQLState) -> dict[str, Any]:
        with span("sql.validate") as sp:
            try:
                safe = validate_sql(state.get("sql", ""))
                safe = enforce_limit(safe, self.max_rows)
                sp.set_attribute("sql.valid", True)
                return {"validated_sql": safe}
            except SQLValidationError as e:
                sp.set_attribute("sql.valid", False)
                sp.set_attribute("sql.rejection", str(e))
                return {"error": f"SQL validation failed: {e}"}

    def _execute(self, state: SQLState) -> dict[str, Any]:
        if state.get("error"):
            return {}
        with span("sql.execute") as sp:
            try:
                conn = get_readonly_connection()
                try:
                    conn.row_factory = _dict_factory
                    cur = conn.execute(state["validated_sql"])
                    rows = cur.fetchall()
                finally:
                    conn.close()
                sp.set_attribute("sql.rows_returned", len(rows))
                return {"rows": rows}
            except Exception as e:
                sp.set_attribute("sql.exec_error", str(e))
                return {"error": f"SQL execution failed: {e}"}

    def _answer(self, state: SQLState) -> dict[str, Any]:
        if state.get("error"):
            return {
                "answer": (
                    "I couldn't safely answer that from the vulnerability database. "
                    f"({state['error']})"
                )
            }
        rows = state.get("rows", [])
        with span("sql.answer", **{"sql.num_rows": len(rows)}):
            results_str = _format_rows(rows)
            resp = _chat(
                model=get_model_name(),
                messages=[
                    {
                        "role": "user",
                        "content": ANSWER_PROMPT.format(
                            question=state["query"],
                            sql=state.get("validated_sql", ""),
                            results=results_str,
                        ),
                    }
                ],
                temperature=0.1,
            )
            return {"answer": (resp.choices[0].message.content or "").strip()}

    def run(self, query: str) -> AgentResponse:
        with span("agent.run", **{"agent.name": self.name, "query": query[:200]}):
            result = self.graph.invoke({"query": query})
        sql = result.get("validated_sql") or result.get("sql", "")
        citations = []
        if sql:
            citations.append(
                Citation(
                    source_id="sql:kev",
                    chunk_id="",
                    section=sql,
                    text=_format_rows(result.get("rows", []))[:400],
                )
            )
        return AgentResponse(
            answer=result.get("answer", ""),
            citations=citations,
            metadata={
                "sql": sql,
                "rows_returned": len(result.get("rows", [])),
                "rejected": bool(result.get("error")),
                "error": result.get("error", ""),
            },
        )

    def stream(self, query: str) -> Iterator[dict[str, Any]]:
        yield from self.graph.stream({"query": query})


def _dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def _format_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    lines = []
    for r in rows[:100]:
        lines.append(" | ".join(f"{k}={v}" for k, v in r.items()))
    return "\n".join(lines)