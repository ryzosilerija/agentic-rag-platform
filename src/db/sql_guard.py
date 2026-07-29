"""SQL safety validation — layers 2 and 3 of the defense-in-depth."""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from src.db.database import KEV_TABLE

ALLOWED_TABLES = {KEV_TABLE}

_FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "truncate", "attach", "detach", "pragma", "vacuum", "reindex",
    "grant", "revoke",
}


class SQLValidationError(Exception):
    """Raised when a generated SQL statement fails a safety check."""


def validate_sql(sql: str) -> str:
    """Validate and return a safe, single SELECT statement."""
    if not sql or not sql.strip():
        raise SQLValidationError("Empty SQL.")

    lowered = sql.lower()
    for kw in _FORBIDDEN_KEYWORDS:
        if f" {kw} " in f" {lowered} " or lowered.startswith(kw + " "):
            raise SQLValidationError(f"Forbidden keyword detected: {kw!r}")

    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except Exception as e:
        raise SQLValidationError(f"Could not parse SQL: {e}") from e

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SQLValidationError(f"Exactly one statement allowed; got {len(statements)}.")

    stmt = statements[0]

    if not isinstance(stmt, (exp.Select, exp.Union, exp.With)):
        raise SQLValidationError(
            f"Only SELECT queries are allowed; got {type(stmt).__name__}."
        )

    for table in stmt.find_all(exp.Table):
        name = (table.name or "").lower()
        if name and name not in ALLOWED_TABLES:
            raise SQLValidationError(
                f"Table {name!r} is not in the allowlist {sorted(ALLOWED_TABLES)}."
            )

    forbidden_nodes = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Create)
    for node_type in forbidden_nodes:
        if list(stmt.find_all(node_type)):
            raise SQLValidationError(
                f"Statement contains a forbidden {node_type.__name__} node."
            )

    return sql.strip()


def enforce_limit(sql: str, max_rows: int = 100) -> str:
    """Append a LIMIT if the query has none, so results stay bounded."""
    try:
        stmt = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return sql
    if isinstance(stmt, exp.Select) and not stmt.args.get("limit"):
        stmt.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        return stmt.sql(dialect="sqlite")
    return sql