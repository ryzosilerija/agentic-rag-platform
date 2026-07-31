"""Neo4j client wrapper — cached driver + query helper."""

from __future__ import annotations

from functools import lru_cache

from neo4j import Driver, GraphDatabase

from src.config import settings


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def run_query(cypher: str, **params) -> list[dict]:
    with get_driver().session() as session:
        result = session.run(cypher, **params)
        return [dict(r) for r in result]


def close_driver() -> None:
    try:
        get_driver().close()
        get_driver.cache_clear()
    except Exception:
        pass