"""M9 verification: SQL agent functional tests + adversarial safety tests.

Usage:
    python -m scripts.test_sql_agent
"""

from __future__ import annotations

from src.agents.sql_agent import SQLAgent
from src.db.sql_guard import SQLValidationError, validate_sql

FUNCTIONAL_QUERIES = [
    "How many vulnerabilities in the catalog are from Microsoft?",
    "Which vendor has the most known exploited vulnerabilities?",
    "How many vulnerabilities are known to be used in ransomware campaigns?",
    "List 3 Apache vulnerabilities in the catalog.",
]

MALICIOUS_SQL = [
    "DROP TABLE kev",
    "SELECT * FROM kev; DROP TABLE kev",
    "DELETE FROM kev WHERE 1=1",
    "UPDATE kev SET vendor='x'",
    "SELECT * FROM sqlite_master",
    "ATTACH DATABASE 'evil.db' AS evil",
    "SELECT * FROM kev UNION SELECT * FROM secrets",
    "INSERT INTO kev VALUES ('x')",
]


def main() -> int:
    print("=" * 70)
    print(" Layer 2+3: adversarial SQL must be REJECTED")
    print("=" * 70)
    all_blocked = True
    for sql in MALICIOUS_SQL:
        try:
            validate_sql(sql)
            print(f"  [LEAK!] NOT blocked: {sql}")
            all_blocked = False
        except SQLValidationError as e:
            print(f"  [blocked] {sql[:50]:<50} -> {str(e)[:40]}")
    print(f"\n  All malicious queries blocked: {all_blocked}\n")

    print("=" * 70)
    print(" Functional: SQL agent answers real questions")
    print("=" * 70)
    agent = SQLAgent()
    for q in FUNCTIONAL_QUERIES:
        print(f"\nQ: {q}")
        resp = agent.run(q)
        print(f"  SQL: {resp.metadata.get('sql', '')[:120]}")
        print(f"  rows: {resp.metadata.get('rows_returned')}  rejected: {resp.metadata.get('rejected')}")
        print(f"  A: {resp.answer[:200]}")

    return 0 if all_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())