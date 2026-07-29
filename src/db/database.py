"""SQLite database for the SQL agent — loads the CISA KEV catalog."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.mcp_server.tools.kev import _fetch_kev

DB_PATH = Path("data/db/cybersec.db")
KEV_TABLE = "kev"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {KEV_TABLE} (
    cve_id                TEXT PRIMARY KEY,
    vendor                TEXT,
    product               TEXT,
    vulnerability_name    TEXT,
    date_added            TEXT,
    short_description     TEXT,
    required_action       TEXT,
    due_date              TEXT,
    known_ransomware_use  TEXT
);
"""


def build_database() -> int:
    """Fetch the KEV catalog and (re)build the SQLite table. Returns row count."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = _fetch_kev()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(_SCHEMA)
        conn.execute(f"DELETE FROM {KEV_TABLE}")
        conn.executemany(
            f"""INSERT OR REPLACE INTO {KEV_TABLE}
                (cve_id, vendor, product, vulnerability_name, date_added,
                 short_description, required_action, due_date, known_ransomware_use)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    r.get("cveID", ""),
                    r.get("vendorProject", ""),
                    r.get("product", ""),
                    r.get("vulnerabilityName", ""),
                    r.get("dateAdded", ""),
                    r.get("shortDescription", ""),
                    r.get("requiredAction", ""),
                    r.get("dueDate", ""),
                    r.get("knownRansomwareCampaignUse", ""),
                )
                for r in entries
            ],
        )
        conn.commit()
        count = conn.execute(f"SELECT COUNT(*) FROM {KEV_TABLE}").fetchone()[0]
    finally:
        conn.close()
    return count


def get_readonly_connection() -> sqlite3.Connection:
    """Open the database strictly read-only (SQLite mode=ro). Safety layer 1."""
    if not DB_PATH.exists():
        raise RuntimeError(f"Database not found at {DB_PATH}. Run scripts.build_db first.")
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def get_schema_description() -> str:
    """Human-readable schema shown to the LLM so it can write correct SQL."""
    return f"""Table: {KEV_TABLE}  (CISA Known Exploited Vulnerabilities catalog)
Columns:
  cve_id               TEXT  -- e.g. 'CVE-2021-44228'
  vendor               TEXT  -- e.g. 'Microsoft', 'Apache'
  product              TEXT
  vulnerability_name   TEXT
  date_added           TEXT  -- ISO 'YYYY-MM-DD', when CISA added it
  short_description    TEXT
  required_action      TEXT
  due_date             TEXT  -- ISO 'YYYY-MM-DD'
  known_ransomware_use TEXT  -- 'Known' or 'Unknown'

Notes:
  - Dates are ISO strings; compare with strftime() or string comparison.
  - Use LIKE for partial vendor/product matches.
  - This is the ONLY table available."""