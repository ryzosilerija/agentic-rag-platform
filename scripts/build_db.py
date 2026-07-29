"""Build the SQL agent's database from the CISA KEV catalog.

Usage:
    python -m scripts.build_db
"""

from __future__ import annotations

from src.db.database import DB_PATH, build_database


def main() -> int:
    print("Building cybersec database from CISA KEV catalog...")
    count = build_database()
    print(f"  wrote {count} rows to {DB_PATH}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())