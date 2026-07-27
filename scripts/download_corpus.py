"""Download the demo cybersecurity corpus to data/corpus/.

Sources: OWASP (Top 10 via HHS mirror, ASVS 5.0.0, Cheat Sheets),
NIST (CSF 2.0, SP 800-63B).
"""

from __future__ import annotations

from pathlib import Path

import httpx

CORPUS_DIR = Path("data/corpus")

SOURCES: list[tuple[str, str]] = [
    # OWASP Top 10 2021 — HHS government mirror (owasp.org no longer hosts a PDF)
    (
        "https://www.hhs.gov/sites/default/files/owasp-top-10.pdf",
        "owasp-top-10-2021.pdf",
    ),
    # OWASP Application Security Verification Standard 5.0.0 (May 2025)
    (
        "https://github.com/OWASP/ASVS/raw/master/5.0/OWASP_Application_Security_Verification_Standard_5.0.0_en.pdf",
        "owasp-asvs-5.0.0.pdf",
    ),
    # NIST Cybersecurity Framework 2.0 (Feb 2024)
    (
        "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
        "nist-csf-2.0.pdf",
    ),
    # NIST SP 800-63B (Digital Identity Guidelines - Authentication)
    (
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf",
        "nist-sp-800-63b.pdf",
    ),
    # OWASP Cheat Sheets (markdown - exercises the .md loader path)
    (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/Authentication_Cheat_Sheet.md",
        "owasp-cs-authentication.md",
    ),
    (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.md",
        "owasp-cs-sql-injection.md",
    ),
    (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.md",
        "owasp-cs-xss.md",
    ),
    (
        "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/Authorization_Cheat_Sheet.md",
        "owasp-cs-authorization.md",
    ),
]


def download_one(url: str, filename: str) -> Path | None:
    out_path = CORPUS_DIR / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        kb = out_path.stat().st_size // 1024
        print(f"  already have {filename} ({kb} KB)")
        return out_path
    print(f"  downloading {filename}...")
    try:
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=60.0,
            headers={"User-Agent": "agentic-rag-platform/0.1 (portfolio project)"},
        ) as r:
            r.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=65536):
                    f.write(chunk)
        kb = out_path.stat().st_size // 1024
        print(f"  saved  {filename} ({kb} KB)")
        return out_path
    except Exception as e:
        print(f"  ! FAILED {filename}: {e}")
        if out_path.exists():
            out_path.unlink()
        return None


def main() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading corpus to {CORPUS_DIR}/")
    ok, failed = 0, 0
    for url, filename in SOURCES:
        if download_one(url, filename):
            ok += 1
        else:
            failed += 1
    print()
    print(f"Done. {ok} ok, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())