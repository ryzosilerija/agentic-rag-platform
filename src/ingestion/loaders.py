"""Document loaders — dispatch by file extension, return Document objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """A raw loaded document with metadata (pre-chunking)."""

    content: str
    source_id: str
    source_path: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_pdf(path: Path) -> Document:
    """Extract PDF to markdown using pymupdf4llm, preserving per-page boundaries.

    Emits invisible <!--page:N--> sentinels at each page boundary so the
    chunker can recover the source page for citations. Sentinels are HTML
    comments (invisible in rendered markdown) and are stripped from chunk
    text before embedding.
    """
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    parts = []
    for p in pages:
        pageno = p["metadata"]["page_number"]
        parts.append(f"<!--page:{pageno}-->\n{p['text']}")
    md_text = "\n\n".join(parts)
    return Document(
        content=md_text,
        source_id=path.stem,
        source_path=str(path),
        source_type="pdf",
        metadata={"char_len": len(md_text)},
    )


def load_markdown(path: Path) -> Document:
    return Document(
        content=path.read_text(encoding="utf-8"),
        source_id=path.stem,
        source_path=str(path),
        source_type="md",
    )


def load_html(path: Path) -> Document:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return Document(
        content="\n\n".join(lines),
        source_id=path.stem,
        source_path=str(path),
        source_type="html",
        metadata={"title": (soup.title.string.strip() if soup.title and soup.title.string else "")},
    )


def load_text(path: Path) -> Document:
    return Document(
        content=path.read_text(encoding="utf-8", errors="ignore"),
        source_id=path.stem,
        source_path=str(path),
        source_type="txt",
    )


_LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".txt": load_text,
}


def load_file(path: Path) -> Document:
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return _LOADERS[ext](path)


def load_folder(folder: Path) -> list[Document]:
    """Recursively load all supported files under folder."""
    docs: list[Document] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in _LOADERS:
            try:
                docs.append(load_file(path))
                print(f"  loaded {path.name}")
            except Exception as e:
                print(f"  ! failed to load {path.name}: {e}")
    return docs