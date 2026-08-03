"""Chunking — split documents while preserving section context."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.ingestion.loaders import Document


@dataclass
class Chunk:
    text: str
    chunk_id: str
    source_id: str
    source_path: str
    source_type: str
    chunk_index: int
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_CHARS_PER_CHUNK = 1600
_OVERLAP = 200

_PAGE_SENTINEL = re.compile(r"<!--page:(\d+)-->")


def _extract_page_and_strip(text: str, last_page: int | None) -> tuple[str, int | None]:
    """Find the last page sentinel in this text (carrying forward last_page if
    none is present), then strip all sentinels out of the returned text."""
    page = last_page
    for m in _PAGE_SENTINEL.finditer(text):
        page = int(m.group(1))
    clean = _PAGE_SENTINEL.sub("", text).strip()
    return clean, page


def chunk_document(doc: Document) -> list[Chunk]:
    """Chunk a single document, using markdown-aware splitting for md/pdf."""
    if doc.source_type in ("md", "pdf"):
        return _chunk_markdown(doc)
    return _chunk_flat(doc)


def _chunk_markdown(doc: Document) -> list[Chunk]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"), ("##", "h2"), ("###", "h3"),
            ("####", "h4"), ("#####", "h5"),
        ],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHARS_PER_CHUNK,
        chunk_overlap=_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    try:
        sections = header_splitter.split_text(doc.content)
    except Exception:
        return _chunk_flat(doc)
    chunks: list[Chunk] = []
    idx = 0
    last_page: int | None = None
    for section in sections:
        heading = " > ".join(v for v in section.metadata.values() if v)
        for piece in char_splitter.split_text(section.page_content):
            text, last_page = _extract_page_and_strip(piece, last_page)
            if not text:
                continue
            meta = doc.metadata.copy()
            if last_page is not None:
                meta["page"] = last_page
            chunks.append(
                Chunk(
                    text=text,
                    chunk_id=f"{doc.source_id}#{idx}",
                    source_id=doc.source_id,
                    source_path=doc.source_path,
                    source_type=doc.source_type,
                    chunk_index=idx,
                    section=heading,
                    metadata=meta,
                )
            )
            idx += 1
    return chunks


def _chunk_flat(doc: Document) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHARS_PER_CHUNK,
        chunk_overlap=_OVERLAP,
    )
    chunks: list[Chunk] = []
    last_page: int | None = None
    for i, p in enumerate(splitter.split_text(doc.content)):
        text, last_page = _extract_page_and_strip(p, last_page)
        if not text:
            continue
        meta = doc.metadata.copy()
        if last_page is not None:
            meta["page"] = last_page
        chunks.append(
            Chunk(
                text=text,
                chunk_id=f"{doc.source_id}#{i}",
                source_id=doc.source_id,
                source_path=doc.source_path,
                source_type=doc.source_type,
                chunk_index=i,
                metadata=meta,
            )
        )
    return chunks