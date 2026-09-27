"""Chunk RawDocs into smaller Chunks with stable chunk_ids."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .loaders import RawDoc


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int | None
    line_start: int | None
    line_end: int | None
    row: int | None

    def to_metadata(self) -> dict:
        md = {k: v for k, v in asdict(self).items()
              if k not in {"chunk_id", "text"} and v is not None}
        md["chunk_id"] = self.chunk_id
        md["text"] = self.text[:6000]
        return md


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )


def chunk_raw_docs(raw_docs: Iterable[RawDoc]) -> list[Chunk]:
    splitter = _make_splitter()
    chunks: list[Chunk] = []
    for doc in raw_docs:
        pieces = splitter.split_text(doc.text) or [doc.text]
        for i, piece in enumerate(pieces):
            chunk_id = _make_chunk_id(doc.source, doc.page, doc.row, i)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=piece,
                source=doc.source,
                page=doc.page,
                line_start=doc.line_start,
                line_end=doc.line_end,
                row=doc.row,
            ))
    return chunks


def _make_chunk_id(source: str, page: int | None, row: int | None, idx: int) -> str:
    safe = source.replace("/", "_").replace(" ", "_")
    anchor = f"p{page}" if page is not None else (f"r{row}" if row is not None else "t")
    return f"{safe}::{anchor}::{idx}"
