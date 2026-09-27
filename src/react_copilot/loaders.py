"""Load corpus files into structured RawDocs with rich metadata."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import json
import csv

from pypdf import PdfReader

from .config import CORPUS_DIR, SUPPORTED_EXTS


@dataclass
class RawDoc:
    text: str
    source: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    row: int | None = None
    extra: dict = field(default_factory=dict)


def _load_text_file(path: Path) -> list[RawDoc]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    text = "\n".join(lines)
    return [RawDoc(
        text=text,
        source=path.name,
        line_start=1,
        line_end=max(1, len(lines)),
    )]


def _load_pdf(path: Path) -> list[RawDoc]:
    docs: list[RawDoc] = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            docs.append(RawDoc(text=text, source=path.name, page=i))
    return docs


def _load_json(path: Path) -> list[RawDoc]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return [RawDoc(text=path.read_text(errors="replace"),
                       source=path.name, row=1, extra={"error": str(e)})]

    docs: list[RawDoc] = []
    rows = None
    if isinstance(data, dict):
        for key in ("documents", "records", "items", "entries", "data"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
    elif isinstance(data, list):
        rows = data

    if rows is not None and all(isinstance(r, dict) for r in rows):
        for i, row in enumerate(rows, start=1):
            text = json.dumps(row, indent=2, ensure_ascii=False)
            docs.append(RawDoc(text=text, source=path.name, row=i))
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)
        docs.append(RawDoc(text=text, source=path.name, row=1))
    return docs


def _load_csv(path: Path) -> list[RawDoc]:
    docs: list[RawDoc] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            text = " | ".join(f"{k}: {v}" for k, v in row.items() if v not in (None, ""))
            docs.append(RawDoc(text=text, source=path.name, row=i))
    return docs


_LOADERS = {
    ".md": _load_text_file,
    ".markdown": _load_text_file,
    ".txt": _load_text_file,
    ".pdf": _load_pdf,
    ".json": _load_json,
    ".csv": _load_csv,
}


def iter_corpus_files(corpus_dir: Path = CORPUS_DIR) -> Iterable[Path]:
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[RawDoc]:
    out: list[RawDoc] = []
    for p in iter_corpus_files(corpus_dir):
        loader = _LOADERS.get(p.suffix.lower())
        if loader is None:
            continue
        try:
            docs = loader(p)
        except Exception as e:
            print(f"[loaders] WARN failed to load {p.name}: {e}")
            continue
        out.extend(docs)
    return out
