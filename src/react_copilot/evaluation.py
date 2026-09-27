
"""Evaluation utilities: baseline, ReAct run, grounded-precision heuristic."""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from .config import (
    OLLAMA_BASE_URL, LLM_MODEL, EVAL_QUESTIONS, RUNS_OUT, CORPUS_DIR,
)
from .agent import run_agent
from .loaders import iter_corpus_files
from .embeddings import get_embeddings

CITATION_RE = re.compile(r"\[([^\[\]:]+)(?::([^\]]+))?\]")


# ----------------------------------------------------------------------
# Baseline: LLM only, no tools
# ----------------------------------------------------------------------
def run_baseline(question: str) -> dict:
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
                     temperature=0.1, num_predict=400)
    sys = ("You are a research assistant. Answer concisely from your own "
           "knowledge. Do NOT claim to cite files. If unsure, say so.")
    resp = llm.invoke([SystemMessage(content=sys),
                       HumanMessage(content=question)])
    return {
        "answer": (resp.content or "").strip(),
        "steps": 0,
        "retrieved_files": [],
        "log": [],
    }


# ----------------------------------------------------------------------
# Citation extraction
# ----------------------------------------------------------------------
def extract_citations(answer: str) -> list[tuple[str, str | None]]:
    """Return list of (file, page_or_locator_or_None)."""
    out = []
    for m in CITATION_RE.finditer(answer or ""):
        f = m.group(1).strip()
        loc = (m.group(2) or "").strip() or None
        if "." in f:
            out.append((f, loc))
    return out


# ----------------------------------------------------------------------
# Semantic ground-truth set
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _load_all_corpus_text() -> dict[str, str]:
    """filename -> full text (PDF pages concatenated; text-ish read as-is)."""
    out: dict[str, str] = {}
    for p in iter_corpus_files(CORPUS_DIR):
        try:
            if p.suffix.lower() == ".pdf":
                from pypdf import PdfReader
                r = PdfReader(str(p))
                out[p.name] = "\n".join(pg.extract_text() or "" for pg in r.pages)
            else:
                out[p.name] = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
    return out


@lru_cache(maxsize=None)
def _valid_sources_for(gold_snippet: str,
                       top_n: int = 6,
                       min_sim: float = 0.50) -> frozenset[str]:
    """Files whose text is semantically close to the gold snippet.

    Used to recognise alternative-but-valid source files, which is necessary
    because the synthetic corpus contains multiple near-duplicate documents
    per topic (e.g. five *_ai_evals.md variants).
    """
    corpus = _load_all_corpus_text()
    if not corpus or not gold_snippet.strip():
        return frozenset()

    names = list(corpus.keys())
    texts = [corpus[n][:3000] for n in names]  # truncate for cost
    emb = get_embeddings()
    try:
        v_snip = np.asarray(emb.embed_query(gold_snippet), dtype=float)
        v_files = np.asarray(emb.embed_documents(texts), dtype=float)
    except Exception as e:
        print(f"[eval] WARN embedding failed for gold snippet: {e}")
        return frozenset()

    # cosine similarity (rows of v_files vs v_snip)
    v_snip = v_snip / (np.linalg.norm(v_snip) + 1e-9)
    v_files = v_files / (np.linalg.norm(v_files, axis=1, keepdims=True) + 1e-9)
    sims = v_files @ v_snip

    idx = np.argsort(-sims)[:top_n]
    return frozenset(names[i] for i in idx if sims[i] >= min_sim)


# ----------------------------------------------------------------------
# Grounded precision
# ----------------------------------------------------------------------
def grounded_precision(answer: str,
                       gold_file: str,
                       gold_snippet: str = "") -> float:
    """Fraction of citations that point to a source supporting the claim.

    A cited file counts as valid if it either
      (a) exactly matches the annotated gold file, OR
      (b) is in the semantic ground-truth set derived from the gold snippet
          (i.e., an alternative-but-valid source, given the duplicate-topic
          nature of the corpus).
    """
    cites = extract_citations(answer)
    if not cites:
        return 0.0

    valid = {gold_file}
    if gold_snippet:
        valid |= set(_valid_sources_for(gold_snippet))

    hits = sum(1 for f, _ in cites if f in valid)
    return hits / len(cites)


# ----------------------------------------------------------------------
# Simple token-overlap accuracy heuristic
# ----------------------------------------------------------------------
_WORD = re.compile(r"[a-zA-Z0-9]+")
_STOP = {"the","a","an","is","are","of","to","in","and","or","for","on",
         "by","with","as","that","this","it","be","can","do","does","how",
         "what","why","when","where","which","who","two","three"}


def _tokens(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s or "")} - _STOP


def accuracy_heuristic(answer: str, gold_snippet: str, thresh: float = 0.30) -> float:
    a, g = _tokens(answer), _tokens(gold_snippet)
    if not g:
        return 0.0
    return 1.0 if len(a & g) / len(g) >= thresh else 0.0


# ----------------------------------------------------------------------
# Run full evaluation
# ----------------------------------------------------------------------
def load_questions(path: Path = EVAL_QUESTIONS) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "question", "gold_source_file", "gold_supporting_snippet"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"evaluation_questions.csv missing columns: {missing}")
    return df


def run_full_eval(out_csv: Path | None = None,
                  modes: Iterable[str] = ("baseline", "react")) -> pd.DataFrame:
    df = load_questions()
    rows: list[dict] = []

    # Pre-warm the semantic ground-truth cache once so we pay the embedding
    # cost up front (n=20 snippets) instead of inside the ReAct loop.
    print("[eval] pre-computing semantic ground-truth sets ...")
    for _, r in df.iterrows():
        _valid_sources_for(str(r["gold_supporting_snippet"]))
    print("[eval] done.")

    for _, r in df.iterrows():
        qid    = r["id"]
        q      = r["question"]
        gold_f = r["gold_source_file"]
        gold_s = r["gold_supporting_snippet"]

        for mode in modes:
            print(f"[eval] {qid} | {mode}")
            if mode == "baseline":
                res = run_baseline(q)
            else:
                res = run_agent(q)

            gp = (grounded_precision(res["answer"], gold_f, gold_s)
                  if mode == "react" else 0.0)
            acc = accuracy_heuristic(res["answer"], gold_s)

            rows.append({
                "question_id":        qid,
                "question":           q,
                "mode":               mode,
                "steps_used":         res["steps"],
                "retrieved_files":    ";".join(res["retrieved_files"]),
                "answer":             res["answer"],
                "citations":          ";".join(
                                        f"{f}:{loc}" if loc else f for f, loc in
                                        extract_citations(res["answer"])),
                "grounded_precision": round(gp, 3),
                "notes":              f"acc={acc}",
            })

    out = pd.DataFrame(rows)
    out_csv = out_csv or (RUNS_OUT / "runs.csv")
    out.to_csv(out_csv, index=False)
    print(f"[eval] wrote {len(out)} rows -> {out_csv}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    for mode in modes:
        sub = out[out["mode"] == mode]
        if sub.empty:
            continue
        print(f"{mode:8s} | n={len(sub)} | "
              f"mean grounded_precision={sub['grounded_precision'].mean():.3f} | "
              f"mean steps={sub['steps_used'].mean():.2f}")

    # Accuracy extracted from notes column
    print("\n=== Accuracy (token-overlap heuristic, thresh=0.30) ===")
    out["_acc"] = out["notes"].str.extract(r"acc=([\d.]+)").astype(float)
    for mode in modes:
        sub = out[out["mode"] == mode]
        if sub.empty:
            continue
        print(f"{mode:8s} | mean accuracy={sub['_acc'].mean():.3f}")

    return out