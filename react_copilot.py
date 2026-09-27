#!/usr/bin/env python3
"""ReAct Research Copilot - CLI."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from react_copilot.agent import run_agent
from react_copilot.config import CORPUS_DIR
from react_copilot.loaders import load_corpus
from react_copilot.chunking import chunk_raw_docs
from react_copilot.vectorstore import (
    get_pinecone_client, ensure_index, upsert_chunks,
)


def _print_log(log: list[dict]) -> None:
    print("\n---- ReAct trace ----")
    for entry in log:
        t = entry.get("type")
        if t == "reason":
            print(f"[step {entry['step']}] Reason: {entry['content']}")
            for tc in entry.get("tool_calls", []):
                print(f"           Act   : {tc['name']}({tc['args']})")
        elif t == "observe":
            print(f"           Observe ({entry['tool']}): "
                  f"{entry['content'][:300]}...")
        elif t == "nudge":
            print(f"[step {entry['step']}] ⚠️  Nudge: {entry['content']}")
    print("---------------------\n")


def cmd_index(reset: bool = False) -> None:
    print(f"[cli] indexing corpus from {CORPUS_DIR}")
    pc = get_pinecone_client()
    if reset:
        from react_copilot.config import PINECONE_INDEX_NAME
        existing = {i["name"] for i in pc.list_indexes()}
        if PINECONE_INDEX_NAME in existing:
            print(f"[cli] deleting index {PINECONE_INDEX_NAME}")
            pc.delete_index(PINECONE_INDEX_NAME)
    ensure_index(pc)
    chunks = chunk_raw_docs(load_corpus(CORPUS_DIR))
    print(f"[cli] {len(chunks)} chunks produced")
    n = upsert_chunks(chunks)
    print(f"[cli] upserted {n} vectors")


def cmd_ask(question: str, verbose: bool) -> None:
    print(f"\nQuestion: {question}\n")
    res = run_agent(question)
    if verbose:
        _print_log(res["log"])
    print("Final answer:\n")
    print(res["answer"])
    print(f"\n(steps={res['steps']}, files_read={res['retrieved_files']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", "-q", type=str, help="Question to ask")
    ap.add_argument("--index", action="store_true",
                    help="(Re)index the corpus into Pinecone")
    ap.add_argument("--reset", action="store_true",
                    help="With --index: delete and recreate the index first")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print the full ReAct trace")
    args = ap.parse_args()

    if args.index:
        cmd_index(reset=args.reset)
        return
    if not args.question:
        ap.error("provide --question '...' or --index")
    cmd_ask(args.question, args.verbose)


if __name__ == "__main__":
    main()
