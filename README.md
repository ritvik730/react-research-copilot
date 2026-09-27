# ReAct Research Copilot

A single-agent **ReAct (Reason + Act)** research assistant that answers questions over a local document corpus with grounded, inline citations. Built with **LangChain**, **LangGraph**, **Pinecone**, and **Ollama** running entirely on a local machine.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Key Results](#key-results)
3. [Architecture](#architecture)
4. [The ReAct Loop](#the-react-loop)
5. [Tools](#tools)
6. [Retrieval Pipeline](#retrieval-pipeline)
7. [Project Structure](#project-structure)
8. [Prerequisites](#prerequisites)
9. [Installation](#installation)
10. [Configuration](#configuration)
11. [Running the Project](#running-the-project)
12. [Evaluation Methodology](#evaluation-methodology)
13. [Evaluation Results](#evaluation-results)
14. [Failure Analysis](#failure-analysis)
15. [Safety and Robustness](#safety-and-robustness)
16. [Citation Format](#citation-format)
17. [Troubleshooting](#troubleshooting)
18. [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
19. [Limitations and Future Work](#limitations-and-future-work)
20. [Reproducibility](#reproducibility)

---

## What This Project Does

Large language models **hallucinate** when asked about content they weren't trained on — especially private documents. This project solves that with a **ReAct agent** that:

1. **Reasons** about what evidence it needs
2. **Acts** by calling retrieval tools (`search`, `open_file`, `read_chunk`)
3. **Observes** the results
4. **Repeats** until it has enough evidence
5. Produces a concise final answer with **inline citations** pointing to the exact source

The agent never claims to know something it didn't retrieve. If the evidence is insufficient, it says so and asks a clarifying question.

**Example interaction:**

```
$ uv run python react_copilot.py -q "What does faithfulness mean in LLM evaluation?" -v

---- ReAct trace ----
[step 1] Reason:
           Act   : search({'query': 'faithfulness in LLM evaluation', 'k': 5})
           Observe (search): {"results":[{"chunk_id":"01_ai_evals.md::t::0", ...}]}
[step 2] Reason: I should read the top result to get the full text.
           Act   : read_chunk({'chunk_id': '01_ai_evals.md::t::0'})
           Observe (read_chunk): {"text":"... Evaluation should separate retrieval quality ..."}
---------------------

Final answer:

Faithfulness means the answer is supported by retrieved evidence
[01_ai_evals.md]. According to the document, it refers to the degree to
which the generated text is grounded in reality and accurately reflects
the input or context.
```

---

## Key Results

Evaluated on **20 questions** with a **fixed gold source + gold snippet** per question, comparing an LLM-only baseline against the ReAct + tools system:

| Mode                | Grounded Precision | Accuracy | Mean Steps |
|---------------------|-------------------:|---------:|-----------:|
| LLM-only (baseline) | 0.000              | 0.500    | 0.00       |
| **ReAct + tools**   | **0.750**          | **0.900**| **3.65**   |

- **+0.75 grounded precision** — ReAct answers cite sources that support their claims.
- **+0.40 accuracy** — moving from "the model guesses" to "the model reads and answers."
- **3.65 tool calls per question** — well under the 8-step safety guard.

Full evaluation data: [`runs/runs.csv`](runs/runs.csv).

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │      User Question          │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   System Prompt + Question  │
                    └──────────────┬──────────────┘
                                   ▼
       ┌───────────────────────────────────────────────────┐
       │              LangGraph State Machine              │
       │                                                   │
       │   ┌─────────┐   tool_calls?   ┌─────────────┐     │
       │   │ Reason  │───────yes──────▶│  ToolNode   │     │
       │   │ (LLM)   │                 │  (executes) │     │
       │   └────┬────┘◀────────────────┴──────┬──────┘     │
       │        │        observations         │            │
       │        │ no tool calls               │            │
       │        ▼                             │            │
       │   ┌─────────┐                        │            │
       │   │ Nudge   │ (if read_chunk not yet │            │
       │   │ Guard   │  called → loop back)   │            │
       │   └────┬────┘                        │            │
       │        ▼                             │            │
       │   ┌─────────┐                        │            │
       │   │  END    │                        │            │
       │   └─────────┘                        │            │
       └───────────────────────────────────────┼───────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────┐
                    │          Tools (allow-list)          │
                    │  search(query,k) / open_file(file) / │
                    │  read_chunk(chunk_id)                │
                    └──────────────────┬───────────────────┘
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         Pinecone Vector Index        │
                    │     (768d, cosine, serverless)       │
                    └──────────────────┬───────────────────┘
                                       ▲
                                       │  upsert (batch 50)
                    ┌──────────────────┴───────────────────┐
                    │        Ingestion Pipeline            │
                    │  loaders → chunking → embeddings     │
                    │       (nomic-embed-text via Ollama)  │
                    └──────────────────┬───────────────────┘
                                       ▲
                    ┌──────────────────┴───────────────────┐
                    │      Local Corpus (data/corpus/)     │
                    │   .md  .pdf  .json  .csv  (+manifest)│
                    └──────────────────────────────────────┘
```

### Component Overview

| Layer | Technology | File |
|---|---|---|
| LLM | `llama3.2:3b` via Ollama | `src/react_copilot/agent.py` |
| Embeddings | `nomic-embed-text` via Ollama | `src/react_copilot/embeddings.py` |
| Vector store | Pinecone serverless (AWS us-east-1) | `src/react_copilot/vectorstore.py` |
| Agent framework | LangGraph `StateGraph` | `src/react_copilot/agent.py` |
| Tools | LangChain `@tool` | `src/react_copilot/tools.py` |
| Retrieval | LangChain `PineconeVectorStore` | `src/react_copilot/vectorstore.py` |
| Chunking | `RecursiveCharacterTextSplitter` | `src/react_copilot/chunking.py` |
| Loaders | `pypdf` + custom JSON/CSV readers | `src/react_copilot/loaders.py` |

---

## The ReAct Loop

The agent follows the **Reason → Act → Observe** cycle:

1. **Reason** — the LLM sees the conversation history and decides whether to call a tool or produce a final answer. It emits either `tool_calls` (structured) or plain text.
2. **Act** — `ToolNode` executes the requested tool(s) inside a safe wrapper. Every tool returns a JSON-serialized observation.
3. **Observe** — the observation is appended to the message history and the loop returns to **Reason**.

The loop is a `LangGraph.StateGraph` with three nodes:

```
Reason ──(tool_calls)──▶ Tools ──▶ Reason
   │
   └──(no tool_calls)──▶ Nudge (if read_chunk not yet called) ──▶ Reason
   │
   └──(no tool_calls & read_chunk called)──▶ END
```

### Guards inside the loop

- **`MAX_STEPS = 8`** — hard cap. The loop exits even if the model wants to keep going.
- **Nudge guard** — if the model tries to answer without ever calling `read_chunk` (i.e., it's about to answer from a search *preview* rather than actual content), the agent injects a directive reminder and loops back. This prevents the "hallucinated quote from preview" failure mode.
- **Recursion limit** — `MAX_STEPS * 2 + 6` bounds the LangGraph execution.
- **Tool allow-list** — only three tools are registered. Tools themselves reject paths outside the corpus folder.

---

## Tools

Every tool returns a JSON string with a stable schema. This makes them safe for the LLM to parse and easy to unit-test.

### `search(query: str, k: int = 5) -> JSON`

Semantic search over the Pinecone index.

```json
{
  "results": [
    {
      "chunk_id": "01_ai_evals.md::t::0",
      "file": "01_ai_evals.md",
      "page": null,
      "row": null,
      "lines": "1-8",
      "score": 0.6993,
      "preview": "# Practical LLM Evaluation Basics ..."
    }
  ],
  "elapsed_s": 0.32
}
```

- `k` is clamped to `[1, 10]`.
- Previews are **truncated to 280 chars** — the model is explicitly instructed that previews are *not evidence*.

### `open_file(file: str) -> JSON`

Lists all chunks belonging to a single file (metadata filter `source == file`).

```json
{
  "file": "guide_chunking.pdf",
  "chunks": [
    {"chunk_id": "guide_chunking.pdf::p1::0", "page": 1, "lines": null, ...}
  ]
}
```

- Rejects any filename containing `/`, `\`, or `..`.
- Only filenames present in the corpus directory allow-list are accepted.

### `read_chunk(chunk_id: str) -> JSON`

Fetches the **full text** of a chunk by its Pinecone ID.

```json
{
  "chunk_id": "01_ai_evals.md::t::0",
  "file": "01_ai_evals.md",
  "page": null,
  "row": null,
  "lines": "1-8",
  "text": "# Practical LLM Evaluation Basics ... (full chunk text)"
}
```

- This is the **only** tool that returns full text. The prompt requires the model to call it before citing anything.

---

## Retrieval Pipeline

### Ingestion

1. **Load** — `loaders.py` walks `data/corpus/` recursively and dispatches per file type:
   - `.md`, `.markdown`, `.txt` → read as-is, tag `line_start` / `line_end`.
   - `.pdf` → `pypdf.PdfReader`, one `RawDoc` per page with `page=i`.
   - `.json` → pretty-printed as text; if a top-level list is present (e.g. `documents`, `records`, `items`), each element becomes its own row with `row=i`.
   - `.csv` → each row becomes `"col1: val1 | col2: val2 | ..."` with `row=i`.

2. **Chunk** — `RecursiveCharacterTextSplitter` with:
   - `chunk_size = 800` characters
   - `chunk_overlap = 120` characters
   - separators: `["\n\n", "\n", ". ", "? ", "! ", " ", ""]`

3. **Embed** — each chunk is embedded with `nomic-embed-text` (768 dimensions) via local Ollama.

4. **Upsert** — batches of 50 are added to the Pinecone index with these metadata keys:

   | Key | Type | Present for |
   |---|---|---|
   | `chunk_id` | string | all |
   | `source` | string (filename) | all |
   | `text` | string (truncated to 6000 chars) | all |
   | `page` | int | PDFs |
   | `row` | int | JSON/CSV |
   | `line_start` | int | text-ish files |
   | `line_end` | int | text-ish files |

### Query

Given a natural-language query:

1. Embed the query with the same model.
2. Top-`k` nearest chunks returned by Pinecone (cosine similarity).
3. Return file + page/row/line locators + short preview.

The agent uses these locators to build its final citation.

---

## Project Structure

```
react-research-copilot/
├── data/                                  # Local corpus + eval files (yours)
│   ├── corpus/
│   │   ├── *.md  *.pdf  *.json  *.csv
│   ├── corpus_manifest.json
│   ├── evaluation_questions.csv
│   └── runs_template.csv
├── src/
│   └── react_copilot/
│       ├── __init__.py
│       ├── config.py                      # Env + constants
│       ├── loaders.py                     # File-type → RawDoc
│       ├── chunking.py                    # RawDoc → Chunk
│       ├── embeddings.py                  # OllamaEmbeddings wrapper
│       ├── vectorstore.py                 # Pinecone index + upsert + fetch
│       ├── tools.py                       # search / open_file / read_chunk
│       ├── prompts.py                     # System + user prompts
│       ├── agent.py                       # LangGraph ReAct loop
│       └── evaluation.py                  # Baseline vs ReAct harness
├── scripts/
│   ├── __init__.py
│   ├── index_corpus.py                    # One-shot CLI indexing
│   └── run_evaluation.py                  # One-shot CLI eval
├── notebooks/
│   └── demo.ipynb                         # Walkthrough notebook
├── presentation/
│   └── slides.md                          # Marp deck (7 slides)
├── runs/                                  # Output logs (gitignored)
│   └── runs.csv
├── react_copilot.py                       # Main CLI entry point
├── pyproject.toml                         # uv-managed dependencies
├── uv.lock                                # Locked dependency versions
├── .env.example                           # Template for secrets
├── .gitignore
└── README.md                              # (this file)
```

---

## Prerequisites

- **macOS, Linux, or WSL2** (tested on macOS)
- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — the fast Python package manager
  ```bash
  brew install uv
  # or
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **[Ollama](https://ollama.com/download)** running locally on `http://localhost:11434`
- **[Pinecone](https://www.pinecone.io/)** account (free tier is enough — serverless)

### Models to pull

```bash
ollama pull llama3.2:3b        # ~2.0 GB, the reasoning LLM
ollama pull nomic-embed-text   # ~275 MB, the embedding model
```

Verify:

```bash
ollama list
# NAME                       ID              SIZE
# llama3.2:3b                a80c4f17acd5    2.0 GB
# nomic-embed-text:latest    0a109f422b47    274 MB
```

---

## Installation

```bash
git clone <your-repo-url> react-research-copilot
cd react-research-copilot

# Create virtualenv + install all dependencies
uv sync

# (Optional) also install dev tools like Jupyter
uv sync --extra dev
```

`uv sync` creates `.venv/` and installs every dependency pinned in `uv.lock`. No need to activate manually — `uv run` handles that.

---

## Configuration

Copy the example env file and fill in your Pinecone key:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Pinecone
PINECONE_API_KEY=pcsk_xxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_INDEX_NAME=react-copilot
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
EMBED_MODEL=nomic-embed-text

# Absolute path to this project root
PROJECT_ROOT=/Users/you/path/to/react-research-copilot
```

Verify the key is loaded:

```bash
uv run python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('key set:', bool(os.getenv('PINECONE_API_KEY')))"
# key set: True
```

### Environment Variables Reference

| Variable | Purpose | Default |
|---|---|---|
| `PINECONE_API_KEY` | Pinecone API key (required) | — |
| `PINECONE_INDEX_NAME` | Index name | `react-copilot` |
| `PINECONE_CLOUD` | Pinecone cloud | `aws` |
| `PINECONE_REGION` | Pinecone region | `us-east-1` |
| `OLLAMA_BASE_URL` | Ollama endpoint | `http://localhost:11434` |
| `LLM_MODEL` | LLM tag | `llama3.2:3b` |
| `EMBED_MODEL` | Embedding tag | `nomic-embed-text` |
| `PROJECT_ROOT` | Absolute project path | cwd |

---

## Running the Project

### 1. Index the corpus

Builds the Pinecone index from `data/corpus/`. Safe to rerun; use `--reset` to rebuild from scratch.

```bash
uv run python react_copilot.py --index --reset
```

Expected output:

```
[cli] indexing corpus from /Users/.../data/corpus
[vectorstore] creating index 'react-copilot' (768d, cosine, aws/us-east-1)
[cli] 54 chunks produced
[vectorstore] upserted 50/54
[vectorstore] upserted 54/54
[cli] upserted 54 vectors
```

- **First run takes 1–3 minutes** — each embedding is a local HTTP call.
- **Subsequent runs are faster** — the index is reused unless `--reset` is passed.

### 2. Ask a question

```bash
uv run python react_copilot.py -q "What does faithfulness mean in LLM evaluation?"
```

With the full ReAct trace (recommended for demos):

```bash
uv run python react_copilot.py -q "What does faithfulness mean in LLM evaluation?" -v
```

### 3. Run the full evaluation

Compares LLM-only vs ReAct + tools on all 20 questions from `data/evaluation_questions.csv`.

```bash
uv run python -m scripts.run_evaluation
```

- Runs **40 LLM queries** (20 questions × 2 modes)
- Writes `runs/runs.csv` with one row per (question, mode)
- Prints a summary at the end
- **Takes 15–30 minutes** on `llama3.2:3b`

### 4. Open the notebook

```bash
uv run jupyter notebook notebooks/demo.ipynb
```

### CLI Reference

```bash
uv run python react_copilot.py --help

# Options:
#   -q, --question TEXT   Question to ask
#   --index               (Re)index the corpus
#   --reset               With --index: delete and recreate
#   -v, --verbose         Print the full ReAct trace
```

---

## Evaluation Methodology

Every question in `data/evaluation_questions.csv` has:

| Column | Meaning |
|---|---|
| `id` | Question ID (Q01–Q20) |
| `question` | The natural-language question |
| `gold_source_file` | The primary file that contains the answer |
| `gold_supporting_snippet` | A short passage that supports the answer |

### Two modes

- **Baseline** — the LLM is asked the question directly with no tools and no corpus access. It's told not to claim to cite anything.
- **ReAct + tools** — the full agent runs as in production.

### Two metrics

#### Grounded precision

Fraction of citations in the final answer that point to a source supporting the claim.

```
grounded_precision = (# of valid citations) / (total citations)
```

A citation is **valid** if the cited file is either:

1. Exactly the annotated `gold_source_file`, **or**
2. One of the top-6 corpus files whose content is semantically closest to the `gold_supporting_snippet` (cosine ≥ 0.5), computed once per question with `nomic-embed-text`.

The second clause handles the fact that our synthetic corpus has **near-duplicate topics** — e.g., five different `*_ai_evals.md` files cover the same material. A correct, well-cited answer that points to a sibling file shouldn't be scored as a failure.

If the answer contains zero citations → `grounded_precision = 0`.

#### Accuracy (token-overlap heuristic)

```
accuracy = 1  if  |tokens(answer) ∩ tokens(gold_snippet)| / |tokens(gold_snippet)| ≥ 0.30
           0  otherwise
```

Tokens are lowercased, deduplicated, and stopword-filtered. This is a *proxy* metric — cheap, deterministic, and honest enough for a 20-question set. It intentionally doesn't try to be a semantic judge.

### Why not use an LLM as the judge?

An LLM judge would introduce another layer of nondeterminism and another point of failure. The whole point of this project is grounded evaluation with deterministic signal. Both metrics above are reproducible byte-for-byte from the CSV.

---

## Evaluation Results

**Setup:** 20 questions, 2 modes, `llama3.2:3b` + `nomic-embed-text`, Pinecone serverless (AWS us-east-1), 54 chunks from 30+ documents.

| Mode                | Grounded Precision | Accuracy | Mean Steps |
|---------------------|-------------------:|---------:|-----------:|
| LLM-only (baseline) | 0.000              | 0.500    | 0.00       |
| **ReAct + tools**   | **0.750**          | **0.900**| **3.65**   |

### Interpretation

- **Grounded precision** jumped from **0.000** (baseline can't cite) to **0.750** — three quarters of ReAct's citations point to sources that actually support the claim. The remaining quarter are documented in the failure analysis below.
- **Accuracy** improved from **0.500** to **0.900**. The baseline occasionally gave plausible-sounding but incorrect answers (e.g., confusing "RAG" with "Risk Assessment and Governance"), whereas ReAct grounded the answer in the retrieved text.
- **Mean steps = 3.65** — ReAct uses a modest number of tool calls, well under the 8-step safety guard. This is what you'd expect for a well-scoped retrieval agent on short factual questions.

The full per-question log lives at `runs/runs.csv`.

---

## Failure Analysis

Of the 20 ReAct runs, **5 fell short**. They split into two clean categories.

### Category 1 — Wrong sibling file cited (4 cases)

The model answered correctly but cited a topically-adjacent file instead of the gold file:

| Q | Question Topic | Gold File | Cited | Why |
|---|---|---|---|---|
| Q13 | RAG pipeline stages | `01_rag.md` | `01_ai_evals.md` | 5 near-duplicate "AI evals" files dominate the top-k for many topics |
| Q16 | Least privilege | `01_webapp_security.md` | `01_ai_evals.md` | Same reason — retrieval cluster is noisy |
| Q19 | Prompt-injection warning | `01_prompt_injection.md` | `01_rag.md` | Retrieval missed the prompt-injection cluster on this phrasing |
| Q20 | Insufficient evidence handling | `guide_evidence_citations.pdf` | `08_finance_variance.md` | The PDF is short and its embedding is weak |

**Mitigation attempted:** The semantic ground-truth set (see *Evaluation Methodology*) already forgives most sibling-file errors — a citation to `04_ai_evals.md` counts as valid if its content is close to the `01_ai_evals.md` gold snippet. These four cases failed *even with that tolerance*, meaning the cited file's topic diverged enough to fall outside the top-6.

### Category 2 — Retrieval miss (1 case)

**Q05** — "What is the purpose of citations in research assistants?"

The agent's `search` returned a "## Notes for Practitioners" footer chunk from `04_ai_evals.md` instead of the substantive content from `guide_evidence_citations.pdf`. The 3B model accepted this low-quality chunk and answered without a citation.

**Mitigation attempted:** The **nudge guard** forces at least one `read_chunk` call before the model is allowed to answer. It worked — the model *did* read a chunk — but the chunk was the wrong one. This is a retrieval-quality problem, not a loop-control problem.

### Fixes implemented

| Fix | Where | Status |
|---|---|---|
| Nudge guard forcing `read_chunk` | `agent.py::_nudge_node` | ✅ Eliminates preview-only answers |
| Citation normalizer (strips URLs, `::`, parens) | `agent.py::_normalize_citations` | ✅ Malformed citations gone from final answers |
| Semantic ground-truth scoring | `evaluation.py::_valid_sources_for` | ✅ Neutralizes duplicate-topic bias |
| Max-step guard (`MAX_STEPS = 8`) | `agent.py::_should_continue` | ✅ No infinite loops observed |
| Tool allow-list (filename check) | `tools.py::_safe_file` | ✅ No path traversal possible |

### What would close the remaining gap

- **Cross-encoder reranking** — a second-stage reranker (`bge-reranker`, `cohere-rerank`) would push the correct file into the top-1 for the 4 sibling-citation failures.
- **Larger local LLM** — `llama3.1:8b` or `qwen2.5:7b` would improve instruction adherence on the citation format.
- **Chunk-level reranking with hybrid search** — combining BM25 with dense embeddings typically recovers the short-PDF failures like Q20.

These are out of scope for this project (which targets a 3B model to demonstrate local-only feasibility) but are worth pursuing in production.

---

## Safety and Robustness

| Concern | Mechanism |
|---|---|
| **Runaway loops** | `MAX_STEPS = 8` hard cap in `_should_continue`; recursion limit on `agent.invoke` |
| **Path traversal** | `_safe_file()` rejects filenames containing `/`, `\`, `..`; only allow-listed corpus files are reachable |
| **Tool misuse** | Only 3 tools registered in `ALL_TOOLS`. No shell, no network, no filesystem access outside the corpus |
| **Preview over-quoting** | Nudge guard forces `read_chunk` before the model can answer |
| **Hallucinated citations** | Prompt forbids URLs; `_normalize_citations` strips any markdown links, `::` residue, and parenthesized chunk IDs from the final answer |
| **Embedding failures** | Wrapped in `try/except`; tool returns `{"error": ...}` and the agent can decide to retry or bail |
| **Missing secrets** | `get_pinecone_client()` raises early with a clear message if `PINECONE_API_KEY` is unset |
| **Retry policy** | Tool calls are single-shot; the ReAct loop itself is the retry mechanism — if a search returns junk, the model can search again |
| **Timeouts** | Pinecone and Ollama clients use their library defaults; the 8-step cap bounds total wall-clock time |

---

## Citation Format

The agent emits inline citations in square brackets:

| Source type | Format | Example |
|---|---|---|
| PDF | `[file.pdf:page]` | `[policy_eval_playbook.pdf:1]` |
| Markdown / text | `[file.md:line_start-line_end]` or `[file.md]` | `[01_ai_evals.md:1-8]` |
| JSON | `[file.json:row N]` | `[corpus_manifest.json:row 3]` |
| CSV | `[file.csv:row N]` | `[evaluation_questions.csv:row 5]` |

### What's forbidden

- ❌ Markdown links with URLs — `[01_ai_evals.md:1-8](https://example.com/...)`
- ❌ Raw chunk IDs — `[01_ai_evals.md::t::0]`
- ❌ Parenthesized citations — `(01_ai_evals.md::t::0)`

`agent.py::_normalize_citations` runs on every final answer as a safety net, rewriting any of the above to the canonical `[file]` or `[file:locator]` form.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PINECONE_API_KEY is missing` | `.env` not loaded or has typos | Recheck `.env`; run the verify command in [Configuration](#configuration) |
| `Connection refused` on `localhost:11434` | Ollama not running | Run `ollama serve` in another terminal, or open the Ollama desktop app |
| `model 'nomic-embed-text' not found` | Embedding model not pulled | `ollama pull nomic-embed-text` |
| `model 'llama3.2:3b' not found` | LLM not pulled | `ollama pull llama3.2:3b` |
| `dimension mismatch` on upsert | Old index from a different embed model | Delete the index in the Pinecone dashboard, rerun `--index --reset` |
| Indexing is slow the first time | One HTTP round-trip per chunk to Ollama | Expected; takes 1–3 min for a small corpus |
| ReAct answers without citing | Model ignored the rule (3B limitation) | Rerun — llama3.2:3b is stochastic. If consistent, lower `temperature` further or switch to `llama3.2:latest` |
| `Nudge: forced read_chunk` fires often | Model is impatient | This is expected behavior. To reduce frequency, add a stricter line to `prompts.py::SYSTEM_PROMPT` |
| Evaluation takes forever | 40 sequential LLM calls on a 3B model | Expected: 15–30 min. Don't kill it mid-run |
| `uv sync` fails on `package = false` | `pyproject.toml` missing `[tool.uv] package = false` | Add the line; it tells uv not to try to build the project as a wheel |

---

## Design Decisions and Trade-offs

### Why a text-based ReAct loop instead of native function calling?

`llama3.2:3b` **does support** Ollama's tool-calling interface, but on a 3B model its adherence is inconsistent. Some calls come back as prose ("I will now call `read_chunk`...") instead of structured tool calls. The LangGraph `ToolNode` + nudge guard handles both cases uniformly:

- Structured calls flow straight to the tool node.
- Prose calls trigger a nudge → the model retries with a real tool call.

This is more robust than relying on native function calling alone.

### Why Pinecone instead of a local vector DB like FAISS?

Pinecone serverless is free for small indices, requires no local infra, and gives us metadata filtering (used by `open_file`'s `filter={"source": {"$eq": file}}`). A local FAISS index would remove the network dependency but loses the metadata filter and adds plumbing. For a prototype targeting reproducibility across machines, Pinecone is the simpler choice. Swapping in FAISS would be a ~30-line change in `vectorstore.py`.

### Why `nomic-embed-text` instead of OpenAI embeddings?

Everything runs locally. No API keys, no per-query cost, no network. `nomic-embed-text` produces 768-dimensional vectors, which is more than enough for a corpus of this size, and it runs in ~50 ms per document on an M-series Mac.

### Why 800-character chunks with 120 overlap?

Roughly 200 tokens. Short enough that search returns focused results, long enough to contain a complete thought. The overlap prevents facts from being cut in half at chunk boundaries. This is well within the "600–1,000 tokens for narrative text" range cited in the corpus itself.

### Why the semantic ground-truth set for evaluation?

The corpus is synthetic and contains **many near-duplicate files** (e.g., five `*_ai_evals.md` variants, eight `*_prompt_injection.md` variants). A strict "gold file must match exactly" metric would penalize *correct* answers that happen to pick a sibling file. The semantic ground-truth set — computed once per question via `nomic-embed-text` — captures the *set of valid sources*, not just the single annotated one. This is closer to what "grounded" should mean in a real evaluation.

### Why the nudge guard instead of a stricter prompt?

We tried a stricter prompt first. The 3B model still answered directly in ~30% of runs because it saw a promising preview and jumped to conclusions. The nudge guard is a *structural* fix — the loop physically cannot end until `read_chunk` has been called at least once. This is the pattern recommended for reliability with smaller models: enforce in code what you can't enforce in prose.

---

## Limitations and Future Work

### Known limitations

1. **Only 3B model.** `llama3.2:3b` is a demonstration-size model. Larger models would show meaningfully higher grounded precision.
2. **No reranker.** Top-k dense retrieval is used raw. A cross-encoder reranker would improve precision on ambiguous queries.
3. **No hybrid search.** Only dense embeddings are used; BM25 would help on keyword-heavy questions.
4. **No streaming.** The CLI waits for the full answer before printing. Streaming would improve UX.
5. **Corpus-scoped tools only.** The agent cannot read files outside `data/corpus/`, by design.
6. **Single-turn.** The agent doesn't support follow-up questions with memory across invocations.

### Natural extensions

- **Multi-turn conversation** — persist `AgentState.messages` across CLI invocations.
- **Streaming output** — `agent.stream()` for token-by-token printing.
- **Reranking** — add a `rerank(chunk_ids)` tool backed by a cross-encoder.
- **Hybrid search** — combine Pinecone dense retrieval with a BM25 index (e.g., `rank_bm25`).
- **Query rewriting** — a small pre-step that paraphrases the user question into 2–3 sub-queries.
- **Larger model fallback** — route hard queries to `llama3.1:8b` when the 3B model fails to converge.

---

## Reproducibility

Everything in this repo is deterministic except for LLM sampling (`temperature=0.1`). To reproduce the published evaluation:

```bash
# 1. Set up
uv sync
cp .env.example .env      # fill in PINECONE_API_KEY

# 2. Pull models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 3. Reset the index and re-index the corpus
uv run python react_copilot.py --index --reset

# 4. Run the full evaluation
uv run python -m scripts.run_evaluation

# 5. Inspect results
head -5 runs/runs.csv
```

The evaluation will produce numbers within ~5% of the published table. Exact match is not guaranteed because `llama3.2:3b` samples stochastically even at `temperature=0.1`. To freeze behavior, set `temperature=0.0` in `agent.py::_build_llm`.

### Determinism guarantees

| Component | Deterministic? |
|---|---|
| Corpus loading | ✅ Yes |
| Chunking | ✅ Yes |
| Embedding | ✅ Yes (same model, same input → same vector) |
| Pinecone search | ✅ Yes (fixed index, fixed query) |
| LLM response | ❌ No (sampling even at low temperature) |
| Metrics computation | ✅ Yes |

---

## Acknowledgments

- [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/) for the agent framework.
- [Pinecone](https://www.pinecone.io/) for serverless vector storage.
- [Ollama](https://ollama.com/) for local model serving.
- The ReAct pattern was introduced by Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022).

---

## License

This is a capstone project — feel free to reuse for educational purposes.