---
marp: true
theme: default
paginate: true
---

# ReAct Research Copilot
### A tool-using single agent over a local document corpus

LangChain - LangGraph - Pinecone - Ollama (llama3.2:3b)

---

## 1. Problem & Approach

- **Problem:** LLMs hallucinate when answering questions about a private corpus.
- **Approach:** ReAct loop - Reason -> Act (tool) -> Observe -> repeat.
- Tools: `search`, `open_file`, `read_chunk`
- Evidence-based final answers with inline citations `[file:page]`.

---

## 2. Architecture

- User question -> LLM (llama3.2:3b)
- LLM emits tool_calls -> LangGraph ToolNode
- Tools: search / open_file / read_chunk
- All backed by Pinecone (768d, cosine, serverless us-east-1)

---

## 3. Demo Workflow

1. `python react_copilot.py --index --reset`
2. `python react_copilot.py -q "What is faithfulness?" -v`
3. Trace: Reason -> Act(search) -> Observe -> Act(read_chunk) -> Observe -> Final Answer with citations.

---

## 4. Evaluation

- 20 questions, two modes: **LLM-only** vs **ReAct + tools**
- Metrics: grounded precision, token-overlap accuracy, mean steps
- Output: `runs/runs.csv`

---

## 5. Failure Cases & Fixes

| Failure | Fix tried |
|---|---|
| Model answers without searching | Reinforced prompt + forced first-turn tool call |
| Cited a file it never read | Filter citations against `retrieved_files` |
| Loop > 8 steps | Hard `MAX_STEPS` guard |
| Wrong chunk retrieved | `open_file` fallback for full-file scan |

---

## 6. Takeaways

- Small local LLMs (3B) can drive ReAct reliably with structured tools + strict prompts.
- Grounding via Pinecone + explicit citations cuts hallucination sharply.
- LangGraph gives explicit control over steps, retries, and safety.
