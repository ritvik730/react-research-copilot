"""LangGraph ReAct agent: Reason -> Act -> Observe loop with max-step guard."""
from __future__ import annotations
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import OLLAMA_BASE_URL, LLM_MODEL, MAX_STEPS
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .tools import ALL_TOOLS
import re as _re

# Strip markdown link URL: [text](url) -> [text]
_MD_LINK_RE = _re.compile(r"\[([^\]]+)\]\([^)]*\)")

# Paren citation: (file.ext::anchor::N) or (file.ext) -> [file.ext]
_PAREN_CITE_RE = _re.compile(
    r"\(([A-Za-z0-9_\-]+\.(?:md|pdf|txt|json|csv))(?:::[A-Za-z0-9_\-]*::\d+)?\)"
)

# Bracket citation: [file.ext] or [file.ext:anything]
_BRACKET_CITE_RE = _re.compile(
    r"\[([A-Za-z0-9_\-]+\.(?:md|pdf|txt|json|csv))(?::([^\]]+))?\]"
)

# Valid locator shapes we will KEEP
_LINE_RANGE_RE = _re.compile(r"^\d+(?:-\d+)?$")            # 1  or  1-8
_ROW_RE        = _re.compile(r"^row\s*\d+$", _re.IGNORECASE) # row 3


def _normalize_citations(answer: str) -> str:
    s = answer
    s = _MD_LINK_RE.sub(r"[\1]", s)
    s = _PAREN_CITE_RE.sub(r"[\1]", s)

    def _fix(match: "_re.Match") -> str:
        fname = match.group(1)
        loc   = (match.group(2) or "").strip()
        if not loc:
            return f"[{fname}]"
        if _LINE_RANGE_RE.match(loc) or _ROW_RE.match(loc):
            return f"[{fname}:{loc}]"
        # Anything else (chunk_id residue, "t::0", etc.) -> file only
        return f"[{fname}]"

    s = _BRACKET_CITE_RE.sub(_fix, s)
    return s

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    question: str
    retrieved_files: list[str]
    log: list[dict[str, Any]]
    read_chunk_called: bool


def _build_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=512,
    )


def _reason_node(state: AgentState) -> dict:
    llm = _build_llm().bind_tools(ALL_TOOLS)
    response = llm.invoke(state["messages"])

    step = state.get("steps", 0) + 1
    entry = {
        "step": step,
        "type": "reason",
        "content": (response.content or "").strip()[:600],
        "tool_calls": [
            {"name": tc.get("name"), "args": tc.get("args")}
            for tc in (getattr(response, "tool_calls", None) or [])
        ],
    }
    log = list(state.get("log", [])) + [entry]
    return {"messages": [response], "steps": step, "log": log}


def _should_continue(state: AgentState) -> str:
    if state.get("steps", 0) >= MAX_STEPS:
        return "end"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    # About to end. Force at least one read_chunk before accepting a final answer.
    if not state.get("read_chunk_called", False):
        return "nudge"
    return "end"


def _nudge_node(state: AgentState) -> dict:
    """Inject a reminder that read_chunk is required before answering."""
    reminder = HumanMessage(content=(
        "STOP. You tried to answer WITHOUT calling `read_chunk`. "
        "Do NOT call `search` again. Do NOT answer. "
        "Call `read_chunk` NOW on the best chunk_id from your last search result. "
        "Example: read_chunk(chunk_id=\"01_ai_evals.md::t::0\")."
    ))
    log = list(state.get("log", [])) + [{
        "step": state.get("steps", 0),
        "type": "nudge",
        "content": "forced read_chunk",
    }]
    return {"messages": [reminder], "log": log}


def _tools_node(state: AgentState) -> dict:
    node = ToolNode(ALL_TOOLS)
    result = node.invoke(state)

    new_msgs = result.get("messages", [])
    retrieved = set(state.get("retrieved_files", []))
    log = list(state.get("log", []))
    read_chunk_called = state.get("read_chunk_called", False)

    for m in new_msgs:
        if isinstance(m, ToolMessage):
            if m.name == "read_chunk":
                read_chunk_called = True

            import json as _json
            try:
                obs = _json.loads(m.content) if isinstance(m.content, str) else m.content
            except Exception:
                obs = {}
            file_hint = None
            if isinstance(obs, dict) and m.name != "search":
                file_hint = obs.get("file")
                if file_hint:
                    retrieved.add(file_hint)

            log.append({
                "step": state.get("steps", 0),
                "type": "observe",
                "tool": m.name,
                "content": str(m.content)[:600],
            })

    return {
        "messages": new_msgs,
        "retrieved_files": sorted(retrieved),
        "log": log,
        "read_chunk_called": read_chunk_called,
    }


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("reason", _reason_node)
    g.add_node("tools",  _tools_node)
    g.add_node("nudge",  _nudge_node)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", _should_continue,
                            {"tools": "tools", "nudge": "nudge", "end": END})
    g.add_edge("tools", "reason")
    g.add_edge("nudge", "reason")
    return g.compile()


def run_agent(question: str) -> dict:
    agent = build_agent()

    init_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_user_prompt(question)),
        ],
        "steps": 0,
        "question": question,
        "retrieved_files": [],
        "log": [],
        "read_chunk_called": False,
    }

    final_state = agent.invoke(init_state, {"recursion_limit": MAX_STEPS * 2 + 6})

    answer = ""
    for m in reversed(final_state["messages"]):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            answer = m.content or ""
            break

    return {
        "question":        question,
        "answer":          _normalize_citations(answer.strip()),
        "steps":           final_state.get("steps", 0),
        "retrieved_files": final_state.get("retrieved_files", []),
        "log":             final_state.get("log", []),
    }