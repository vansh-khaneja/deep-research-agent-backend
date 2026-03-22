import logging
from contextvars import ContextVar
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Session-scoped storage using contextvars (safe for concurrent async tasks)
_current_session: ContextVar[str] = ContextVar("current_session", default="default")
_sessions: dict[str, dict] = {}


def _empty_store() -> dict:
    return {
        "insights": [],
        "tools_used": [],
        "progress": {"current_step": 0, "description": "Starting..."},
    }


def _get_store() -> dict:
    sid = _current_session.get()
    if sid not in _sessions:
        _sessions[sid] = _empty_store()
    return _sessions[sid]


def set_session(session_id: str):
    _current_session.set(session_id)


def reset():
    sid = _current_session.get()
    _sessions[sid] = _empty_store()


def cleanup_session(session_id: str):
    _sessions.pop(session_id, None)


# --- Progress tracking (live updates for status endpoint) ---

def update_progress(step: int, description: str):
    store = _get_store()
    store["progress"] = {"current_step": step, "description": description}


def get_progress() -> dict:
    return _get_store()["progress"]


# --- Tool use tracking ---

def record_tool_use(tool_name: str, step: int, input_summary: str = ""):
    _get_store()["tools_used"].append({"tool": tool_name, "step": step, "input": input_summary})


def get_tools_used() -> list[dict]:
    return _get_store()["tools_used"]


# --- Insights (these are LangChain @tools so the agent can call them) ---

@tool
def store_insight(step_number: int, content: str, source: str) -> str:
    """Store a research finding. You MUST call this after every meaningful discovery.
    Include specific numbers (revenue, margins, growth rates) and cite the source.

    Args:
        step_number: Sequential step number (1, 2, 3, ...)
        content: The insight text with specific data points and numbers
        source: Where this came from — a URL, 'yfinance', or 'calculated'
    """
    store = _get_store()
    store["insights"].append({"step": step_number, "content": content, "source": source})
    # Also record as tool use for the status endpoint
    store["tools_used"].append({"tool": "store_insight", "step": step_number, "input": content[:100]})
    total = len(store["insights"])
    logger.info(f"Memory: stored insight from step {step_number}, total={total}")
    return f"Stored. Total insights: {total}"


@tool
def get_insights() -> list[dict]:
    """Get all research findings gathered so far. Use this to review what you've collected."""
    return _get_store()["insights"]


@tool
def get_summary() -> str:
    """Get a condensed summary of all findings so far. Call this before searching
    to check what you already know and avoid redundant searches."""
    insights = _get_store()["insights"]
    if not insights:
        return "No findings yet."
    lines = [f"- [Step {i['step']}] {i['content']}" for i in insights[-20:]]
    return "\n".join(lines)
