import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Simple in-memory store
_insights: list[dict] = []
_searched_queries: set[str] = set()
_tools_used: list[dict] = []  # {"tool": name, "step": number, "input": summary}


def reset():
    global _insights, _searched_queries, _tools_used
    _insights = []
    _searched_queries = set()
    _tools_used = []


def record_tool_use(tool_name: str, step: int, input_summary: str = ""):
    _tools_used.append({"tool": tool_name, "step": step, "input": input_summary})


def get_tools_used() -> list[dict]:
    return _tools_used


@tool
def store_insight(step_number: int, content: str, source: str) -> str:
    """Store a research finding. Call this after every search to save key facts and numbers."""
    _insights.append({"step": step_number, "content": content, "source": source})
    logger.info(f"Memory: stored insight from step {step_number}, total={len(_insights)}")
    return f"Stored. Total insights: {len(_insights)}"


@tool
def get_insights() -> list[dict]:
    """Get all research findings gathered so far."""
    return _insights


@tool
def get_summary() -> str:
    """Get a condensed summary of all findings for context."""
    if not _insights:
        return "No findings yet."
    lines = [f"- [Step {i['step']}] {i['content']}" for i in _insights[-20:]]
    return "\n".join(lines)


@tool
def is_duplicate_query(query: str) -> bool:
    """Check if this search query was already executed. Returns True if duplicate."""
    normalized = query.lower().strip()
    if normalized in _searched_queries:
        return True
    _searched_queries.add(normalized)
    return False
