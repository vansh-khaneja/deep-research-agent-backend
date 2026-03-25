import json
import os
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


def log_tool_call(session_id: str, tool: str, tool_input: str = "", tool_output: str = ""):
    """Track tool usage counts in memory for tools_summary in findings.json."""
    if session_id:
        store = _get_store(session_id)
        if "tools_used" not in store:
            store["tools_used"] = {}
        store["tools_used"][tool] = store["tools_used"].get(tool, 0) + 1

# Global session storage — shared across all threads
_sessions: dict[str, dict] = {}


def _empty_store() -> dict:
    return {
        "progress": {"current_step": 0, "description": "Starting..."},
        "insights": [],
        "activity": [],
        "force_stop": False,
        "covered_steps": [],  # list of plan step indices that are covered
    }


def _get_store(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = _empty_store()
    return _sessions[session_id]


def cleanup_session(session_id: str):
    _sessions.pop(session_id, None)


# --- Progress tracking ---

def update_progress(session_id: str, step: int, description: str):
    store = _get_store(session_id)
    store["progress"] = {"current_step": step, "description": description}


def get_progress(session_id: str) -> dict:
    return _get_store(session_id)["progress"]


# --- Live insights ---

def store_insight(session_id: str, step: str, content: str, source: str):
    store = _get_store(session_id)
    store["insights"].append({"step": step, "content": content, "source": source})


def get_insights(session_id: str) -> list[dict]:
    return _get_store(session_id)["insights"]


# --- Activity / reasoning stream ---

def log_activity(session_id: str, event_type: str, message: str, detail: str = ""):
    store = _get_store(session_id)
    store["activity"].append({
        "type": event_type,
        "message": message,
        "detail": detail,
        "ts": time.time(),
    })


def set_covered_steps(session_id: str, covered: list[int]):
    store = _get_store(session_id)
    store["covered_steps"] = covered


def get_covered_steps(session_id: str) -> list[int]:
    return _get_store(session_id).get("covered_steps", [])


def get_activity(session_id: str) -> list[dict]:
    return _get_store(session_id)["activity"]


# --- Force stop ---

def request_stop(session_id: str):
    store = _get_store(session_id)
    store["force_stop"] = True


def is_stop_requested(session_id: str) -> bool:
    return _get_store(session_id).get("force_stop", False)


# --- Findings persistence (JSON, most recent only) ---

_FINDINGS_PATH = "./data/findings.json"


def save_findings(session_id: str, query: str, plan: str, notes: list[str],
                  financial_data: list[dict], report: str, sources: list[str]):
    """Save research findings to JSON. Overwrites previous — only keeps most recent."""
    os.makedirs(os.path.dirname(_FINDINGS_PATH), exist_ok=True)

    # Build tools summary from in-memory tool log
    store = _get_store(session_id)
    tools_summary = store.get("tools_used", {})

    findings = {
        "session_id": session_id,
        "query": query,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "plan": plan,
        "notes": notes,
        "financial_data": financial_data,
        "report": report,
        "sources": sources,
        "total_notes": len(notes),
        "total_sources": len(sources),
        "tools_summary": tools_summary,
    }
    with open(_FINDINGS_PATH, "w") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved findings for session {session_id} ({len(notes)} notes, {len(sources)} sources)")


def load_findings() -> dict | None:
    """Load the most recent findings."""
    if not os.path.exists(_FINDINGS_PATH):
        return None
    with open(_FINDINGS_PATH) as f:
        return json.load(f)
