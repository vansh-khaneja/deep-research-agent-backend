import uuid
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pydantic import BaseModel, Field as PydanticField


class SessionStatus(str, Enum):
    AWAITING_APPROVAL = "awaiting_approval"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    FAILED = "failed"


class Sector(str, Enum):
    IT = "it"
    PHARMA = "pharma"
    BANKING = "banking"
    AUTO = "auto"
    ENERGY = "energy"
    FMCG = "fmcg"
    TELECOM = "telecom"
    UNKNOWN = "unknown"


# --- Research State (dataclass, like notebook) ---

@dataclass
class ResearchState:
    """Mutable state for the deep research loop."""
    session_id: str
    query: str
    sector: str = ""
    plan: List[str] = field(default_factory=list)
    plan_entities: List[str] = field(default_factory=list)
    report_structure: List[str] = field(default_factory=list)
    sector_context: str = ""
    insights: List[Dict[str, Any]] = field(default_factory=list)
    financial_data: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 8
    done: bool = False
    final_report: str = ""


# --- Research Plan (from planner, for API) ---

class ResearchStep(BaseModel):
    step_number: int
    description: str
    query_template: str
    tools: list[str] = PydanticField(default_factory=lambda: ["web_search"])
    success_criteria: list[str] = PydanticField(default_factory=list)


class ResearchPlan(BaseModel):
    sector: Sector
    research_goal: str
    query_type: str = ""
    research_depth: str = ""
    steps: list[ResearchStep] = PydanticField(default_factory=list)
    entities: list[str] = PydanticField(default_factory=list)
    tools_required: list[str] = PydanticField(default_factory=list)
    report_structure: list[str] = PydanticField(default_factory=list)


# --- Session (API layer) ---

class ResearchSession(BaseModel):
    id: str = PydanticField(default_factory=lambda: uuid.uuid4().hex[:12])
    user_query: str
    status: SessionStatus = SessionStatus.AWAITING_APPROVAL
    plan: ResearchPlan | None = None
    total_steps: int = 0
    current_step: int = 0
    current_step_description: str | None = None
    report_markdown: str | None = None
    created_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None


# --- Request/Response ---

class ResearchRequest(BaseModel):
    query: str


class ApprovalRequest(BaseModel):
    approved: bool


class PlanResponse(BaseModel):
    session_id: str
    sector: Sector
    plan: ResearchPlan
    message: str


class InsightItem(BaseModel):
    step: str
    content: str
    source: str


class ActivityItem(BaseModel):
    type: str       # plan, search, insight, discovery, reflection, financial, stop, step
    message: str
    detail: str = ""
    ts: float = 0.0


class StatusResponse(BaseModel):
    session_id: str
    status: SessionStatus
    current_step: int
    total_steps: int
    current_step_description: str | None = None
    insights: list[InsightItem] = PydanticField(default_factory=list)
    activity: list[ActivityItem] = PydanticField(default_factory=list)
    covered_steps: list[int] = PydanticField(default_factory=list)


class ReportResponse(BaseModel):
    session_id: str
    report_markdown: str
