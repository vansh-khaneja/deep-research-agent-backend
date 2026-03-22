import uuid
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    AWAITING_APPROVAL = "awaiting_approval"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    FAILED = "failed"


class Sector(str, Enum):
    IT = "it"
    PHARMA = "pharma"
    UNKNOWN = "unknown"


class ResearchStep(BaseModel):
    step_number: int
    description: str
    query_template: str
    tools: list[str] = Field(default_factory=lambda: ["web_search"])


class ResearchPlan(BaseModel):
    sector: Sector
    research_goal: str
    query_type: str = ""
    research_depth: str = ""
    steps: list[ResearchStep] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    tools_required: list[str] = Field(default_factory=list)
    report_structure: list[str] = Field(default_factory=list)


class ResearchSession(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_query: str
    status: SessionStatus = SessionStatus.AWAITING_APPROVAL
    plan: ResearchPlan | None = None
    total_steps: int = 0
    current_step: int = 0
    current_step_description: str | None = None
    report_markdown: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
    step: int
    content: str
    source: str


class ToolUseItem(BaseModel):
    tool: str
    step: int
    input: str = ""


class StatusResponse(BaseModel):
    session_id: str
    status: SessionStatus
    current_step: int
    total_steps: int
    current_step_description: str | None = None
    insights: list[InsightItem] = Field(default_factory=list)
    tools_used: list[ToolUseItem] = Field(default_factory=list)


class ReportResponse(BaseModel):
    session_id: str
    report_markdown: str
