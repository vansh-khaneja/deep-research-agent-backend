import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.schemas import (
    ResearchRequest, ApprovalRequest,
    PlanResponse, StatusResponse, ReportResponse, InsightItem, ToolUseItem,
    SessionStatus, ResearchSession, Sector,
)
from tools.memory import get_insights, get_tools_used, set_session, get_progress
from tools.llm import LLMClient
from core.planner import SectorClassifier, ResearchPlanner
from core.research_agent import run_research
from agents.registry import AgentRegistry
from agents.it_agent import ITAgent
from agents.pharma_agent import PharmaAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])

# Initialize
llm = LLMClient()
classifier = SectorClassifier(llm)
planner = ResearchPlanner(llm)

registry = AgentRegistry()
registry.register(ITAgent())
registry.register(PharmaAgent())

session_store: dict[str, ResearchSession] = {}


@router.post("/", response_model=PlanResponse)
async def submit_research(req: ResearchRequest):

    # 1. Classify
    route, _ = await classifier.classify(req.query)

    if route == "decline":
        raise HTTPException(status_code=400, detail="Not a financial query or outside supported sectors (IT/Pharma).")

    if route == "clarify":
        raise HTTPException(status_code=400, detail="Could not determine sector. Specify IT or Pharma.")

    # 2. Route to agent
    sector = Sector(route)
    agent = registry.get(sector)

    # 3. Generate plan
    plan = await planner.create_plan(query=req.query, sector=sector, agent=agent)

    # 4. Store session
    session = ResearchSession(
        user_query=req.query,
        plan=plan,
        status=SessionStatus.AWAITING_APPROVAL,
        total_steps=len(plan.steps),
    )
    session_store[session.id] = session

    logger.info(f"Session {session.id}: {plan.query_type}, {plan.research_depth}, {len(plan.steps)} steps")

    return PlanResponse(
        session_id=session.id,
        sector=sector,
        plan=plan,
        message=f"{plan.research_depth.capitalize()} research plan for {sector.value.upper()} sector. "
                f"{len(plan.steps)} steps planned.",
    )


@router.post("/{session_id}/approve", response_model=StatusResponse)
async def approve_plan(session_id: str, req: ApprovalRequest, background_tasks: BackgroundTasks):

    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SessionStatus.AWAITING_APPROVAL:
        raise HTTPException(status_code=400, detail=f"Session is '{session.status}', not awaiting approval")

    if not req.approved:
        session.status = SessionStatus.FAILED
        session.error = "Plan rejected by user"
        return StatusResponse(
            session_id=session.id,
            status=session.status,
            current_step=0,
            total_steps=session.total_steps,
            current_step_description="Plan rejected",
        )

    # Start research agent in background
    session.status = SessionStatus.RESEARCHING
    sector_agent = registry.get(session.plan.sector)
    background_tasks.add_task(run_research, session, sector_agent)

    logger.info(f"Session {session.id}: approved, research agent started")

    return StatusResponse(
        session_id=session.id,
        status=session.status,
        current_step=0,
        total_steps=session.total_steps,
        current_step_description="Research started...",
    )


@router.get("/{session_id}/status", response_model=StatusResponse)
async def get_status(session_id: str):

    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Set session context so memory reads the right session's data
    set_session(session_id)

    raw_insights = get_insights.invoke({})
    insights = [
        InsightItem(step=i["step"], content=i["content"], source=i["source"])
        for i in raw_insights
    ]

    raw_tools = get_tools_used()
    tools_used = [
        ToolUseItem(tool=t["tool"], step=t["step"], input=t["input"])
        for t in raw_tools
    ]

    # Use live progress from memory during research, fall back to session fields when done
    progress = get_progress()
    if session.status == SessionStatus.RESEARCHING:
        current_step = progress["current_step"]
        description = progress["description"]
    else:
        current_step = session.current_step
        description = session.current_step_description

    return StatusResponse(
        session_id=session.id,
        status=session.status,
        current_step=current_step,
        total_steps=session.total_steps,
        current_step_description=description,
        insights=insights,
        tools_used=tools_used,
    )


@router.get("/{session_id}/report", response_model=ReportResponse)
async def get_report(session_id: str):

    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Research not complete. Status: {session.status}")

    if not session.report_markdown:
        raise HTTPException(status_code=400, detail="Report not generated yet")

    return ReportResponse(
        session_id=session.id,
        report_markdown=session.report_markdown,
    )
