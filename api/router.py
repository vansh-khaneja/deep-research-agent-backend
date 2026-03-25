import logging
import asyncio
from fastapi import APIRouter, HTTPException
from models.schemas import (
    ResearchRequest, ApprovalRequest,
    PlanResponse, StatusResponse, ReportResponse, InsightItem, ActivityItem,
    SessionStatus, ResearchSession, Sector,
)
from tools.memory import get_progress, get_insights, get_activity, request_stop, get_covered_steps
from tools.llm import LLMClient
from core.planner import SectorClassifier, ResearchPlanner
from core.research_engine import run_research
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])

# Initialize — agents are auto-discovered from the agents/ package
registry = AgentRegistry()
registry.auto_discover()

llm = LLMClient()
classifier = SectorClassifier(llm, sector_descriptions=registry.sector_descriptions())
planner = ResearchPlanner(llm)

session_store: dict[str, ResearchSession] = {}


@router.post("/", response_model=PlanResponse)
async def submit_research(req: ResearchRequest):

    # 1. Classify
    route, _ = await classifier.classify(req.query)

    if route == "decline":
        available = ", ".join(s.upper() for s in registry.sector_descriptions())
        raise HTTPException(status_code=400, detail=f"Not a financial query or outside supported sectors ({available}).")

    if route == "clarify":
        raise HTTPException(status_code=422, detail="CLARIFY_SECTOR")

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
                f"{len(plan.steps)} steps planned. Deep research with discovery + reflection.",
    )


@router.post("/{session_id}/approve", response_model=StatusResponse)
async def approve_plan(session_id: str, req: ApprovalRequest):

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

    # Start deep research in background
    session.status = SessionStatus.RESEARCHING
    sector_agent = registry.get(session.plan.sector)
    asyncio.create_task(run_research(session, sector_agent))

    logger.info(f"Session {session.id}: approved, deep research started")

    return StatusResponse(
        session_id=session.id,
        status=session.status,
        current_step=0,
        total_steps=session.total_steps,
        current_step_description="Deep research started — Plan → Search → Discover → Reflect...",
    )


@router.get("/{session_id}/status", response_model=StatusResponse)
async def get_status(session_id: str):

    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    progress = get_progress(session_id)
    if session.status == SessionStatus.RESEARCHING:
        current_step = progress["current_step"]
        description = progress["description"]
    else:
        current_step = session.current_step
        description = session.current_step_description

    # Get live insights from memory
    raw_insights = get_insights(session_id)
    insights = [
        InsightItem(step=i["step"], content=i["content"], source=i["source"])
        for i in raw_insights
    ]

    # Get activity/reasoning stream
    raw_activity = get_activity(session_id)
    activity = [
        ActivityItem(type=a["type"], message=a["message"], detail=a.get("detail", ""), ts=a.get("ts", 0))
        for a in raw_activity
    ]

    return StatusResponse(
        session_id=session.id,
        status=session.status,
        current_step=current_step,
        total_steps=session.total_steps,
        current_step_description=description,
        insights=insights,
        activity=activity,
        covered_steps=get_covered_steps(session_id),
    )


@router.post("/{session_id}/stop")
async def stop_research(session_id: str):
    """Force stop research and generate report from whatever has been collected."""
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SessionStatus.RESEARCHING:
        raise HTTPException(status_code=400, detail=f"Session is '{session.status}', not researching")

    request_stop(session_id)
    logger.info(f"Session {session_id}: force stop requested")

    return {"session_id": session_id, "message": "Stop requested. Report will generate from collected data."}


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
