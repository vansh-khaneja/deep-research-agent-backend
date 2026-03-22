import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import settings
from models.schemas import ResearchSession, SessionStatus
from agents.base_agent import BaseAgent

from tools.search import web_search
from tools.finance import get_company_overview, get_financials
from tools.ticker_resolver import resolve_ticker
from tools.memory import store_insight, get_summary, reset as reset_memory, record_tool_use
from core.report_agent import generate_report

logger = logging.getLogger(__name__)

# Smart LLM — for insight extraction, financial decisions (needs reasoning)
llm_smart = ChatOpenAI(
    api_key=settings.openai_api_key,
    model=settings.openai_model_smart,
    temperature=0.2,
)

# Fast LLM — for relevance checks, query refinement (simple yes/no or short text)
llm_fast = ChatOpenAI(
    api_key=settings.openai_api_key,
    model=settings.openai_model_fast,
    temperature=0.2,
)


# --- State ---

class ResearchState(TypedDict):
    user_query: str
    research_goal: str
    sector_context: str

    current_query: str
    step_number: int
    max_steps: int

    insights: list[dict]
    entities: list[str]

    last_results: list[dict] | None
    relevant: bool | None
    need_financials: bool | None

    done: bool


# --- Nodes ---

def search_node(state: ResearchState):
    results = web_search.invoke({"query": state["current_query"]})
    record_tool_use("web_search", state["step_number"], state["current_query"])
    logger.info(f"Step {state['step_number']}: searched '{state['current_query']}' → {len(results)} results")
    return {"last_results": results}


def insight_node(state: ResearchState):
    results = state["last_results"] or []
    if not results:
        return {"relevant": False}

    text = "\n\n".join(
        f"Title: {r['title']}\nContent: {r['content'][:500]}"
        for r in results[:4]
    )

    prompt = f"""You are a financial research analyst. Extract ONE key insight from these search results.

Research Goal: {state['research_goal']}

Search Results:
{text}

Return STRICT JSON with exactly 2 fields:

1. "insight" — one key finding in 2-3 sentences WITH specific numbers (revenue, %, ratios)

2. "entities" — ONLY publicly traded company names.
   GOOD examples: ["Infosys", "TCS", "Wipro", "Sun Pharma", "HCL Technologies"]
   BAD examples (NEVER include these): ["revenue", "EBITDA", "Q3 FY26", "growth outlook", "net profit", "deal pipeline", "employee metrics", "market", "sector"]

   Rule: If you cannot buy stock in it, it is NOT an entity.

{{
  "insight": "...",
  "entities": ["CompanyName1", "CompanyName2"]
}}"""

    res = llm_smart.invoke([HumanMessage(content=prompt)])

    try:
        data = json.loads(res.content)
    except json.JSONDecodeError:
        data = {"insight": res.content[:300], "entities": []}

    # Store in memory
    store_insight.invoke({
        "step_number": state["step_number"],
        "content": data["insight"],
        "source": results[0].get("url", "web") if results else "web",
    })

    # Safety net — skip obvious non-company words the LLM might still pass
    raw_entities = data.get("entities", [])
    clean_entities = [
        e for e in raw_entities
        if len(e) > 2
        and e[0].isupper()  # company names start with uppercase
        and not any(w in e.lower() for w in ["revenue", "profit", "ebitda", "margin", "growth", "outlook", "metric", "q3", "q4", "fy2", "sector"])
    ]

    new_entities = list(set(state.get("entities", []) + clean_entities))

    return {
        "insights": state.get("insights", []) + [data["insight"]],
        "entities": new_entities,
    }


def relevance_node(state: ResearchState):
    insights = state.get("insights", [])
    if not insights:
        return {"relevant": False}

    latest = insights[-1]

    prompt = f"""Research Goal: {state['research_goal']}

Latest Finding:
{latest}

Is this finding useful and relevant for deep financial research on the goal?
Answer only YES or NO."""

    res = llm_fast.invoke([HumanMessage(content=prompt)])
    relevant = "YES" in res.content.upper()
    logger.info(f"Step {state['step_number']}: relevance = {relevant}")

    return {"relevant": relevant}


_fetched_tickers: set[str] = set()


def financial_decision_node(state: ResearchState):
    entities = state.get("entities", [])
    if not entities:
        return {"need_financials": False}

    unfetched = [e for e in entities if e.lower() not in _fetched_tickers]
    if not unfetched:
        return {"need_financials": False}

    summary = get_summary.invoke({})

    prompt = f"""You are a financial research agent.

Research goal: {state['research_goal']}

Entities discovered: {entities}
Entities WITHOUT financial data yet: {unfetched}

Research findings so far:
{summary}

Do we need to fetch detailed financial data (stock price, revenue, margins, balance sheet) for the unfetched entities?

Consider:
- If the research goal involves financial analysis, comparison, or valuation → YES
- If we already have enough financial numbers from web search → NO
- If the query asks about trends, news, or regulations only → NO

Answer YES or NO with brief reason."""

    res = llm_fast.invoke([HumanMessage(content=prompt)])
    need = "YES" in res.content.upper()
    logger.info(f"Step {state['step_number']}: need_financials = {need} (unfetched: {unfetched}), LLM: {res.content[:50]}")

    return {"need_financials": need}


def financial_node(state: ResearchState):
    entities = state.get("entities", [])
    if not entities:
        return {}

    from tools.finance import get_balance_sheet, get_stock_price_history
    from tools.calculator import calculate_growth_rate, calculate_margin

    unfetched = [e for e in entities if e.lower() not in _fetched_tickers]
    step = state["step_number"]

    for idx, company in enumerate(unfetched):
        current_step = step + idx  # each company gets its own step number

        # Resolve ticker
        ticker = resolve_ticker.invoke({"company_name": company})
        record_tool_use("resolve_ticker", current_step, company)

        if ticker == "NOT_FOUND":
            logger.warning(f"Ticker not found for {company}")
            continue

        _fetched_tickers.add(company.lower())

        # Company overview
        overview = get_company_overview.invoke({"ticker": ticker})
        record_tool_use("get_company_overview", current_step, ticker)

        store_insight.invoke({
            "step_number": current_step,
            "content": f"{company} ({ticker}): Price={overview.get('current_price')}, "
                       f"P/E={overview.get('pe_ratio')}, MarketCap={overview.get('market_cap')}, "
                       f"Revenue Growth={overview.get('revenue_growth')}, "
                       f"Profit Margin={overview.get('profit_margin')}, "
                       f"ROE={overview.get('return_on_equity')}, D/E={overview.get('debt_to_equity')}",
            "source": "yfinance",
        })

        # Quarterly financials
        financials = get_financials.invoke({"ticker": ticker})
        record_tool_use("get_financials", current_step, ticker)

        quarters = financials.get("quarterly", [])
        if len(quarters) >= 2:
            q0, q1 = quarters[0], quarters[1]
            store_insight.invoke({
                "step_number": current_step,
                "content": f"{company} latest quarter ({q0.get('period')}): "
                           f"Revenue={q0.get('revenue')}, Net Income={q0.get('net_income')}, "
                           f"EBITDA={q0.get('ebitda')}, Operating Income={q0.get('operating_income')}",
                "source": "yfinance",
            })

            # Calculate QoQ growth
            if q0.get("revenue") and q1.get("revenue"):
                growth = calculate_growth_rate.invoke({"current": q0["revenue"], "previous": q1["revenue"]})
                record_tool_use("calculate_growth_rate", current_step, f"{company} QoQ revenue")
                store_insight.invoke({
                    "step_number": current_step,
                    "content": f"{company} QoQ revenue growth: {growth.get('formatted', 'N/A')}",
                    "source": "calculated",
                })

            # Calculate profit margin
            if q0.get("net_income") and q0.get("revenue"):
                margin = calculate_margin.invoke({"part": q0["net_income"], "total": q0["revenue"]})
                record_tool_use("calculate_margin", current_step, f"{company} profit margin")
                store_insight.invoke({
                    "step_number": current_step,
                    "content": f"{company} net profit margin: {margin.get('formatted', 'N/A')}",
                    "source": "calculated",
                })

        # Balance sheet
        balance = get_balance_sheet.invoke({"ticker": ticker})
        record_tool_use("get_balance_sheet", current_step, ticker)

        bs = balance.get("balance_sheet", {})
        if bs:
            store_insight.invoke({
                "step_number": current_step,
                "content": f"{company} balance sheet: Total Assets={bs.get('total_assets')}, "
                           f"Equity={bs.get('stockholders_equity')}, "
                           f"Cash={bs.get('cash')}, Debt={bs.get('total_debt')}",
                "source": "yfinance",
            })

        # Price history
        prices = get_stock_price_history.invoke({"ticker": ticker})
        record_tool_use("get_stock_price_history", current_step, ticker)

        store_insight.invoke({
            "step_number": current_step,
            "content": f"{company} stock: Current={prices.get('current')}, "
                       f"6mo High={prices.get('period_high')}, Low={prices.get('period_low')}",
            "source": "yfinance",
        })

        logger.info(f"Step {current_step}: financial deep dive done for {company} ({ticker})")

    return {}


def refine_query_node(state: ResearchState):
    summary = get_summary.invoke({})

    prompt = f"""You are a financial research agent.

Research goal: {state['research_goal']}
Sector context: {state['sector_context']}

Findings so far:
{summary}

Step {state['step_number']} of {state['max_steps']} completed.

Generate the NEXT specific web search query to deepen this research.
Focus on what's MISSING — financials, competitor data, risks, analyst views, or forward outlook.
Return ONLY the search query, nothing else."""

    res = llm_fast.invoke([HumanMessage(content=prompt)])
    new_query = res.content.strip().strip('"')

    logger.info(f"Step {state['step_number']}: refined query → '{new_query}'")

    return {
        "current_query": new_query,
        "step_number": state["step_number"] + 1,
    }


# --- Router ---

def controller_router(state: ResearchState):
    if state["step_number"] >= state["max_steps"]:
        return "finish"

    if not state.get("relevant", True):
        return "refine"

    if state.get("need_financials", False):
        return "financial"

    return "refine"


# --- Build Graph ---

def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("search", search_node)
    workflow.add_node("insight", insight_node)
    workflow.add_node("relevance", relevance_node)
    workflow.add_node("financial_decision", financial_decision_node)
    workflow.add_node("financial", financial_node)
    workflow.add_node("refine", refine_query_node)

    workflow.set_entry_point("search")

    workflow.add_edge("search", "insight")
    workflow.add_edge("insight", "relevance")
    workflow.add_edge("relevance", "financial_decision")

    workflow.add_conditional_edges(
        "financial_decision",
        controller_router,
        {
            "financial": "financial",
            "refine": "refine",
            "finish": "__end__",
        }
    )

    workflow.add_edge("financial", "refine")
    workflow.add_edge("refine", "search")

    return workflow.compile()


# Compile once
research_graph = build_graph()


# --- Entry Point ---

async def run_research(session: ResearchSession, sector_agent: BaseAgent) -> None:
    """Execute the full research loop for a session."""
    try:
        session.status = SessionStatus.RESEARCHING
        reset_memory()
        _fetched_tickers.clear()

        initial_state: ResearchState = {
            "user_query": session.user_query,
            "research_goal": session.plan.research_goal,
            "sector_context": sector_agent.research_context,
            "current_query": session.plan.steps[0].query_template if session.plan.steps else session.user_query,
            "step_number": 1,
            "max_steps": len(session.plan.steps) + 5,  # planned steps + room for dynamic ones
            "insights": [],
            "entities": session.plan.entities or [],
            "last_results": None,
            "relevant": None,
            "need_financials": None,
            "done": False,
        }

        logger.info(f"Starting research graph for session {session.id}, max_steps={initial_state['max_steps']}")

        # Run the graph
        result = await research_graph.ainvoke(initial_state)

        session.current_step = result.get("step_number", 0)
        session.current_step_description = "Generating report..."

        logger.info(
            f"Research done for session {session.id}: "
            f"{len(result.get('insights', []))} insights, "
            f"{len(result.get('entities', []))} entities. Synthesizing report..."
        )

        # Generate report
        session.report_markdown = await generate_report(session, sector_agent)
        session.current_step_description = "Report ready"
        session.status = SessionStatus.COMPLETED

        logger.info(f"Report generated for session {session.id}")

    except Exception as e:
        logger.error(f"Research failed for session {session.id}: {e}", exc_info=True)
        session.status = SessionStatus.FAILED
        session.error = str(e)
