import logging

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import settings
from models.schemas import ResearchSession, SessionStatus
from agents.base_agent import BaseAgent

from tools.search import web_search
from tools.finance import (
    get_company_overview, get_financials,
    get_balance_sheet, get_stock_price_history,
)
from tools.ticker_resolver import resolve_ticker
from tools.calculator import (
    calculate_growth_rate, calculate_cagr,
    calculate_margin, calculate_ratio, compare_metrics,
)
from tools.memory import (
    set_session, reset as reset_memory,
    store_insight, get_insights, get_summary,
    update_progress,
)
from core.report_agent import generate_report

logger = logging.getLogger(__name__)

# --- LLM ---
# Use the fast model for the agent loop — it makes many calls and needs higher rate limits.
# The smart model is reserved for the final report synthesis where quality matters most.

llm = ChatOpenAI(
    api_key=settings.openai_api_key,
    model=settings.openai_model_fast,
    temperature=0.2,
)

# --- All tools the agent can use ---

ALL_TOOLS = [
    # Research
    web_search,
    # Financial data
    resolve_ticker,
    get_company_overview,
    get_financials,
    get_balance_sheet,
    get_stock_price_history,
    # Calculations
    calculate_growth_rate,
    calculate_cagr,
    calculate_margin,
    calculate_ratio,
    compare_metrics,
    # Memory
    store_insight,
    get_insights,
    get_summary,
]

# --- System prompt template ---

AGENT_SYSTEM_PROMPT = """You are a senior financial research agent. You have access to tools for web search, financial data retrieval, and calculations.

## Your Research Plan

Research Goal: {research_goal}
Query Type: {query_type}
Sector: {sector}

Sector Context:
{sector_context}

Planned Research Steps (use as guidance, not as a rigid script):
{planned_steps}

Key Entities to Investigate: {entities}
Expected Report Sections: {report_structure}

## How to Work

1. **BATCH TOOL CALLS.** This is critical for performance. Call MULTIPLE tools in a single turn whenever possible:
   - After a web_search, call store_insight in the SAME turn (don't wait for a separate turn)
   - Call resolve_ticker for multiple companies in one turn
   - After resolve_ticker returns, call get_company_overview + get_financials + get_balance_sheet in one turn
   - After getting financial data, call store_insight + calculate_growth_rate + calculate_margin all at once
   - GOOD: web_search → [store_insight + next web_search] → [store_insight + resolve_ticker] → [get_financials + get_company_overview + get_balance_sheet]
   - BAD: web_search → store_insight → next web_search → store_insight (one tool per turn wastes API calls)

2. **Use store_insight after every meaningful finding.** The final report is built entirely from stored insights. Findings NOT stored will be LOST. Include specific numbers and the source.

3. **Start broad, then go deep.** Begin with web searches to understand the landscape, then use financial APIs for specific company data.

4. **For company financial data, always:**
   - First call resolve_ticker to get the Yahoo Finance ticker symbol
   - Then batch: get_company_overview + get_financials + get_balance_sheet + get_stock_price_history in ONE turn
   - Use calculator tools (calculate_growth_rate, calculate_margin, etc.) to derive metrics from raw data

5. **Avoid redundancy.** Don't search for information you already have from a previous tool call.

6. **Prioritize authoritative sources** — official filings, investor presentations, analyst reports over SEO content farms.

7. **Cover all angles the plan asks for**, but adapt based on what you find. If you discover something important not in the plan, investigate it. If a planned step is already covered by earlier findings, skip it.

8. **Stop when done.** Once you have sufficient data covering the research goal, key financials, competitive context, risks, and forward outlook — stop. Do not keep searching for marginal additions.

9. **Always include specific numbers** in your insights: revenue figures, growth percentages, margins, P/E ratios, deal values, etc.

## Important Rules
- ALWAYS call multiple tools per turn when possible — this dramatically reduces latency
- Call store_insight frequently — findings NOT stored will be LOST
- Include the source URL or "yfinance" or "calculated" in each store_insight call
- Step numbers should be sequential (1, 2, 3, ...)
- Do NOT fabricate data — only store what tools return
"""


def _format_planned_steps(session: ResearchSession) -> str:
    if not session.plan.steps:
        return "No specific steps planned — use your judgment."
    lines = []
    for s in session.plan.steps:
        tools_str = ", ".join(s.tools) if s.tools else "web_search"
        lines.append(f"  {s.step_number}. {s.description} (suggested tools: {tools_str})")
    return "\n".join(lines)


# --- Progress-tracking wrapper ---

class ProgressTracker:
    """Wraps the agent stream to update progress as tool calls happen."""

    def __init__(self):
        self.tool_call_count = 0

    def on_tool_call(self, tool_name: str, tool_input: str = ""):
        self.tool_call_count += 1
        short_input = tool_input[:80] if tool_input else ""
        update_progress(self.tool_call_count, f"{tool_name}: {short_input}")


# --- Entry Point ---

async def run_research(session: ResearchSession, sector_agent: BaseAgent) -> None:
    """Execute the research agent for a session."""
    try:
        session.status = SessionStatus.RESEARCHING
        set_session(session.id)
        reset_memory()

        max_steps = len(session.plan.steps) * 4 + 10  # generous budget for tool calls

        system_prompt = AGENT_SYSTEM_PROMPT.format(
            research_goal=session.plan.research_goal,
            query_type=session.plan.query_type,
            sector=session.plan.sector.value,
            sector_context=sector_agent.research_context,
            planned_steps=_format_planned_steps(session),
            entities=", ".join(session.plan.entities) if session.plan.entities else "None specified",
            report_structure=", ".join(session.plan.report_structure) if session.plan.report_structure else "Use standard structure",
            max_steps=max_steps,
        )

        # Build the react agent
        agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            prompt=system_prompt,
        )

        tracker = ProgressTracker()
        update_progress(0, "Starting research...")

        logger.info(
            f"Starting research agent for session {session.id}, "
            f"goal='{session.plan.research_goal}', max_steps={max_steps}"
        )

        # Stream the agent to track progress in real-time
        user_message = (
            f"Research this query: {session.user_query}\n\n"
            f"Begin your research now. Remember to call store_insight after each meaningful finding."
        )

        async for event in agent.astream(
            {"messages": [HumanMessage(content=user_message)]},
            config={"recursion_limit": max_steps},
        ):
            # Track tool calls for progress updates
            if "messages" in event:
                for msg in event["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tracker.on_tool_call(
                                tc.get("name", "unknown"),
                                str(tc.get("args", ""))[:80],
                            )

        session.current_step = tracker.tool_call_count
        session.current_step_description = "Generating report..."

        # Check we actually gathered insights
        insights = get_insights.invoke({})
        logger.info(
            f"Research done for session {session.id}: "
            f"{len(insights)} insights from {tracker.tool_call_count} tool calls. "
            f"Generating report..."
        )

        if not insights:
            logger.warning(f"Session {session.id}: No insights stored! Agent may have failed to call store_insight.")

        # Generate report
        session.report_markdown = await generate_report(session, sector_agent)
        session.current_step_description = "Report ready"
        session.status = SessionStatus.COMPLETED
        update_progress(tracker.tool_call_count, "Report ready")

        logger.info(f"Report generated for session {session.id}")

    except Exception as e:
        logger.error(f"Research failed for session {session.id}: {e}", exc_info=True)
        session.status = SessionStatus.FAILED
        session.error = str(e)
