import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import settings
from models.schemas import ResearchSession
from agents.base_agent import BaseAgent
from tools.memory import get_insights, get_summary

logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    api_key=settings.openai_api_key,
    model=settings.openai_model_smart,
    temperature=0.3,
)

REPORT_PROMPT = """You are a senior equity research analyst writing an institutional-grade financial research report.

RESEARCH GOAL: {goal}
QUERY TYPE: {query_type}
SECTOR: {sector}

REPORT STRUCTURE (follow this):
{report_structure}

SECTOR-SPECIFIC TEMPLATE:
{report_template}

RESEARCH FINDINGS:
{findings}

RULES:
1. Use ONLY the findings above. Do NOT fabricate any numbers or facts.
2. Where financial data is provided, use those exact numbers.
3. Separate reported vs adjusted figures when one-time charges exist.
4. Include specific deal names, partnership details, and analyst views if found.
5. Build comparative tables (markdown) wherever data allows — at least 2 tables.
6. Include forward-looking analysis: management guidance, growth drivers, risk factors.
7. Break down by geography and segment where data is available.
8. Cite sources inline like [Source: url or yfinance].
9. Use markdown: headers, bold for key numbers, bullet points, tables.
10. Report length: 2500-4000 words. Be comprehensive and analytical.

Generate the full research report in markdown."""


async def generate_report(session: ResearchSession, sector_agent: BaseAgent) -> str:
    """Generate the final markdown report from research findings."""

    insights = get_insights.invoke({})
    summary = get_summary.invoke({})

    # Format findings
    findings_text = ""
    for i in insights:
        findings_text += f"- [Step {i['step']}] {i['content']} (source: {i['source']})\n"

    report_structure = "\n".join(f"  - {s}" for s in session.plan.report_structure)

    prompt = REPORT_PROMPT.format(
        goal=session.plan.research_goal,
        query_type=session.plan.query_type,
        sector=session.plan.sector.value,
        report_structure=report_structure,
        report_template=sector_agent.report_template,
        findings=findings_text,
    )

    logger.info(f"Generating report for session {session.id} from {len(insights)} insights")

    response = await llm.ainvoke([HumanMessage(content=prompt)])

    report = response.content
    logger.info(f"Report generated: {len(report)} chars")

    return report
