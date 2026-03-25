# ============================================
# Financial Report Generation Agent
# Section-by-section writing with final editing
# ============================================

import json
import re
import logging
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings
from models.schemas import ResearchState
from agents.base_agent import BaseAgent
from tools.memory import log_activity

logger = logging.getLogger("deep-research")

llm_smart = ChatOpenAI(
    api_key=settings.openai_api_key,
    model=settings.openai_model_smart,
    temperature=0.3,
)

llm_editor = ChatOpenAI(
    api_key=settings.openai_api_key,
    model=settings.openai_model_smart,
    temperature=0.2,
)


# =======================
# Step 6: Section Writer
# =======================

SECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a senior equity research analyst writing ONE section of a financial research report.

Rules:
- Write ONLY the section requested — no other sections
- Use specific numbers, dates, company names from the findings
- Do NOT fabricate data — only use what's in the findings
- Professional, institutional tone
- Cite sources inline like [Source: url]
- Financial data contains QUARTERLY and TTM figures — label correctly
- Always specify currency (INR/USD)
- 300-600 words per section

FORMATTING REQUIREMENTS (MUST FOLLOW):
- Use **bold** for key metrics and important numbers
- Use bullet points (- ) for listing multiple data points, risks, or takeaways
- Use markdown tables (| col1 | col2 |) for ANY comparative data (company vs company, quarter vs quarter, metric comparisons)
- Use ### subheadings within the section to organize content
- Mix prose paragraphs with tables and bullet lists — never write a wall of text
- For Financial Analysis: ALWAYS include a quarterly comparison table
- For Competitive Landscape: ALWAYS include a comparison table
- For Key Risks: use bullet points with bold risk names

Example formatting:
### Revenue Performance
Revenue reached **$5.1B** in Q3, up **1.7% YoY**.

| Metric | Q3 FY26 | Q2 FY26 | QoQ Change |
|--------|---------|---------|------------|
| Revenue | $5.1B | $5.07B | +0.6% |

Key drivers:
- **AI services** contributed 5.5% of revenue
- **Large deal wins** totaled $4.8B TCV"""),
    ("human",
     """REPORT TITLE: {title}
SECTION TO WRITE: {section_heading}
QUERY: {query}
SECTOR: {sector}

RELEVANT RESEARCH FINDINGS:
{findings}

FINANCIAL DATA:
{financial_data}

Write ONLY the "{section_heading}" section in markdown. Start with ## {section_heading}
Remember: Use tables, bullet points, bold numbers, and subheadings. No walls of text.""")
])

section_chain = SECTION_PROMPT | llm_smart | StrOutputParser()


# =======================
# Step 7: Final Editor
# =======================

EDITOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a senior editor reviewing a financial research report.

Your job:
1. Remove repetition — if the same fact appears in multiple sections, keep it only where most relevant
2. Improve flow — ensure smooth transitions between sections
3. Strengthen insights — upgrade vague statements to specific conclusions
4. Align tone — consistent professional voice throughout
5. Add a # title at the top
6. Ensure Executive Summary is concise (150-250 words) and captures key findings
7. Do NOT add new data — only edit existing content
8. Keep all source citations intact
9. Output the final report in clean markdown

CRITICAL — PRESERVE FORMATTING:
- Keep ALL markdown tables — do not convert them to prose
- Keep ALL bullet point lists
- Keep **bold** formatting on key numbers
- Keep ### subheadings within sections
- The report should look visually structured, not like a plain essay"""),
    ("human",
     """QUERY: {query}

DRAFT REPORT (section by section):
{draft_report}

Edit and polish this into a cohesive, professional research report.""")
])

editor_chain = EDITOR_PROMPT | llm_editor | StrOutputParser()


# =======================
# Main Function
# =======================

async def generate_report(state: ResearchState, sector_agent: BaseAgent) -> str:
    """Generate report section-by-section, then do a final editing pass."""

    sid = state.session_id
    logger.info(f"Generating section-by-section report from {len(state.insights)} insights")

    # --- Organize notes by topic ---
    log_activity(sid, "writing", "Organizing collected research notes", "")

    topic_notes = {}
    for insight in state.insights:
        step = insight.get("step", "general")
        if step not in topic_notes:
            topic_notes[step] = []
        topic_notes[step].append(insight.get("insight", ""))

    all_notes = "\n".join(
        f"- {i.get('insight', '')[:500]}" for i in state.insights
    )

    # Financial data
    fin_text = json.dumps(state.financial_data, default=str, indent=2)[:8000] if state.financial_data else "No financial data available."

    # Report structure — prefer planner's, fall back to agent's template, then generic
    if state.report_structure:
        sections = state.report_structure
    else:
        # Extract section headings from the sector agent's report template
        template_headings = re.findall(r'^##\s+(.+)$', sector_agent.report_template, re.MULTILINE)
        sections = template_headings if template_headings else [
            "Executive Summary",
            "Company/Sector Overview",
            "Financial Analysis",
            "Business Segments & Strategy",
            "Competitive Landscape",
            "Key Risks & Challenges",
            "Outlook & Recommendations",
            "Sources",
        ]

    writable_sections = [s for s in sections if s.lower() not in ("sources", "references")]

    title = f"{state.sector.upper()} Sector Research Report"

    # --- Write each section ---
    draft_sections = []

    for i, heading in enumerate(sections):
        if heading.lower() in ("sources", "references"):
            continue

        log_activity(sid, "writing", f"Writing: {heading}", "")
        logger.info(f"Writing section {i+1}/{len(sections)}: {heading}")

        section_text = await asyncio.to_thread(section_chain.invoke, {
            "title": title,
            "section_heading": heading,
            "query": state.query,
            "sector": state.sector,
            "findings": all_notes[:8000],
            "financial_data": fin_text if "financial" in heading.lower() or "executive" in heading.lower() else fin_text[:2000],
        })

        draft_sections.append(section_text)
        logger.info(f"Section '{heading}' generated: {len(section_text)} chars")

    # Add sources section
    log_activity(sid, "writing", "Compiling sources", "")

    sources = set()
    for insight in state.insights:
        src = insight.get("source", "")
        if src and src.startswith("http"):
            sources.add(src)
        import re
        urls = re.findall(r'https?://[^\s\]"]+', insight.get("insight", ""))
        sources.update(urls)

    sources_section = "## Sources\n\n"
    for url in sorted(sources):
        url = url.rstrip(".,;:)]}")
        sources_section += f"- [{url[:80]}]({url})\n"

    draft_sections.append(sources_section)

    # Combine all sections
    draft_report = "\n\n".join(draft_sections)
    logger.info(f"Draft report assembled: {len(draft_report)} chars, {len(draft_sections)} sections")

    # --- Final editing pass ---
    log_activity(sid, "writing", "Reviewing and polishing final report", "")

    final_report = await asyncio.to_thread(editor_chain.invoke, {
        "query": state.query,
        "draft_report": draft_report[:15000],
    })

    log_activity(sid, "writing", "Report ready", "")
    logger.info(f"Final report: {len(final_report)} chars")
    return final_report
