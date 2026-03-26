# ============================================
# Deep Research Agent — Step-by-Step LangGraph
# Sequential plan execution with:
# - ReAct agent per step (LLM decides which tools to call)
# - Plan Critic verifies step completion via success criteria
# - Retry logic for incomplete steps
# - All tools available at every step
# ============================================

from __future__ import annotations

import re
import json
import logging
import asyncio
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool as lc_tool

import httpx
import trafilatura
from tavily import TavilyClient

from config import settings
from models.schemas import ResearchSession, SessionStatus, ResearchState
from agents.base_agent import BaseAgent
from tools.memory import (
    update_progress, store_insight as memory_store_insight, log_activity,
    is_stop_requested, log_tool_call, set_covered_steps, save_findings,
)
from tools.rag import search_documents

logger = logging.getLogger("deep-research")


# =======================
# LLM
# =======================

def _llm(temperature: float = 0.2):
    return ChatOpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url, model=settings.model_fast, temperature=temperature)

def _llm_smart(temperature: float = 0.3):
    return ChatOpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url, model=settings.model_smart, temperature=temperature)


# =======================
# Graph State
# =======================

class AgentState(TypedDict):
    # Core
    session_id: str
    question: str
    sector: str
    sector_context: str
    entities: List[str]
    report_structure: List[str]

    # Plan (structured)
    plan_steps: List[dict]  # [{description, query_template, tools, success_criteria}]

    # Sequential step execution
    current_step_index: int
    step_notes: List[List[str]]  # notes per step
    step_sources: List[List[str]]  # sources per step
    step_status: List[str]  # "pending" | "done" | "skipped"

    # Current step's ReAct agent
    agent_messages: List[Any]
    agent_iteration: int
    max_agent_iterations: int

    # Critic
    critic_verdict: str  # "pass" | "fail" | ""
    step_retries: int

    # Global
    financial_data: List[dict]
    all_notes: List[str]  # flattened notes across all steps
    answer_draft: str
    citations: List[str]
    done: bool


# =======================
# Utils
# =======================

def _clean_text(txt: str, max_chars: int = 40_000) -> str:
    return re.sub(r"\s+", " ", txt).strip()[:max_chars]

def _web_search(query: str, k: int = 5) -> list:
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        res = client.search(query=query, search_depth="advanced", max_results=k)
        hits = []
        for r in res.get("results", []):
            url = r.get("url", "")
            if url:
                hits.append({"url": url, "title": r.get("title", url), "snippet": (r.get("content") or "")[:300]})
        return hits[:k]
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def _web_fetch(url: str, timeout: float = 20.0) -> Optional[dict]:
    if not url or not url.startswith(("http://", "https://")):
        return None
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": "DeepResearchAgent/1.0"}) as client:
            r = client.get(url)
            r.raise_for_status()
            downloaded = trafilatura.extract(r.text, include_comments=False, include_images=False, url=url)
            content = _clean_text(downloaded if downloaded else r.text)
            # Skip binary/PDF content
            if content.startswith("%PDF") or "\x00" in content[:100]:
                return None
            title_m = re.search(r"<title>(.*?)</title>", r.text, re.I)
            title = title_m.group(1).strip() if title_m else url
            return {"url": url, "title": title, "content": content}
    except Exception:
        return None

def _resolve_ticker(company_name: str) -> str:
    try:
        import requests
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        res = requests.get(url, params={"q": company_name}, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        quotes = res.json().get("quotes", [])
        for q in quotes:
            if q.get("exchDisp") == "NSE" and q.get("quoteType") == "EQUITY":
                return q["symbol"]
        for q in quotes:
            if q.get("quoteType") == "EQUITY":
                return q["symbol"]
    except Exception:
        pass
    return ""

def _fetch_financial_data(ticker: str) -> Dict[str, Any]:
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        fi = stock.fast_info
        price_currency = info.get("currency", "INR" if ticker.endswith(".NS") else "USD")
        financial_currency = info.get("financialCurrency", price_currency)

        overview = {
            "ticker": ticker, "name": info.get("longName") or info.get("shortName", ""),
            "financial_currency": financial_currency, "price_currency": price_currency,
            "current_price": round(fi.last_price, 2) if fi.last_price else None,
            "market_cap": fi.market_cap,
            "pe_ratio": info.get("trailingPE"), "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"), "return_on_equity": info.get("returnOnEquity"),
        }

        qf = stock.quarterly_financials
        quarters = []
        if qf is not None and not qf.empty:
            for col in qf.columns[:4]:
                q = {"period": str(col.date()), "currency": financial_currency}
                for m in ["Total Revenue", "Net Income", "EBITDA", "Operating Income"]:
                    if m in qf.index:
                        val = qf.loc[m, col]
                        q[m.lower().replace(" ", "_")] = int(val) if val == val else None
                quarters.append(q)

        annual_totals = {"currency": financial_currency}
        if len(quarters) >= 4:
            for metric in ["total_revenue", "net_income", "ebitda", "operating_income"]:
                vals = [q.get(metric) for q in quarters[:4] if q.get(metric) is not None]
                if len(vals) == 4:
                    annual_totals[f"{metric}_ttm"] = sum(vals)

        return {"overview": overview, "quarterly_data": quarters, "trailing_12m_totals": annual_totals}
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# =======================
# Tool Builder
# =======================

def _build_agent_tools(sid: str):
    """Build all tools available to the agent during each step."""
    from tools.calculator import calculate_growth_rate, calculate_cagr, calculate_margin, calculate_ratio, compare_metrics

    @lc_tool
    def web_search_tool(query: str) -> str:
        """Search the web for financial information. Use short, specific queries (5-8 words)."""
        log_activity(sid, "searching", f"Searching: {query}", "")
        hits = _web_search(query)
        log_tool_call(sid, "web_search", query, json.dumps(hits[:3], ensure_ascii=False)[:2000])
        if not hits:
            return "No results found."
        return "\n\n".join(f"[{h['title']}]\n{h['snippet']}\nURL: {h['url']}" for h in hits)

    @lc_tool
    def web_read_tool(url: str) -> str:
        """Read and extract the full text content from a web page URL. Also works with PDF files."""
        log_activity(sid, "reading", f"Reading: {url[:80]}", "")

        # Handle PDF files — download and extract text
        if url.lower().endswith('.pdf'):
            try:
                import io
                from pypdf import PdfReader
                resp = httpx.get(url, follow_redirects=True, timeout=30.0,
                                 headers={"User-Agent": "DeepResearchAgent/1.0"})
                resp.raise_for_status()
                reader = PdfReader(io.BytesIO(resp.content))
                text = ""
                for page in reader.pages[:20]:  # max 20 pages
                    text += page.extract_text() or ""
                text = _clean_text(text)
                if len(text) < 50:
                    log_tool_call(sid, "web_read", url, "PDF: no extractable text")
                    return "PDF downloaded but no extractable text found."
                log_tool_call(sid, "web_read", url, f"PDF: {len(text)} chars from {min(len(reader.pages), 20)} pages")
                return f"PDF Content ({len(reader.pages)} pages):\n\n{text[:8000]}"
            except Exception as e:
                log_tool_call(sid, "web_read", url, f"PDF FAILED: {e}")
                return f"Failed to read PDF: {e}"

        # Handle regular web pages
        doc = _web_fetch(url)
        read_output = f"title: {doc['title'][:100]} | {len(doc.get('content',''))} chars" if doc else "FAILED"
        log_tool_call(sid, "web_read", url, read_output)
        if not doc or not doc.get("content"):
            return "Failed to read page or no content."
        return f"Title: {doc['title']}\n\n{doc['content'][:8000]}"

    @lc_tool
    def fetch_stock_data(company_name: str) -> str:
        """Fetch real-time financial data for a company from Yahoo Finance. Pass the company name (e.g. 'Infosys', 'TCS', 'Sun Pharma')."""
        log_activity(sid, "financial", f"Fetching financial data for {company_name}", "")
        ticker = _resolve_ticker(company_name)
        log_tool_call(sid, "ticker_resolve", company_name, ticker or "NOT_FOUND")
        if not ticker:
            return f"Could not resolve ticker for '{company_name}'"
        data = _fetch_financial_data(ticker)
        log_tool_call(sid, "yfinance", ticker, json.dumps(data.get("overview", {}), ensure_ascii=False)[:2000])
        return json.dumps(data, indent=2, default=str)[:6000]

    @lc_tool
    def search_documents_tool(query: str) -> str:
        """Search uploaded PDF documents in the knowledge base. Returns relevant passages."""
        results = search_documents(query, session_id=sid, k=3)
        log_tool_call(sid, "rag_search", query, json.dumps(results[:2], ensure_ascii=False)[:1000] if results else "[]")
        if not results:
            return "No relevant documents found in knowledge base."
        return "\n\n".join(f"[Doc: {r['source']}] (score: {r['score']:.2f})\n{r['content'][:500]}" for r in results)

    # Calculator tools — wrap existing ones with logging
    @lc_tool
    def calc_growth_rate(current: float, previous: float) -> str:
        """Calculate growth rate between two values. Returns percentage."""
        result = calculate_growth_rate(current, previous)
        log_tool_call(sid, "calculate_growth_rate", f"{current} vs {previous}", json.dumps(result))
        return json.dumps(result)

    @lc_tool
    def calc_margin(part: float, total: float) -> str:
        """Calculate margin/percentage (part/total * 100). E.g., net income / revenue."""
        result = calculate_margin(part, total)
        log_tool_call(sid, "calculate_margin", f"{part} / {total}", json.dumps(result))
        return json.dumps(result)

    @lc_tool
    def calc_compare(metric_name: str, company_a: str, value_a: float, company_b: str, value_b: float) -> str:
        """Compare a metric between two companies. E.g., compare P/E ratio of TCS vs Infosys."""
        result = compare_metrics(company_a, value_a, company_b, value_b, metric_name)
        log_tool_call(sid, "compare_metrics", f"{metric_name}: {company_a} vs {company_b}", json.dumps(result))
        return json.dumps(result)

    return [web_search_tool, web_read_tool, fetch_stock_data, search_documents_tool,
            calc_growth_rate, calc_margin, calc_compare]


# =======================
# Graph Nodes
# =======================

SYSTEM_PROMPT = """You are a Deep Financial Research Agent. You have access to tools for:
- web_search_tool: Search the web (use short 5-8 word queries)
- web_read_tool: Read full content from a URL
- fetch_stock_data: Get real-time financial data from Yahoo Finance
- search_documents_tool: Search uploaded PDF documents
- calc_growth_rate: Calculate growth rate between two numbers
- calc_margin: Calculate margin/percentage
- calc_compare: Compare a metric between two companies

You are researching ONE specific step of a research plan.
Be thorough — call multiple tools, read multiple pages, fetch financial data.

{sector_context}

IMPORTANT: Before calling tools, briefly explain your reasoning — what you're looking for and why.
After receiving tool results, briefly summarize what you learned and what you still need.
This thinking process will be shown to the user as research activity.

When you have enough information to answer the step's questions, write your findings as a detailed summary."""


def _build_system_prompt(sector_context: str) -> str:
    ctx = f"SECTOR GUIDANCE:\n{sector_context}" if sector_context else ""
    return SYSTEM_PROMPT.format(sector_context=ctx)


def node_init_step(state: AgentState) -> dict:
    """Initialize the agent for the current plan step."""
    sid = state["session_id"]
    idx = state["current_step_index"]
    steps = state["plan_steps"]

    if idx >= len(steps) or state.get("done"):
        return {"done": True}

    if is_stop_requested(sid):
        log_activity(sid, "stop", "Research stopped by user", "")
        return {"done": True}

    step = steps[idx]
    total = len(steps)

    log_activity(sid, "step", f"Step {idx + 1}/{total}: {step['description'][:80]}", "")
    update_progress(sid, idx, f"Researching step {idx + 1}/{total}: {step['description'][:60]}...")

    # Build the initial prompt for this step's agent
    # Include context from previous steps
    prev_context = ""
    for i in range(idx):
        prev_notes = state["step_notes"][i]
        if prev_notes:
            prev_context += f"\n--- Step {i+1} findings (already done) ---\n"
            prev_context += "\n".join(prev_notes[-3:])  # last 3 notes from each prev step

    criteria_text = "\n".join(f"  - {c}" for c in step.get("success_criteria", []))

    prompt = f"""You are researching step {idx + 1} of {total} for the query: "{state['question']}"

CURRENT STEP: {step['description']}

SUCCESS CRITERIA — your research must answer these questions:
{criteria_text}

SUGGESTED SEARCH QUERY: {step.get('query_template', '')}

ENTITIES TO RESEARCH: {', '.join(state.get('entities', [])) or 'None specified'}
(If any entity is a company, call fetch_stock_data for it)

{f"CONTEXT FROM PREVIOUS STEPS:{prev_context}" if prev_context else "This is the first step."}

INSTRUCTIONS:
1. Start by searching the web with relevant queries
2. Read the most promising URLs for detailed content
3. IMPORTANT: If the step mentions a company name, you MUST call fetch_stock_data("company name") to get real-time financial data from Yahoo Finance. Do not skip this.
4. IMPORTANT: When you have two numbers to compare (revenue, profit, margins), you MUST call calc_growth_rate or calc_margin instead of computing in your head. These tools give exact results.
5. Once you have enough data to answer ALL success criteria, write a detailed summary of your findings

Be specific — include actual numbers, dates, company names, and source URLs."""

    messages = [
        SystemMessage(content=_build_system_prompt(state.get("sector_context", ""))),
        HumanMessage(content=prompt),
    ]

    return {
        "agent_messages": messages,
        "agent_iteration": 0,
        "critic_verdict": "",
    }


def node_agent_act(state: AgentState) -> dict:
    """Execute one round of the ReAct agent — call LLM, handle tool calls."""
    import time
    sid = state["session_id"]
    messages = list(state["agent_messages"])

    if is_stop_requested(sid):
        return {"done": True}

    # Small delay to avoid rate limits
    time.sleep(0.5)

    tools = _build_agent_tools(sid)
    llm_with_tools = _llm_smart(temperature=0.2).bind_tools(tools)

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # If the LLM made tool calls, execute them
    if response.tool_calls:
        tool_map = {t.name: t for t in tools}
        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                messages.append(ToolMessage(content=str(result)[:4000], tool_call_id=tc["id"]))
            else:
                messages.append(ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]))

        # Log the agent's reasoning text (if it explained why it's calling these tools)
        if response.content and len(response.content.strip()) > 20:
            title = response.content.strip()[:150].split("\n")[0]
            log_activity(sid, "thinking", title, response.content.strip()[:500])

        # Log tool calls as one compact line — e.g. "Used 3 tools: web_search x1, web_read x2"
        from collections import Counter
        tool_counts = Counter(tc["name"] for tc in response.tool_calls)
        parts = [f"{name.replace('_tool','')} x{count}" if count > 1 else name.replace('_tool','') for name, count in tool_counts.items()]
        log_activity(sid, "searching", f"Used {len(response.tool_calls)} tools: {', '.join(parts)}", "")
    else:
        # No tool calls — this is the agent's final summary/reasoning for this step
        if response.content and len(response.content.strip()) > 20:
            # Extract a short title from the first line
            lines = response.content.strip().split("\n")
            title = lines[0][:150] if lines else "Summarizing findings"
            log_activity(sid, "thinking", title, response.content.strip()[:800])

    return {
        "agent_messages": messages,
        "agent_iteration": state["agent_iteration"] + 1,
    }


def node_plan_critic(state: AgentState) -> dict:
    """Check if the current step's success criteria are met by the agent's findings."""
    sid = state["session_id"]
    idx = state["current_step_index"]
    step = state["plan_steps"][idx]

    # Extract the agent's final text response (last AI message without tool calls)
    findings = ""
    for msg in reversed(state["agent_messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            findings = msg.content
            break

    if not findings:
        findings = "No findings produced."

    criteria = step.get("success_criteria", [])
    criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))

    critic_prompt = f"""You are a research quality critic. Check if the research findings adequately answer each success criterion.

STEP: {step['description']}

SUCCESS CRITERIA:
{criteria_text}

RESEARCH FINDINGS:
{findings[:4000]}

For each criterion, say YES (answered with specific data) or NO (missing or vague).
Then give an overall verdict: "pass" if ALL criteria are answered, "fail" if any are missing.

Return JSON:
{{"checks": [{{"criterion": "...", "answered": true/false, "evidence": "brief quote"}}], "verdict": "pass|fail", "missing": ["what's still needed"]}}"""

    try:
        response = _llm(temperature=0.1).invoke([HumanMessage(content=critic_prompt)])
        content = response.content

        # Parse JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            verdict = result.get("verdict", "fail").lower()
            missing = result.get("missing", [])
            checks = result.get("checks", [])

            # Log detailed results with per-criterion breakdown
            passed = sum(1 for c in checks if c.get("answered"))
            total = len(checks)

            # Build clean detail for frontend — no special characters
            detail_lines = []
            for c in checks:
                status = "answered" if c.get("answered") else "missing"
                detail_lines.append(f"[{status}] {c.get('criterion', '')[:80]}")
            if missing:
                detail_lines.append(f"Still needed: {', '.join(missing)}")
            detail = " | ".join(detail_lines)

            verdict_text = "complete" if verdict == "pass" else "needs more data"
            log_activity(sid, "thinking",
                         f"Step check: {passed}/{total} criteria met, {verdict_text}",
                         detail)
            log_tool_call(sid, "plan_critic", f"step {idx}", json.dumps(result, ensure_ascii=False)[:2000])
        else:
            verdict = "pass"  # If can't parse, assume pass
            log_activity(sid, "thinking", "Plan Critic: could not parse response, assuming pass", "")
    except Exception as e:
        verdict = "pass"
        logger.error(f"Critic error: {e}")

    return {"critic_verdict": verdict}


def node_advance_step(state: AgentState) -> dict:
    """Advance to the next step or retry the current one."""
    sid = state["session_id"]
    idx = state["current_step_index"]
    verdict = state["critic_verdict"]

    # Extract findings from agent messages — collect ALL AI reasoning, use last as main findings
    all_findings = []
    sources = []
    for msg in state["agent_messages"]:
        if isinstance(msg, AIMessage) and msg.content and len(msg.content.strip()) > 30:
            if not msg.tool_calls:
                all_findings.append(msg.content.strip())
        if isinstance(msg, ToolMessage) and msg.content:
            urls = re.findall(r'https?://[^\s\]"]+', msg.content)
            sources.extend(urls[:3])
    # Use the last (most complete) AI response as the main findings
    findings = all_findings[-1] if all_findings else ""

    # Update step notes and sources
    new_step_notes = list(state["step_notes"])
    new_step_sources = list(state["step_sources"])
    new_step_status = list(state["step_status"])

    if findings:
        new_step_notes[idx] = new_step_notes[idx] + [findings[:2000]]
    new_step_sources[idx] = list(set(new_step_sources[idx] + sources))

    # Decide: advance, retry, or done
    retries = state["step_retries"]

    if verdict == "pass" or retries >= 2:
        # Mark step as done
        status = "done" if verdict == "pass" else "skipped"
        new_step_status[idx] = status
        if status == "done":
            log_activity(sid, "step",
                         f"Step {idx + 1} complete",
                         state["plan_steps"][idx]["description"][:60])

        # Update frontend checkboxes
        covered = [i for i, s in enumerate(new_step_status) if s in ("done", "skipped")]
        set_covered_steps(sid, covered)

        # Flatten all notes for findings
        all_notes = []
        for notes in new_step_notes:
            all_notes.extend(notes)

        # Save findings after each step
        try:
            save_findings(
                session_id=sid,
                query=state["question"],
                plan=json.dumps([s["description"] for s in state["plan_steps"]]),
                notes=all_notes,
                financial_data=state.get("financial_data", []),
                report="",
                sources=list(set(s for sl in new_step_sources for s in sl)),
            )
        except Exception:
            pass

        next_idx = idx + 1
        is_done = next_idx >= len(state["plan_steps"])

        return {
            "step_notes": new_step_notes,
            "step_sources": new_step_sources,
            "step_status": new_step_status,
            "current_step_index": next_idx,
            "step_retries": 0,
            "all_notes": all_notes,
            "done": is_done,
        }
    else:
        # Retry — keep same step index, add context about what's missing
        log_activity(sid, "thinking", f"Retrying step {idx + 1} — criteria not fully met (attempt {retries + 2})", "")

        return {
            "step_notes": new_step_notes,
            "step_sources": new_step_sources,
            "step_retries": retries + 1,
            "critic_verdict": "",
        }


# =======================
# Routing Functions
# =======================

def agent_routing(state: AgentState) -> str:
    """Should agent continue calling tools or move to critic?"""
    if state.get("done"):
        return "done"

    messages = state.get("agent_messages", [])
    if not messages:
        return "critic"

    last = messages[-1]

    # If last message is AIMessage with no tool calls — agent is done, go to critic
    if isinstance(last, AIMessage) and not last.tool_calls:
        return "critic"

    # If we've exceeded max iterations, force to critic
    if state["agent_iteration"] >= state["max_agent_iterations"]:
        return "critic"

    # Otherwise keep going (tool results just came back, agent needs to process them)
    return "continue"


def step_routing(state: AgentState) -> str:
    """Should we go to next step, retry, or finish?"""
    if state.get("done"):
        return "done"

    if is_stop_requested(state["session_id"]):
        return "done"

    # If critic said fail and we have retries left
    if state["critic_verdict"] == "fail" and state["step_retries"] > 0 and state["step_retries"] <= 2:
        return "retry"

    # If there are more steps
    if state["current_step_index"] < len(state["plan_steps"]):
        return "next_step"

    return "done"


# =======================
# Build Graph
# =======================

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("init_step", node_init_step)
    graph.add_node("agent_act", node_agent_act)
    graph.add_node("plan_critic", node_plan_critic)
    graph.add_node("advance_step", node_advance_step)

    graph.set_entry_point("init_step")

    # init_step → agent_act
    graph.add_edge("init_step", "agent_act")

    # agent_act → agent_act (more tools) | plan_critic (agent done) | END (force stop)
    graph.add_conditional_edges("agent_act", agent_routing, {
        "continue": "agent_act",
        "critic": "plan_critic",
        "done": END,
    })

    # plan_critic → advance_step
    graph.add_edge("plan_critic", "advance_step")

    # advance_step → init_step (next/retry) | END (all done)
    graph.add_conditional_edges("advance_step", step_routing, {
        "next_step": "init_step",
        "retry": "init_step",
        "done": END,
    })

    return graph.compile(checkpointer=MemorySaver())


# =======================
# Entry Point
# =======================

async def run_research(session: ResearchSession, sector_agent: BaseAgent) -> None:
    """Execute the step-by-step research agent for a session."""
    try:
        session.status = SessionStatus.RESEARCHING
        sid = session.id
        update_progress(sid, 0, "Starting deep research agent...")
        log_activity(sid, "step", "Deep research agent started", f"Query: {session.user_query[:80]}")

        plan_steps = [
            {
                "description": step.description,
                "query_template": step.query_template,
                "tools": step.tools,
                "success_criteria": step.success_criteria,
            }
            for step in session.plan.steps
        ]

        num_steps = len(plan_steps)
        log_activity(sid, "plan", f"Plan: {num_steps} steps to execute sequentially",
                     " | ".join(s["description"][:40] for s in plan_steps))

        initial_state: AgentState = {
            "session_id": sid,
            "question": session.user_query,
            "sector": session.plan.sector.value,
            "sector_context": sector_agent.research_context,
            "entities": session.plan.entities or [],
            "report_structure": session.plan.report_structure or [],
            "plan_steps": plan_steps,
            "current_step_index": 0,
            "step_notes": [[] for _ in range(num_steps)],
            "step_sources": [[] for _ in range(num_steps)],
            "step_status": ["pending"] * num_steps,
            "agent_messages": [],
            "agent_iteration": 0,
            "max_agent_iterations": 8,
            "critic_verdict": "",
            "step_retries": 0,
            "financial_data": [],
            "all_notes": [],
            "answer_draft": "",
            "citations": [],
            "done": False,
        }

        config = {"configurable": {"thread_id": f"research-{sid}"}}

        app = build_graph()
        final_state = await asyncio.to_thread(app.invoke, initial_state, config)

        # Generate final report
        update_progress(sid, num_steps, "Generating final report...")
        log_activity(sid, "writing", "Research complete — generating report", "")

        from core.report_agent import generate_report
        all_notes = final_state.get("all_notes", [])
        all_sources = list(set(s for sl in final_state.get("step_sources", []) for s in sl))

        research_state = ResearchState(
            session_id=sid,
            query=session.user_query,
            sector=session.plan.sector.value,
            plan=[],
            plan_entities=session.plan.entities or [],
            report_structure=session.plan.report_structure or [],
            insights=[{"step": "research", "insight": n, "source": "agent"} for n in all_notes],
            financial_data=final_state.get("financial_data", []),
        )
        session.report_markdown = await generate_report(research_state, sector_agent)

        # Save final findings with report
        save_findings(
            session_id=sid,
            query=session.user_query,
            plan=json.dumps([s["description"] for s in plan_steps]),
            notes=all_notes,
            financial_data=final_state.get("financial_data", []),
            report=session.report_markdown or "",
            sources=all_sources,
        )

        session.current_step = num_steps
        session.total_steps = num_steps
        session.current_step_description = "Report ready"
        session.status = SessionStatus.COMPLETED
        update_progress(sid, num_steps, "Report ready")
        log_activity(sid, "stop", "Research complete — report generated",
                     f"{len(all_notes)} notes, {len(all_sources)} sources")

        logger.info(f"Session {sid}: completed, {num_steps} steps")

    except Exception as e:
        logger.error(f"Research failed for session {session.id}: {e}", exc_info=True)
        session.status = SessionStatus.FAILED
        session.error = str(e)
