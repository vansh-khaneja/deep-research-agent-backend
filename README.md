# Financial Deep Research Agent

AI-powered deep research agent for financial sectors. Conducts multi-step research workflows with autonomous tool usage, self-evaluation, and structured report generation — similar to Claude's Deep Research or OpenAI's research mode, specialized for finance.

Currently supports **IT** and **Pharma** sectors, with a pluggable architecture to add more.

## Architecture

```
User Query
    │
    ▼
Sector Classifier (GPT-4o-mini)
    │
    ▼
Research Planner (GPT-4o) ──→ Structured Plan with steps + success criteria
    │
    ▼
[User Reviews & Approves Plan]
    │
    ▼
LangGraph Research Engine (per step):
    init_step → agent_act ⟲ (ReAct loop) → plan_critic → advance_step → next step
    │
    ▼
Report Agent (GPT-4o) ──→ Section-by-section writing → Final editing pass
    │
    ▼
Markdown Report
```

### Core Components

| Component | File | Role |
|-----------|------|------|
| **Sector Classifier** | `core/planner.py` | Routes query to the right sector agent (IT/Pharma/decline/clarify) |
| **Research Planner** | `core/planner.py` | Generates structured plan with steps, success criteria, entities |
| **Research Engine** | `core/research_engine.py` | LangGraph state machine — ReAct agent per step + plan critic + retry |
| **Report Agent** | `core/report_agent.py` | Writes each section separately, then a final editing pass |
| **Sector Agents** | `agents/it_agent.py`, `agents/pharma_agent.py` | Sector-specific context, metrics, and report templates |
| **Agent Registry** | `agents/registry.py` | Auto-discovers agents from `agents/` directory |
| **RAG Module** | `tools/rag.py` | Session-scoped ChromaDB for uploaded PDFs |
| **LLM Client** | `tools/llm.py` | Wrapper with structured output support |
| **Memory** | `tools/memory.py` | Session state, activity logging, findings persistence |

### Tools Available to the Research Agent

| Tool | Source | Purpose |
|------|--------|---------|
| `web_search_tool` | Tavily API | Web search for news, reports, analysis |
| `web_read_tool` | httpx + trafilatura | Read full content from web pages and PDFs |
| `fetch_stock_data` | Yahoo Finance + yfinance | Real-time financial data (resolves company name → ticker automatically) |
| `search_documents_tool` | ChromaDB (RAG) | Search uploaded PDF documents |
| `calc_growth_rate` | Python | Programmatic growth rate calculation |
| `calc_margin` | Python | Margin/percentage calculation |
| `calc_compare` | Python | Side-by-side metric comparison between companies |

### How the Research Engine Works

1. **Plan** — LLM generates ordered steps with success criteria per step
2. **ReAct Loop** — For each step, GPT-4o autonomously picks tools, searches, reads pages, fetches financial data
3. **Plan Critic** — GPT-4o-mini evaluates if success criteria are met (YES/NO per criterion)
4. **Retry** — If criteria not met, retries the step (max 2 retries), then moves on
5. **Context Passing** — Each step's findings are passed to the next step as context
6. **Report** — Section-by-section writing with a final editing/deduplication pass

### Adding a New Sector Agent

Drop a new file in `agents/` — it gets auto-discovered:

```python
# agents/banking_agent.py
from models.schemas import Sector
from agents.base_agent import BaseAgent

class BankingAgent(BaseAgent):
    @property
    def sector(self) -> Sector:
        return Sector.BANKING

    @property
    def sector_description(self) -> str:
        return "Banking / Financial Services sector"

    @property
    def planning_context(self) -> str:
        return """Key areas: NPA ratios, NIM, CASA ratio, credit growth..."""

    @property
    def research_context(self) -> str:
        return """Focus on banking sector: RBI policies, credit growth..."""

    @property
    def report_template(self) -> str:
        return """## Executive Summary\n## Financial Analysis\n..."""

    @property
    def key_metrics(self) -> list[str]:
        return ["NPA Ratio", "Net Interest Margin", "CASA Ratio", ...]
```

No other file changes needed — the registry and classifier pick it up automatically.

## Project Structure

```
├── main.py                  # FastAPI app, CORS, router mounting
├── config.py                # Settings from .env
│
├── api/
│   ├── router.py            # /research endpoints (submit, approve, status, stop, report)
│   └── documents.py         # /documents/upload endpoint (PDF upload for RAG)
│
├── core/
│   ├── planner.py           # SectorClassifier + ResearchPlanner
│   ├── research_engine.py   # LangGraph state machine (ReAct + Critic)
│   └── report_agent.py      # Section-by-section report generation
│
├── agents/
│   ├── base_agent.py        # Abstract base class
│   ├── registry.py          # Auto-discovery and registration
│   ├── it_agent.py          # IT sector configuration
│   └── pharma_agent.py      # Pharma sector configuration
│
├── tools/
│   ├── llm.py               # LLM client with structured output
│   ├── memory.py            # Session state, activity log, findings persistence
│   ├── rag.py               # ChromaDB session-scoped RAG
│   └── calculator.py        # Financial math tools
│
├── models/
│   └── schemas.py           # Pydantic models, enums, dataclasses
│
└── data/
    └── findings.json        # Latest research output (auto-generated)
```

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+ (for frontend)

### Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

Create a `.env` file:

```
OPENROUTER_API_KEY=your-openrouter-key
TAVILY_API_KEY=your-tavily-key
OPENAI_API_KEY=your-openai-key          # only for embeddings
MODEL_SMART=openai/gpt-4o               # or anthropic/claude-sonnet-4, google/gemini-2.5-pro, etc.
MODEL_FAST=openai/gpt-4o-mini           # or anthropic/claude-haiku-3.5, etc.
```

LLM calls go through [OpenRouter](https://openrouter.ai), so you can swap models without code changes. Embeddings still use OpenAI directly (OpenRouter doesn't support embedding models).

### Run Backend

```bash
uvicorn main:app --reload
```

Server starts at `http://localhost:8000`

### Frontend

```bash
cd deep-research-agent-frontend
npm install
npm run dev
```

Frontend starts at `http://localhost:3000`

## API Endpoints

### 1. Submit Query

```
POST /research/
Body: {"query": "Analyze Infosys Q3 results and growth outlook"}
```

Returns a structured research plan for user review.

### 2. Approve Plan

```
POST /research/{session_id}/approve
Body: {"approved": true}
```

Starts the research agent in background.

### 3. Check Status (poll this)

```
GET /research/{session_id}/status
```

Returns current step, live insights, activity stream, and covered steps.

### 4. Stop Research

```
POST /research/{session_id}/stop
```

Force stops research and generates report from whatever has been collected.

### 5. Get Report

```
GET /research/{session_id}/report
```

Returns the final markdown report.

### 6. Upload Document (for RAG)

```
POST /documents/upload/{session_id}
Form: file (PDF), company (optional), doc_type (optional), year (optional)
```

Uploads a PDF into the session's ChromaDB collection for the agent to search.

## Example Queries

- `"Analyze Infosys Q3 FY26 financial performance"`
- `"Compare TCS, Infosys, and Wipro financials"`
- `"What are the emerging trends in pharmaceutical R&D spending?"`
- `"Deep dive into Sun Pharma financials and drug pipeline"`
- `"Impact of recent FDA regulations on Indian pharma companies"`

## Tech Stack

- **Backend**: FastAPI, Python 3.12
- **Agent Framework**: LangGraph (state machine), LangChain (tool calling)
- **LLM Router**: OpenRouter (swap models via config — GPT-4o, Claude, Gemini, etc.)
- **Web Search**: Tavily API
- **Financial Data**: yfinance + Yahoo Finance search API
- **RAG**: ChromaDB + OpenAI embeddings
- **Web Scraping**: httpx + trafilatura + pypdf
- **Frontend**: Next.js, React, Tailwind CSS
