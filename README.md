# Financial Deep Research Agent

AI-powered deep research agent for IT and Pharma sectors. Conducts multi-step research workflows with web search, financial data analysis, and report generation.

## Architecture

```
User Query → Classifier → Planner → [User Approves] → Research Agent → Report Agent → Markdown Report
```

### Components

- **Classifier** — LLM-based sector detection (IT/Pharma/decline/clarify)
- **Planner** — Generates structured research plan with steps, entities, tools
- **Research Agent** — LangGraph-based iterative loop:
  - Search → Extract Insights → Check Relevance → Financial Deep Dive → Refine Query → Loop
- **Report Agent** — Synthesizes all findings into institutional-grade markdown report

### Tools

| Tool | Source | Purpose |
|------|--------|---------|
| `web_search` | Tavily | Web search for news, reports, analyst views |
| `get_company_overview` | yfinance | Price, P/E, market cap, margins |
| `get_financials` | yfinance | Quarterly revenue, net income, EBITDA |
| `get_balance_sheet` | yfinance | Assets, liabilities, equity, debt |
| `get_stock_price_history` | yfinance | Historical price trends |
| `resolve_ticker` | Yahoo Search | Company name → ticker symbol |
| `calculate_growth_rate` | Python | Revenue/profit growth % |
| `calculate_margin` | Python | Profit margin, EBITDA margin |
| `calculate_cagr` | Python | Compound annual growth rate |
| `compare_metrics` | Python | Side-by-side company comparison |
| `store_insight` | Memory | Save research findings |
| `get_summary` | Memory | Get condensed findings for context |

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+ (for frontend)

### Backend

```bash
cd deep-research-agent

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
OPENAI_API_KEY=sk-your-key
TAVILY_API_KEY=tvly-your-key
OPENAI_MODEL_SMART=gpt-4o
OPENAI_MODEL_FAST=gpt-4o-mini
```

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

Returns research plan for user review.

### 2. Approve Plan

```
POST /research/{session_id}/approve
Body: {"approved": true}
```

Starts the research agent in background.

### 3. Check Status

```
GET /research/{session_id}/status
```

Returns current step, insights gathered, tools used.

### 4. Get Report

```
GET /research/{session_id}/report
```

Returns final markdown report.

## Example Queries

- `"Analyze Infosys Q3 FY26 financial performance, compare with TCS and Wipro"`
- `"What are the emerging trends in pharmaceutical R&D spending?"`
- `"Deep dive into Sun Pharma financials and pipeline"`
- `"Compare the financial health of major IT services companies"`

## Tech Stack

- **Backend**: FastAPI, LangGraph, LangChain, OpenAI
- **LLM**: GPT-4o (planning, synthesis), GPT-4o-mini (classification, routing)
- **Search**: Tavily
- **Financial Data**: yfinance
- **Frontend**: Next.js, React, Tailwind CSS
