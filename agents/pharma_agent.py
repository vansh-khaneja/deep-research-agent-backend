from models.schemas import Sector
from agents.base_agent import BaseAgent


class PharmaAgent(BaseAgent):

    @property
    def sector(self) -> Sector:
        return Sector.PHARMA

    @property
    def planning_context(self) -> str:
        return """You are planning research for the Pharmaceutical / Healthcare sector.
Key areas to investigate:
- R&D pipeline: drugs in Phase I/II/III, recent approvals, ANDA filings
- Revenue by geography (US, India, Europe, ROW) and by therapy area
- Biosimilar portfolio and market opportunity
- API (Active Pharmaceutical Ingredient) manufacturing capabilities
- Patent cliff exposure and genericization impact
- Regulatory actions: FDA approvals, warning letters, import alerts
- EBITDA margins, R&D spending as % of revenue
- M&A activity and strategic partnerships
- Specialty vs generic portfolio mix
- USFDA inspection outcomes

Major Indian Pharma companies and their tickers:
- Sun Pharma: SUNPHARMA.NS
- Dr. Reddy's: DRREDDY.NS
- Cipla: CIPLA.NS
- Aurobindo Pharma: AUROPHARMA.NS
- Lupin: LUPIN.NS
- Divi's Laboratories: DIVISLAB.NS
- Biocon: BIOCON.NS

Global Pharma companies:
- Pfizer: PFE
- Johnson & Johnson: JNJ
- Novartis: NVS
- Roche: RHHBY"""

    @property
    def research_context(self) -> str:
        return """Focus on pharmaceutical and healthcare sector. Include queries about:
drug pipeline approvals, R&D spending, biosimilar market, FDA regulatory actions,
patent expirations, generic drug launches, specialty pharma strategy,
quarterly financial results, API manufacturing, therapeutic area trends."""

    @property
    def report_template(self) -> str:
        return """Structure the report as follows:
# {Company/Sector} - Pharma Sector Research Report

## Executive Summary
Brief overview of key findings and conclusions.

## Company/Sector Overview
Background, therapeutic focus, market position.

## Financial Analysis
Revenue trends, profitability, R&D spending, key ratios.
Use actual numbers from the data provided - do NOT estimate or fabricate.

## R&D Pipeline & Products
Key drugs, pipeline status, recent approvals, ANDA filings.

## Regulatory Landscape
FDA actions, approvals, compliance status.

## Market Dynamics
Biosimilar opportunity, generic competition, patent cliffs.

## Key Risks & Challenges
Regulatory risks, pricing pressure, pipeline failures, API dependency.

## Outlook & Recommendations
Forward-looking analysis based on pipeline and market trends.

## Sources
List all sources used."""

    @property
    def key_metrics(self) -> list[str]:
        return [
            "Revenue Growth (YoY/QoQ)",
            "EBITDA Margin",
            "R&D Spending (% of Revenue)",
            "US Revenue %",
            "ANDA Filings/Approvals",
            "Biosimilar Revenue",
            "Net Profit Margin",
            "Debt to Equity",
            "P/E Ratio",
            "Return on Equity",
        ]
