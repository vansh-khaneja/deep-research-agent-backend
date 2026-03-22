from models.schemas import Sector
from agents.base_agent import BaseAgent


class ITAgent(BaseAgent):

    @property
    def sector(self) -> Sector:
        return Sector.IT

    @property
    def planning_context(self) -> str:
        return """You are planning research for the IT / Technology services sector.
Key areas to investigate:
- Revenue breakdown by geography and service line (consulting, outsourcing, digital, cloud)
- Deal pipeline and large deal wins (TCV - Total Contract Value)
- Employee metrics: headcount, attrition rate, utilization rate
- Digital revenue as % of total revenue
- Client concentration and top client metrics
- AI/automation adoption and investments
- Margins: EBITDA, operating margin, net margin trends
- Competitive positioning vs global peers (Accenture, IBM, Cognizant)
- Currency impact and hedging strategy
- Guidance and management commentary

Major Indian IT companies and their tickers:
- TCS (Tata Consultancy Services): TCS.NS
- Infosys: INFY.NS
- Wipro: WIPRO.NS
- HCL Technologies: HCLTECH.NS
- Tech Mahindra: TECHM.NS
- LTIMindtree: LTIM.NS

Global IT companies:
- Accenture: ACN
- IBM: IBM
- Cognizant: CTSH"""

    @property
    def research_context(self) -> str:
        return """Focus on IT services and technology sector. Include queries about:
digital transformation deals, cloud migration revenue, AI/ML service offerings,
quarterly financial results, management guidance, attrition trends,
deal wins and pipeline, sector outlook, regulatory changes affecting IT outsourcing."""

    @property
    def report_template(self) -> str:
        return """Structure the report as follows:
# {Company/Sector} - IT Sector Research Report

## Executive Summary
Brief overview of key findings and conclusions.

## Company/Sector Overview
Background, market position, recent developments.

## Financial Analysis
Revenue trends, profitability, growth rates, key ratios.
Use actual numbers from the data provided - do NOT estimate or fabricate.

## Business Segments & Strategy
Service line breakdown, digital/cloud initiatives, AI strategy.

## Competitive Landscape
Position vs peers, market share, differentiators.

## Key Risks & Challenges
Macro headwinds, attrition, currency, client concentration.

## Outlook & Recommendations
Forward-looking analysis based on guidance and trends.

## Sources
List all sources used."""

    @property
    def key_metrics(self) -> list[str]:
        return [
            "Revenue Growth (YoY/QoQ)",
            "EBITDA Margin",
            "Operating Margin",
            "Net Profit Margin",
            "Attrition Rate",
            "Deal TCV",
            "Digital Revenue %",
            "Employee Count",
            "P/E Ratio",
            "Return on Equity",
        ]
