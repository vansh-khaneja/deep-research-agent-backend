import logging
from pydantic import BaseModel
from tools.llm import LLMClient
from models.schemas import Sector, ResearchPlan, ResearchStep
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# --- Sector Classification ---

class ClassificationResult(BaseModel):
    is_financial: bool
    sector: str = ""
    confidence: float = 0.0


class SectorClassifier:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def classify(self, query: str) -> tuple[str, ClassificationResult]:
        """Returns (route, classification). Route: 'it', 'pharma', 'decline', or 'clarify'."""

        system = """You are a financial research query classifier.

Classify the query into:
1. "it" — IT / Technology services sector
2. "pharma" — Pharmaceutical / Healthcare sector
3. "not_financial" — Not a financial query at all
4. "unclear" — Financial but sector is ambiguous

Respond with JSON matching the schema."""

        result = await self.llm.complete(
            system=system,
            user=f"Classify this query: {query}",
            response_model=ClassificationResult,
        )

        if not result.is_financial or result.sector == "not_financial":
            route = "decline"
        elif result.sector in ("it", "pharma"):
            route = result.sector
        elif result.sector == "unclear":
            route = "clarify"
        else:
            route = "decline"

        logger.info(f"Classified: route={route}, confidence={result.confidence}")
        return route, result


# --- Research Planner ---

PLANNER_PROMPT = """You are a senior financial research planner.

Sector: {sector}
Sector Context:
{agent_context}

Your job is to convert the user's query into a structured deep research plan.

Tasks:
1. Identify the research goal (one clear sentence)

2. Classify query type:
   - company_analysis
   - sector_analysis
   - comparative_study
   - trend_analysis
   - regulation_impact
   - investment_outlook

3. Estimate research depth based on query complexity:
   - shallow (3-5 steps) — simple factual query about one metric or one company's single aspect
   - medium (6-8 steps) — single company deep dive or focused sector topic
   - deep (12-15 steps) — comparative multi-company analysis, sector-wide trends, or queries mentioning multiple dimensions

IMPORTANT: If the query mentions multiple companies, asks for comparison, covers multiple themes (financials + strategy + outlook), or asks for comprehensive/deep analysis — you MUST classify as "deep" and generate 12-15 steps. Do NOT default to 10.

4. Generate ordered research steps. Each step has:
   - step_number: sequential integer starting at 1
   - description: what this step investigates
   - query_template: the actual web search query to execute
   - tools: which tools to use from ["web_search", "financial_api", "rag_docs"]

5. Extract important entities:
   - company names, technologies, themes, regulations mentioned or implied

6. Decide tools needed from: ["web_search", "financial_api", "rag_docs", "calculation_engine"]

7. Define expected report structure (list of section headings)

Key metrics for this sector: {metrics}

Return STRICT JSON matching the schema."""


class PlannerStepOutput(BaseModel):
    step_number: int
    description: str
    query_template: str
    tools: list[str] = ["web_search"]


class PlannerOutput(BaseModel):
    research_goal: str
    query_type: str
    research_depth: str
    steps: list[PlannerStepOutput]
    entities: list[str] = []
    tools_required: list[str] = []
    report_structure: list[str] = []


class ResearchPlanner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def create_plan(self, query: str, sector: Sector, agent: BaseAgent) -> ResearchPlan:

        prompt = PLANNER_PROMPT.format(
            sector=sector.value,
            agent_context=agent.planning_context,
            metrics=", ".join(agent.key_metrics),
        )

        result = await self.llm.complete(
            system=prompt,
            user=f"Query: {query}",
            response_model=PlannerOutput,
        )

        # Build steps
        steps = [
            ResearchStep(
                step_number=s.step_number,
                description=s.description,
                query_template=s.query_template,
                tools=s.tools,
            )
            for s in result.steps
        ]

        # Validate
        self._validate(result, steps)

        plan = ResearchPlan(
            sector=sector,
            research_goal=result.research_goal,
            query_type=result.query_type,
            research_depth=result.research_depth,
            steps=steps,
            entities=result.entities,
            tools_required=result.tools_required,
            report_structure=result.report_structure,
        )

        logger.info(
            f"Plan: type={result.query_type}, depth={result.research_depth}, "
            f"steps={len(steps)}, entities={result.entities}"
        )
        return plan

    def _validate(self, result: PlannerOutput, steps: list[ResearchStep]):
        errors = []

        # Step count checks
        if len(steps) < 3:
            errors.append(f"Too few steps ({len(steps)}), need at least 3")

        if result.research_depth == "deep" and len(steps) < 12:
            errors.append(f"Deep research needs 12+ steps, got {len(steps)}")

        if result.research_depth == "medium" and len(steps) < 6:
            errors.append(f"Medium research needs 6+ steps, got {len(steps)}")

        # Check for empty/invalid fields
        if not result.research_goal or len(result.research_goal) < 10:
            errors.append("Research goal is missing or too short")

        valid_types = ("company_analysis", "sector_analysis", "comparative_study",
                       "trend_analysis", "regulation_impact", "investment_outlook")
        if result.query_type not in valid_types:
            logger.warning(f"Unexpected query_type: {result.query_type}, defaulting to sector_analysis")
            result.query_type = "sector_analysis"

        valid_depths = ("shallow", "medium", "deep")
        if result.research_depth not in valid_depths:
            logger.warning(f"Unexpected research_depth: {result.research_depth}, defaulting to medium")
            result.research_depth = "medium"

        # Check each step
        for s in steps:
            if not s.query_template or len(s.query_template) < 5:
                errors.append(f"Step {s.step_number}: empty or too short query_template")
            if not s.description:
                errors.append(f"Step {s.step_number}: missing description")

        # Check for duplicate queries
        queries = [s.query_template.lower().strip() for s in steps]
        if len(queries) != len(set(queries)):
            errors.append("Duplicate query_templates found across steps")

        if not result.report_structure or len(result.report_structure) < 3:
            errors.append("Report structure needs at least 3 sections")

        if errors:
            logger.warning(f"Plan validation failed: {errors}")
            raise ValueError(f"Invalid research plan: {'; '.join(errors)}")
