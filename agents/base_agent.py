from abc import ABC, abstractmethod
from models.schemas import Sector


class BaseAgent(ABC):

    @property
    @abstractmethod
    def sector(self) -> Sector:
        ...

    @property
    @abstractmethod
    def planning_context(self) -> str:
        """Injected into the planner system prompt for sector-specific plan generation."""
        ...

    @property
    @abstractmethod
    def research_context(self) -> str:
        """Injected into the query generator for sector-aware search queries."""
        ...

    @property
    @abstractmethod
    def report_template(self) -> str:
        """Markdown structure for the final report."""
        ...

    @property
    @abstractmethod
    def key_metrics(self) -> list[str]:
        """Financial metrics important for this sector."""
        ...

    @property
    def typical_sources(self) -> list[str]:
        return ["web", "yfinance", "rag"]
