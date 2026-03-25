import importlib
import inspect
import logging
import pkgutil
from pathlib import Path

from models.schemas import Sector
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    def __init__(self):
        self._agents: dict[Sector, BaseAgent] = {}

    def register(self, agent: BaseAgent):
        self._agents[agent.sector] = agent
        logger.info(f"Registered agent: {agent.sector.value} — {agent.sector_description}")

    def get(self, sector: Sector) -> BaseAgent:
        if sector not in self._agents:
            raise ValueError(
                f"No agent registered for sector: {sector}. "
                f"Available: {[s.value for s in self._agents.keys()]}"
            )
        return self._agents[sector]

    @property
    def available_sectors(self) -> list[Sector]:
        return list(self._agents.keys())

    def sector_descriptions(self) -> dict[str, str]:
        """Return {sector_value: description} for all registered agents."""
        return {
            agent.sector.value: agent.sector_description
            for agent in self._agents.values()
        }

    def auto_discover(self):
        """Scan the agents/ package and register all BaseAgent subclasses."""
        agents_dir = Path(__file__).parent
        for module_info in pkgutil.iter_modules([str(agents_dir)]):
            if module_info.name in ("base_agent", "registry", "__init__"):
                continue
            try:
                module = importlib.import_module(f"agents.{module_info.name}")
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseAgent) and obj is not BaseAgent:
                        self.register(obj())
            except Exception as e:
                logger.warning(f"Could not load agent from agents/{module_info.name}: {e}")
