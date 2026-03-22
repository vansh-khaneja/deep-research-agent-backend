from models.schemas import Sector
from agents.base_agent import BaseAgent


class AgentRegistry:
    def __init__(self):
        self._agents: dict[Sector, BaseAgent] = {}

    def register(self, agent: BaseAgent):
        self._agents[agent.sector] = agent

    def get(self, sector: Sector) -> BaseAgent:
        if sector not in self._agents:
            raise ValueError(
                f"No agent registered for sector: {sector}. "
                f"Available: {list(self._agents.keys())}"
            )
        return self._agents[sector]

    @property
    def available_sectors(self) -> list[Sector]:
        return list(self._agents.keys())
