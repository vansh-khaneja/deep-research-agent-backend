from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    tavily_api_key: str = ""
    openai_api_key: str = ""                  # used for embeddings, and as LLM fallback if no OpenRouter key
    model_smart: str = "openai/gpt-4o"       # planner, research, report synthesis
    model_fast: str = "openai/gpt-4o-mini"   # classification, critic, routing
    embedding_model: str = "text-embedding-3-small"
    max_research_depth: int = 15
    chromadb_path: str = "./data/chromadb"

    @property
    def llm_api_key(self) -> str:
        """Use OpenRouter if key is set, otherwise fall back to OpenAI direct."""
        return self.openrouter_api_key or self.openai_api_key

    @property
    def llm_base_url(self) -> str | None:
        """Only set base_url when using OpenRouter."""
        return self.openrouter_base_url if self.openrouter_api_key else None

    @property
    def llm_model_smart(self) -> str:
        """Strip 'openai/' prefix when using OpenAI direct."""
        if not self.openrouter_api_key:
            return self.model_smart.split("/", 1)[-1]
        return self.model_smart

    @property
    def llm_model_fast(self) -> str:
        if not self.openrouter_api_key:
            return self.model_fast.split("/", 1)[-1]
        return self.model_fast

    class Config:
        env_file = ".env"


settings = Settings()
