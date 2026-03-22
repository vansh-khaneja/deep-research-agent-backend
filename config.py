from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    tavily_api_key: str = ""
    alpha_vantage_api_key: str = ""
    openai_model_smart: str = "gpt-4o"        # planner, report synthesis
    openai_model_fast: str = "gpt-4o-mini"   # classification, relevance checks, query refinement
    embedding_model: str = "text-embedding-3-small"
    max_research_depth: int = 15
    chromadb_path: str = "./data/chromadb"

    class Config:
        env_file = ".env"


settings = Settings()
