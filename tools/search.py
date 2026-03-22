import logging
from langchain_core.tools import tool
from tavily import TavilyClient
from config import settings

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> list[dict]:
    """Search the web for financial research information using Tavily.
    Returns a list of search results with title, url, content, and relevance score.
    Use this tool to find latest news, reports, analyst views, and financial data."""

    client = TavilyClient(api_key=settings.tavily_api_key)

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            })

        logger.info(f"Web search: '{query}' → {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Web search failed for '{query}': {e}")
        return []
