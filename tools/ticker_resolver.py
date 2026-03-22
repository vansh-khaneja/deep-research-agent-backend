import logging
import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"


@tool
def resolve_ticker(company_name: str, exchange: str = "NSE") -> str:
    """Resolve a company name to its Yahoo Finance ticker symbol.
    Prefers NSE (.NS) tickers for Indian stocks. Returns 'NOT_FOUND' if unresolved.
    Use this before calling any financial data tool."""

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(YAHOO_SEARCH_URL, params={"q": company_name}, headers=headers, timeout=5)
        data = res.json()
        quotes = data.get("quotes", [])

        if not quotes:
            return "NOT_FOUND"

        # Prefer NSE ticker if exchange is NSE
        if exchange == "NSE":
            for q in quotes:
                if q.get("exchDisp") == "NSE" and q.get("quoteType") == "EQUITY":
                    logger.info(f"Resolved '{company_name}' → {q['symbol']} (NSE)")
                    return q["symbol"]

        # Fallback to first equity result
        for q in quotes:
            if q.get("quoteType") == "EQUITY":
                logger.info(f"Resolved '{company_name}' → {q['symbol']}")
                return q["symbol"]

    except Exception as e:
        logger.error(f"Ticker resolution failed for '{company_name}': {e}")

    return "NOT_FOUND"
