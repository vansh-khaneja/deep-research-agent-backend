import logging
import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _normalize_key(key: str) -> str:
    """Convert Yahoo's raw keys to clean snake_case."""
    mapping = {
        "Total Revenue": "revenue",
        "Net Income": "net_income",
        "EBITDA": "ebitda",
        "Gross Profit": "gross_profit",
        "Operating Income": "operating_income",
        "Total Assets": "total_assets",
        "Total Liabilities Net Minority Interest": "total_liabilities",
        "Stockholders Equity": "stockholders_equity",
        "Cash And Cash Equivalents": "cash",
        "Total Debt": "total_debt",
        "Operating Cash Flow": "operating_cashflow",
        "Free Cash Flow": "free_cashflow",
        "Capital Expenditure": "capex",
    }
    return mapping.get(key, key.lower().replace(" ", "_"))


def _safe_int(val) -> int | None:
    if val is None or val != val:  # NaN check
        return None
    return int(val)


@tool
def get_company_overview(ticker: str) -> dict:
    """Get company profile — price, market cap, P/E, sector, 52-week range.
    Use .NS suffix for Indian stocks (e.g. INFY.NS)."""
    try:
        stock = yf.Ticker(ticker)
        fi = stock.fast_info
        info = stock.info or {}
        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "current_price": round(fi.last_price, 2) if fi.last_price else None,
            "market_cap": fi.market_cap,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": fi.year_high,
            "52w_low": fi.year_low,
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
        }
    except Exception as e:
        logger.error(f"Overview failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


@tool
def get_financials(ticker: str) -> dict:
    """Get last 4 quarters income statement — revenue, net_income, ebitda, gross_profit, operating_income.
    All keys are normalized to snake_case."""
    try:
        stock = yf.Ticker(ticker)
        qf = stock.quarterly_financials
        if qf is None or qf.empty:
            return {"ticker": ticker, "error": "No financial data"}

        metrics = ["Total Revenue", "Net Income", "EBITDA", "Gross Profit", "Operating Income"]
        quarters = []
        for col in qf.columns[:4]:
            quarter = {"period": str(col.date())}
            for m in metrics:
                if m in qf.index:
                    quarter[_normalize_key(m)] = _safe_int(qf.loc[m, col])
            quarters.append(quarter)

        return {"ticker": ticker, "quarterly": quarters}
    except Exception as e:
        logger.error(f"Financials failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


@tool
def get_balance_sheet(ticker: str) -> dict:
    """Get latest balance sheet — total_assets, total_liabilities, stockholders_equity, cash, total_debt.
    All keys normalized."""
    try:
        stock = yf.Ticker(ticker)
        bs = stock.quarterly_balance_sheet
        if bs is None or bs.empty:
            return {"ticker": ticker, "error": "No balance sheet data"}

        metrics = ["Total Assets", "Total Liabilities Net Minority Interest",
                    "Stockholders Equity", "Cash And Cash Equivalents", "Total Debt"]
        col = bs.columns[0]
        result = {"period": str(col.date())}
        for m in metrics:
            if m in bs.index:
                result[_normalize_key(m)] = _safe_int(bs.loc[m, col])

        return {"ticker": ticker, "balance_sheet": result}
    except Exception as e:
        logger.error(f"Balance sheet failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


@tool
def get_stock_price_history(ticker: str, period: str = "6mo") -> dict:
    """Get historical stock prices. Period: 1mo, 3mo, 6mo, 1y, 2y, 5y.
    Returns last 10 data points, period high/low."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return {"ticker": ticker, "error": "No price history"}

        prices = [
            {"date": str(idx.date()), "close": round(row["Close"], 2), "volume": int(row["Volume"])}
            for idx, row in hist.iterrows()
        ]
        return {
            "ticker": ticker,
            "period": period,
            "current": prices[-1]["close"] if prices else None,
            "period_high": round(hist["Close"].max(), 2),
            "period_low": round(hist["Close"].min(), 2),
            "last_10": prices[-10:],
        }
    except Exception as e:
        logger.error(f"Price history failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}
