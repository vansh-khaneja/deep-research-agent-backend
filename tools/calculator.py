import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def calculate_growth_rate(current: float, previous: float) -> dict:
    """Calculate percentage growth rate between two values.
    Example: revenue grew from 4500 to 5100 → 13.33%"""
    if previous == 0:
        return {"error": "Cannot calculate: previous value is 0"}
    rate = ((current - previous) / abs(previous)) * 100
    return {"value": round(rate, 2), "formatted": f"{rate:.2f}%"}


@tool
def calculate_cagr(start_value: float, end_value: float, years: int) -> dict:
    """Calculate Compound Annual Growth Rate (CAGR).
    Example: revenue went from 1000 to 1500 over 3 years → CAGR."""
    if start_value <= 0 or years <= 0:
        return {"error": "Cannot calculate: invalid inputs"}
    cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
    return {"value": round(cagr, 2), "formatted": f"{cagr:.2f}%"}


@tool
def calculate_margin(part: float, total: float) -> dict:
    """Calculate margin/ratio as percentage. Example: net income / revenue = profit margin."""
    if total == 0:
        return {"error": "Cannot calculate: total is 0"}
    margin = (part / total) * 100
    return {"value": round(margin, 2), "formatted": f"{margin:.2f}%"}


@tool
def calculate_ratio(numerator: float, denominator: float) -> dict:
    """Calculate a financial ratio. Example: debt / equity = D/E ratio."""
    if denominator == 0:
        return {"error": "Cannot calculate: denominator is 0"}
    ratio = numerator / denominator
    return {"value": round(ratio, 2), "formatted": f"{ratio:.2f}"}


@tool
def compare_metrics(company_a: str, value_a: float, company_b: str, value_b: float, metric_name: str) -> dict:
    """Compare a metric between two companies. Returns which is higher and by how much."""
    diff = abs(value_a - value_b)
    if value_b != 0:
        pct_diff = round((diff / abs(value_b)) * 100, 1)
    else:
        pct_diff = 0.0

    if value_a > value_b:
        higher = company_a
    elif value_b > value_a:
        higher = company_b
    else:
        higher = "equal"

    return {
        "higher": higher,
        "difference": round(diff, 2),
        "pct_difference": pct_diff,
        "formatted": f"{higher} has higher {metric_name} by {pct_diff}% ({value_a} vs {value_b})"
    }
