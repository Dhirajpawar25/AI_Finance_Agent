"""Tool registry — functions the assistant can use to gather financial intel.

These tools are called by the assistant orchestration layer based on intent
detection, then the results are synthesized by the LLM into natural answers.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.services import market_data, sec_data

logger = logging.getLogger(__name__)


def _run_sync(fn, *args, **kwargs):
    """Run a blocking call in a thread to avoid stalling the event loop."""
    return asyncio.to_thread(fn, *args, **kwargs)


async def run_tool(name: str, **kwargs: Any) -> Any:
    """Execute a tool by name with resolved kwargs. Returns raw data."""
    if name == "get_price":
        return await _run_sync(market_data.get_price, kwargs.get("ticker", ""))
    if name == "get_company_profile":
        return await _run_sync(market_data.get_company_profile, kwargs.get("ticker", ""))
    if name == "get_news":
        ticker = kwargs.get("ticker", "")
        days_back = kwargs.get("days_back", 2)
        if ticker:
            items = await _run_sync(market_data.get_finnhub_news, ticker)
            if not items:
                items = await _run_sync(market_data.get_news, ticker, days_back)
            return items
        return await _run_sync(market_data.get_top_market_news, days_back)
    if name == "get_filings":
        return await sec_data.get_filings(
            kwargs.get("ticker", ""),
            form_types=kwargs.get("form_types"),
            limit=kwargs.get("limit", 8),
        )
    if name == "get_market_overview":
        return await _run_sync(market_data.get_market_overview)
    if name == "get_top_movers":
        return await _run_sync(market_data.get_top_movers, kwargs.get("limit", 5))
    if name == "get_regional_market_data":
        return await _run_sync(market_data.get_regional_market_data, kwargs.get("region", "us"))
    if name == "get_earnings_calendar":
        return await _run_sync(market_data.get_earnings_calendar, kwargs.get("days", 7))
    if name == "get_historical":
        return await _run_sync(market_data.get_historical, kwargs.get("ticker", ""), kwargs.get("period", "3mo"))
    if name == "search_company":
        return await sec_data.get_sec_company_search(kwargs.get("query", ""))
    if name == "get_company_facts":
        return await sec_data.get_company_contacts(kwargs.get("ticker", ""))
    logger.warning("Unknown tool requested: %s", name)
    return {"error": f"Unknown tool '{name}'."}


TOOL_CATALOG: dict[str, dict[str, Any]] = {
    "get_price": {
        "description": "Get current stock price and quote data for a ticker.",
        "params": {"ticker": "str"},
    },
    "get_company_profile": {
        "description": "Get company profile, fundamentals, and analyst data for a ticker.",
        "params": {"ticker": "str"},
    },
    "get_news": {
        "description": "Get latest news headlines for a ticker (or top market news if empty).",
        "params": {"ticker": "str (optional)", "days_back": "int (optional)"},
    },
    "get_filings": {
        "description": "Get recent SEC filings for a ticker (10-K, 10-Q, 8-K, Form 4).",
        "params": {"ticker": "str", "form_types": "list[str] (optional)"},
    },
    "get_market_overview": {
        "description": "Get index levels and notable mover stocks.",
        "params": {},
    },
    "get_top_movers": {
        "description": "Get real top gainers and losers across major markets.",
        "params": {"limit": "int (optional)"},
    },
    "get_regional_market_data": {
        "description": "Get regional market snapshot (indices, movers, news) for us/india/europe/japan/china.",
        "params": {"region": "str (e.g. 'india', 'europe', 'japan')"},
    },
    "get_earnings_calendar": {
        "description": "Get upcoming earnings dates for major companies.",
        "params": {"days": "int (optional)"},
    },
    "get_historical": {
        "description": "Get historical price series for a ticker (trend analysis).",
        "params": {"ticker": "str", "period": "str (optional, e.g. 3mo)"},
    },
    "search_company": {
        "description": "Search for a company by name to resolve its ticker.",
        "params": {"query": "str"},
    },
    "get_company_facts": {
        "description": "Get the latest reported financial figures from SEC company facts (revenue, net income, etc.).",
        "params": {"ticker": "str"},
    },
    "gmail_search": {
        "description": "Search the user's Gmail for messages (requires connected Gmail).",
        "params": {"query": "str", "max_results": "int (optional)"},
    },
    "calendar_upcoming": {
        "description": "Get the user's upcoming Google Calendar events (requires connected Calendar).",
        "params": {"days": "int (optional)"},
    },
    "calendar_create_event": {
        "description": "Create a meeting/reminder on the user's Google Calendar.",
        "params": {"summary": "str", "start_dt": "str ISO-8601", "end_dt": "str ISO-8601"},
    },
    "sheets_analyze": {
        "description": "Analyze a Google Sheet for structure and anomalies (requires connected Sheets).",
        "params": {"file_query": "str (name of the sheet)", "sheet_name": "str (optional)"},
    },
    "drive_find_by_name": {
        "description": "Search the user's Google Drive for documents.",
        "params": {"query": "str"},
    },
    "drive_read_file_content": {
        "description": "Read text content of a Drive file (Docs/Sheets/PDF).",
        "params": {"file_id": "str", "mime_type": "str (optional)"},
    },
}


def resolve_ticker(entity: str) -> str | None:
    """Resolve a natural-language entity (e.g. 'Apple', 'Nvidia') to a ticker."""
    entity = entity.strip()
    if not entity:
        return None
    upper = entity.upper()
    # Already a valid ticker pattern and likely a ticker
    if len(upper) <= 5 and upper.replace(".", "").isalnum():
        guessed = market_data.guess_ticker(entity.lower())
        return guessed or upper
    guessed = market_data.guess_ticker(entity)
    return guessed or upper