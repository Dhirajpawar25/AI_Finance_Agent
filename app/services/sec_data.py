"""SEC EDGAR data retrieval — company filings, financial statements, insider transactions."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

EDGAR_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"}

# Common CIK lookups to avoid re-resolution
_KNOWN_CIKS = {
    "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044", "GOOG": "0001652044",
    "AMZN": "0001018724", "NVDA": "0001045810", "META": "0001326801", "TSLA": "0001318605",
    "NFLX": "0001065280", "JPM": "0000019617", "V": "0001403161", "MA": "0001141391",
    "AMD": "0000002488", "INTC": "0000050863", "AVGO": "0001730168", "CRM": "0001108524",
    "ORCL": "0001341439", "IBM": "0000051143", "DIS": "0001744489", "KO": "0000021344",
    "PEP": "0000077476", "WMT": "0000104169", "COST": "0000909832", "MCD": "0000063908",
    "SBUX": "0000829224", "NKE": "0000320187", "BA": "0000012927", "CAT": "0000018230",
    "GE": "0000040545", "F": "0000037996", "GM": "0001467858", "UNH": "0000731766",
    "PFE": "0000078003", "JNJ": "0000200406", "MRK": "0000310158", "LLY": "0000059478",
    "GS": "0000886982", "BA": "0000012927", "PYPL": "0001633917", "UBER": "0001543151",
    "ABNB": "0001559720", "COIN": "0001679788", "SHOP": "0001594805", "SNOW": "0001640147",
    "PLTR": "0001321655", "CRWD": "0001535527", "DDOG": "0001561550", "NET": "0001474433",
    "ARM": "0001973239", "TSM": "0001046179", "ASML": "0000937966", "INTU": "0000896878",
    "NOW": "0001373715", "ADBE": "0000796343", "QCOM": "0000804328", "TXN": "0000097476",
}


async def _get_json(url: str) -> dict[str, Any] | list[Any] | None:
    """Async GET with proper SEC headers."""
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("SEC request failed for %s: %s", url, exc)
        return None


def _lookup_cik(ticker: str) -> str | None:
    """Resolve a ticker to a CIK using local cache or EDGAR company_tickers.json."""
    ticker = ticker.upper()
    if ticker in _KNOWN_CIKS:
        return _KNOWN_CIKS[ticker]
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        info = t.info or {}
        cik = info.get("cik")
        if cik:
            return str(cik).zfill(10)
    except Exception as exc:  # noqa: BLE001
        logger.debug("yfinance CIK lookup failed: %s", exc)

    # Fallback: search EDGAR company_tickers
    try:
        data = httpx.get(f"{EDGAR_BASE}/files/company_tickers.json", headers=HEADERS, timeout=20).json()
        for _k, entry in data.items():
            if entry.get("ticker", "").upper() == ticker:
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:  # noqa: BLE001
        logger.debug("EDGAR company_tickers lookup failed: %s", exc)
    return None


async def get_filings(ticker: str, form_types: list[str] | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """Recent SEC filings for a company. form_types e.g. ['10-K','10-Q','8-K','4']."""
    cik = _lookup_cik(ticker)
    if not cik:
        return []
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    data = await _get_json(url)
    if not isinstance(data, dict):
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filed = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    results = []
    for i in range(len(forms)):
        form = forms[i]
        if form_types and form not in form_types:
            continue
        results.append(
            {
                "ticker": ticker,
                "company": data.get("name", ticker),
                "form": form,
                "filing_date": filed[i] if i < len(filed) else None,
                "accession": accessions[i] if i < len(accessions) else None,
                "document": docs[i] if i < len(docs) else None,
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accessions[i].replace('-', '')}/"
                    f"{docs[i] if i < len(docs) else 'index.html'}"
                    if i < len(accessions) and accessions[i]
                    else None
                ),
            }
        )
        if len(results) >= limit:
            break
    return results


async def get_latest_10k(ticker: str) -> dict[str, Any] | None:
    """Fetch the most recent 10-K filing metadata/URL for a company."""
    filings = await get_filings(ticker, form_types=["10-K"], limit=1)
    return filings[0] if filings else None


async def get_latest_8k(ticker: str) -> dict[str, Any] | None:
    """Fetch the most recent 8-K (material event) filing for a company."""
    filings = await get_filings(ticker, form_types=["8-K"], limit=1)
    return filings[0] if filings else None


async def get_insider_transactions(ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    """Recent Form 4 insider transactions for a company."""
    filings = await get_filings(ticker, form_types=["4"], limit=limit)
    return filings


async def get_sec_company_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Full-text search on EDGAR for companies using the companies index."""
    try:
        data = httpx.get(f"{EDGAR_BASE}/files/company_tickers.json", headers=HEADERS, timeout=20).json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("EDGAR company search failed: %s", exc)
        return []

    query_lower = query.lower()
    results = []
    for _key, entry in data.items():
        name = entry.get("title", "")
        ticker = entry.get("ticker", "")
        if query_lower in name.lower() or query_lower in ticker.lower():
            results.append(
                {
                    "ticker": ticker,
                    "company": name,
                    "cik": str(entry.get("cik_str", "")).zfill(10),
                }
            )
        if len(results) >= limit:
            break
    return results


async def get_company_contacts(ticker: str) -> dict[str, Any] | None:
    """Company fundamentals overview from SEC companyfacts."""
    cik = _lookup_cik(ticker)
    if not cik:
        return None
    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    data = await _get_json(url)
    if not isinstance(data, dict):
        return None
    entity = data.get("entityName", ticker)
    facts = data.get("facts", {}).get("us-gaap", {})

    overview: dict[str, Any] = {"ticker": ticker, "entity": entity}
    metrics = {
        "revenue": "Revenues", "net income": "NetIncomeLoss",
        "assets": "Assets", "liabilities": "Liabilities",
        "shareholders equity": "StockholdersEquity",
        "cash": "CashAndCashEquivalentsAtCarryingValue",
        "total debt": "LongTermDebt",
        "diluted eps": "EarningsPerShareDiluted",
        "operating margin": "OperatingIncomeLoss",
        "r&d": "ResearchAndDevelopmentExpense",
        "free cash flow": "PaymentsToAcquirePropertyPlantAndEquipment",
    }
    for label, usgaap_key in metrics.items():
        node = facts.get(usgaap_key)
        if not node or "units" not in node:
            continue
        # USD is the canonical unit
        usd = node["units"].get("USD")
        if not usd:
            # try first available unit
            first_unit = next(iter(node["units"].values()), None)
            usd = first_unit
        if usd:
            last = usd[-1]
            overview[label] = {"value": last.get("val"), "end_date": last.get("end"), "filed": last.get("filed")}
    return overview


async def get_recent_filings_for_watchlist(watchlist: list[str], limit_per_company: int = 3) -> list[dict[str, Any]]:
    """Aggregate recent 8-K/10-K/10-Q filings for a list of tickers."""
    all_filings: list[dict[str, Any]] = []
    for ticker in watchlist[:10]:
        filings = await get_filings(ticker, form_types=["8-K", "10-K", "10-Q"], limit=limit_per_company)
        all_filings.extend(filings)
    # sort by filing date descending
    all_filings.sort(key=lambda f: f.get("filing_date") or "", reverse=True)
    return all_filings[:15]