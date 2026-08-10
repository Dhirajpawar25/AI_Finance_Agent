"""Market data retrieval using yfinance (with Finnhub/Alpha Vantage fallbacks)."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import yfinance as yf

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _safe_get(ticker: yf.Ticker, field: str, default: Any = None) -> Any:
    """Fetch a field from yfinance fast_info / info without crashing."""
    try:
        data = getattr(ticker, "info", {}) or {}
        if isinstance(data, dict):
            return data.get(field, default)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to get info field %s: %s", field, exc)
    return default


def _pct_change(prev_close: float, current: float) -> float:
    if not prev_close:
        return 0.0
    return round(((current - prev_close) / prev_close) * 100, 2)


def get_price(ticker: str) -> dict[str, Any]:
    """Return a compact price snapshot for a ticker."""
    ticker = ticker.strip().upper()
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            return {"error": f"No price data found for {ticker}."}

        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last
        change = round(last - prev, 4)
        pct = _pct_change(prev, last)

        name = _safe_get(t, "shortName") or _safe_get(t, "longName") or ticker
        currency = _safe_get(t, "currency") or "USD"
        info = {
            "ticker": ticker,
            "name": name,
            "price": round(last, 2),
            "currency": currency,
            "change": round(change, 2),
            "pct_change": pct,
            "day_high": round(float(hist["High"].iloc[-1]), 2),
            "day_low": round(float(hist["Low"].iloc[-1]), 2),
            "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist else None,
        }

        # Add a few fundamentals when available
        for field in ("marketCap", "trailingPE", "forwardPE", "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"):
            val = _safe_get(t, field)
            if val is not None:
                info[field] = val if isinstance(val, (int, float)) else None
        return info
    except Exception as exc:  # noqa: BLE001
        logger.exception("price lookup failed for %s", ticker)
        return {"error": f"Could not retrieve data for {ticker}. Please check the ticker and try again."}


def get_index_snapshot() -> list[dict[str, Any]]:
    """Snapshot of major market indices (S&P 500, Nasdaq, Dow, NIFTY 50)."""
    indices = {
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq",
        "^DJI": "Dow Jones",
        "^NSEI": "NIFTY 50",
    }
    results = []
    for symbol, name in indices.items():
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="5d")
            if hist.empty or len(hist) < 2:
                continue
            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            results.append(
                {
                    "name": name,
                    "value": round(last, 2),
                    "pct_change": _pct_change(prev, last),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("index %s failed: %s", symbol, exc)
    return results


def get_company_profile(ticker: str) -> dict[str, Any]:
    """Company profile + fundamentals for research requests."""
    ticker = ticker.strip().upper()
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        if not info:
            return {"error": f"No company data found for {ticker}."}

        keys = [
            "longName", "industry", "sector", "marketCap", "trailingPE", "forwardPE",
            "priceToBook", "dividendYield", "beta", "revenueGrowth", "profitMargins",
            "operatingMargins", "returnOnEquity", "returnOnAssets", "debtToEquity",
            "currentRatio", "freeCashflow", "totalRevenue", "grossProfits", "ebitda",
            "targetMeanPrice", "recommendationKey", "numberOfAnalystOpinions",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "longBusinessSummary",
        ]
        profile = {k: info.get(k) for k in keys if info.get(k) is not None}
        profile["ticker"] = ticker
        if profile.get("longBusinessSummary"):
            profile["longBusinessSummary"] = profile["longBusinessSummary"][:1200]
        return profile
    except Exception as exc:  # noqa: BLE001
        logger.exception("profile lookup failed for %s", ticker)
        return {"error": f"Could not retrieve profile for {ticker}."}


def get_historical(ticker: str, period: str = "3mo") -> dict[str, Any]:
    """Historical prices for charting / trend analysis."""
    ticker = ticker.strip().upper()
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            return {"error": f"No historical data for {ticker}."}
        closes = [round(float(v), 2) for v in hist["Close"].tolist()]
        dates = [d.strftime("%Y-%m-%d") for d in hist.index.tolist()]
        start, end = closes[0], closes[-1]
        step = max(1, len(closes) // 30)
        series = [{"date": d, "close": c} for d, c in zip(dates, closes)][::step][:30]
        return {
            "ticker": ticker,
            "period": period,
            "start_date": dates[0],
            "end_date": dates[-1],
            "change_pct": _pct_change(start, end),
            "series": series,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("history lookup failed for %s", ticker)
        return {"error": f"Could not retrieve historical data for {ticker}."}


def guess_ticker(text: str) -> str | None:
    """Best-effort mapping from company name fragments to common tickers."""
    aliases = {
        "apple": "AAPL", "google": "GOOGL", "alphabet": "GOOGL", "microsoft": "MSFT",
        "nvidia": "NVDA", "tesla": "TSLA", "amazon": "AMZN", "meta": "META",
        "facebook": "META", "netflix": "NFLX", "berkshire": "BRK-B", "jpmorgan": "JPM",
        "goldman": "GS", "visa": "V", "mastercard": "MA", "intel": "INTC",
        "amd": "AMD", "qualcomm": "QCOM", "broadcom": "AVGO", "salesforce": "CRM",
        "oracle": "ORCL", "ibm": "IBM", "coca-cola": "KO", "pepsi": "PEP",
        "walmart": "WMT", "costco": "COST", "home depot": "HD", "mcdonalds": "MCD",
        "starbucks": "SBUX", "nike": "NKE", "disney": "DIS", "uber": "UBER",
        "lyft": "LYFT", "airbnb": "ABNB", "paypal": "PYPL", "stripe": "STRIP",
        "square": "SQ", "block": "SQ", "adobe": "ADBE", "zoom": "ZM",
        "shopify": "SHOP", "spotify": "SPOT", "twitter": "TWTR", "x corp": "TWTR",
        "snap": "SNAP", "pinterest": "PINS", "coinbase": "COIN", "robinhood": "HOOD",
        "etsy": "ETSY", "beyond meat": "BYND", "peloton": "PTON", "shopify": "SHOP",
        "pfizer": "PFE", "johnson & johnson": "JNJ", "merck": "MRK", "moderna": "MRNA",
        "bioNTech": "BNTX", "astrazeneca": "AZN", "lilly": "LLY", "eli lilly": "LLY",
        "novo nordisk": "NVO", "unitedhealth": "UNH", "ge healthcare": "GEHC",
        "boeing": "BA", "airbus": "EADSY", "lockheed": "LMT", "ge": "GE",
        "general electric": "GE", "caterpillar": "CAT", "deere": "DE",
        "ford": "F", "gm": "GM", "general motors": "GM", "toyota": "TM",
        "honda": "HMC", "bmw": "BMWYY", "mercedes": "MBGAF", "rivian": "RIVN",
        "lucid": "LCID", "fisker": "FSRN", "nio": "NIO", "xpeng": "XPEV",
        "li auto": "LI", "lithium americas": "LAC", "palantir": "PLTR",
        "snowflake": "SNOW", "datadog": "DDOG", "crowdstrike": "CRWD",
        "cloudflare": "NET", "c3.ai": "AI", "soundhound": "SOUN",
        "arm": "ARM", "arm holdings": "ARM", "asml": "ASML", "tsmc": "TSM",
        "taiwan semiconductor": "TSM", "samsung": "SSNLF", "sony": "SONY",
        "infosys": "INFY", "tcs": "TCS", "tata consultancy": "TCS", "wipro": "WIPRO",
        "hcl": "HCLTECH.NS", "reliance": "RELIANCE.NS", "tata motors": "TATAMOTORS.NS",
        "hdfc": "HDFCBANK.NS", "icici": "ICICIBANK.NS", "sbi": "SBIN.NS",
        "bharti airtel": "BHARTIARTL.NS", "airtel": "BHARTIARTL.NS",
        "infosys": "INFY.NS", "coal india": "COALINDIA.NS",
    }
    lowered = text.lower().strip().rstrip("?.!")
    if lowered in aliases:
        return aliases[lowered]
    # token-by-token progressive fallback
    tokens = lowered.split()
    for size in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:size])
        if candidate in aliases:
            return aliases[candidate]
    return None


def get_earnings_calendar(days: int = 7) -> list[dict[str, Any]]:
    """Upcoming earnings dates for major companies (best-effort)."""
    majors = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "MA",
        "DIS", "NFLX", "CRM", "ORCL", "ADBE", "INTC", "AMD", "QCOM", "AVGO", "PFE",
        "JNJ", "UNH", "KO", "PEP", "WMT", "COST", "MCD", "SBUX", "BA", "CAT",
    ]
    results = []
    for ticker in majors:
        try:
            t = yf.Ticker(ticker)
            cal = t.get_earnings_dates(limit=4)
            if cal is None or cal.empty:
                continue
            for idx in range(len(cal)):
                when = cal.index[idx]
                if isinstance(when, datetime) and when.date() >= date.today() and (when.date() - date.today()).days <= days:
                    results.append({"ticker": ticker, "earnings_date": when.date().isoformat(), "time": "before_market" if "pre" in str(cal.iloc[idx].get("Earnings Date", "")).lower() else "after_market"})
                    break
        except Exception as exc:  # noqa: BLE001
            logger.debug("earnings calendar skip %s: %s", ticker, exc)
    return results


def get_news(ticker: str, days_back: int = 3) -> list[dict[str, Any]]:
    """Recent news headlines for a ticker via yfinance."""
    ticker = ticker.strip().upper()
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        cutoff = datetime.now() - timedelta(days=days_back)
        items = []
        for n in news[:15]:
            ts = n.get("providerPublishTime")
            published = datetime.fromtimestamp(ts) if ts else None
            if published and published < cutoff:
                continue
            items.append({
                "title": n.get("title", ""),
                "publisher": n.get("publisher", ""),
                "link": n.get("link", ""),
                "time": published.isoformat() if published else None,
            })
        return items
    except Exception as exc:  # noqa: BLE001
        logger.debug("news lookup failed for %s: %s", ticker, exc)
        return []


def get_top_market_news(days_back: int = 1) -> list[dict[str, Any]]:
    """Top market headlines from across the market."""
    try:
        t = yf.Ticker("^GSPC")  # S&P 500 news feed is a good proxy
        news = t.news or []
        cutoff = datetime.now() - timedelta(days=days_back)
        items = []
        for n in news[:15]:
            ts = n.get("providerPublishTime")
            published = datetime.fromtimestamp(ts) if ts else None
            if published and published < cutoff:
                continue
            items.append({
                "title": n.get("title", ""),
                "publisher": n.get("publisher", ""),
                "link": n.get("link", ""),
                "time": published.isoformat() if published else None,
            })
        return items
    except Exception as exc:  # noqa: BLE001
        logger.debug("top news failed: %s", exc)
        return []


def get_market_overview() -> dict[str, Any]:
    """Compact market overview — indices + real top gainers/losers."""
    try:
        indices = get_index_snapshot()
        movers = get_top_movers(limit=5)
        gainers = movers.get("gainers", []) if isinstance(movers, dict) else []
        losers = movers.get("losers", []) if isinstance(movers, dict) else []
        notable = [{"ticker": m["ticker"], "pct_change": m["pct_change"]} for m in gainers[:5]]
        return {
            "indices": indices,
            "notable_movers": notable,
            "gainers": gainers,
            "losers": losers,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("market overview failed")
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────
# Real top movers
# ─────────────────────────────────────────────────────────────
MOVER_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "AVGO",
    "JPM", "V", "MA", "WMT", "COST", "NFLX", "AMD", "INTC", "QCOM", "CRM",
    "ORCL", "ADBE", "DIS", "KO", "PEP", "MCD", "SBUX", "NKE", "BA", "CAT",
    "GS", "UNH", "PFE", "JNJ", "MRK", "LLY", "UBER", "ABNB", "PYPL", "COIN",
    "PLTR", "SNOW", "DDOG", "CRWD", "NET", "SHOP", "ARM", "TSM", "ASML",
    "PANW", "MU", "GM", "F", "HON", "LMT", "UPS", "FDX", "XOM", "CVX",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "WIPRO.NS",
    "SAP.DE", "SIE.DE", "BAS.DE", "AIR.PA", "MC.PA", "ULVR.L", "AZN.L",
    "BABA", "JD", "BIDU",
]

_movers_cache: dict[str, Any] = {"ts": 0.0, "data": None}


def get_top_movers(limit: int = 5) -> dict[str, Any]:
    """Actual top gainers and losers across a liquid universe (cached ~10 min).

    Fetches a broad universe in a single batched yfinance request for speed,
    then returns the biggest daily % gainers and losers.
    """
    import time

    now = time.time()
    if _movers_cache["data"] and (now - _movers_cache["ts"]) < 600:
        return _movers_cache["data"]

    try:
        hist = yf.download(
            MOVER_UNIVERSE,
            period="5d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            timeout=25,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("top movers download failed: %s", exc)
        return {"error": "Could not fetch top movers right now."}

    if hist is None or hist.empty:
        return {"error": "Could not fetch top movers right now."}

    rows: list[dict[str, Any]] = []
    multiindex = hasattr(hist.columns, "get_level_values")
    for ticker in MOVER_UNIVERSE:
        try:
            if multiindex:
                if ticker not in hist.columns.get_level_values(0):
                    continue
                closes = hist[ticker]["Close"].dropna()
            else:
                closes = hist["Close"].dropna()
            if len(closes) < 2:
                continue
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            if not prev or last <= 0:
                continue
            pct = round(((last - prev) / prev) * 100, 2)
            rows.append({"ticker": ticker, "price": round(last, 2), "pct_change": pct})
        except Exception:  # noqa: BLE001
            continue

    rows.sort(key=lambda r: r["pct_change"], reverse=True)
    result = {
        "gainers": rows[:limit],
        "losers": sorted(rows, key=lambda r: r["pct_change"])[:limit],
        "as_of": datetime.now().isoformat(),
    }
    _movers_cache["ts"] = now
    _movers_cache["data"] = result
    return result


# ─────────────────────────────────────────────────────────────
# Regional markets & news
# ─────────────────────────────────────────────────────────────
REGIONS: dict[str, dict[str, Any]] = {
    "us": {
        "name": "United States",
        "aliases": ["united states", "usa", "america", "american", "wall street", "us market"],
        "indices": {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones"},
        "top_stocks": ["NVDA", "MSFT", "AAPL", "TSLA", "META"],
        "news_tickers": ["AAPL", "MSFT", "NVDA", "TSLA", "JPM"],
    },
    "india": {
        "name": "India",
        "aliases": ["india", "indian", "nifty", "sensex", "india market"],
        "indices": {"^NSEI": "NIFTY 50", "^BSESN": "Sensex"},
        "top_stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
        "news_tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "BHARTIARTL.NS"],
    },
    "europe": {
        "name": "Europe",
        "aliases": ["europe", "european", "uk", "united kingdom", "britain", "germany", "france", "ftse", "dax", "cac", "london", "paris", "frankfurt"],
        "indices": {"^FTSE": "FTSE 100 (UK)", "^GDAXI": "DAX (Germany)", "^FCHI": "CAC 40 (France)"},
        "top_stocks": ["SAP.DE", "SIE.DE", "AZN.L", "ULVR.L", "AIR.PA"],
        "news_tickers": ["SAP.DE", "AZN.L", "ULVR.L", "BAS.DE", "MC.PA"],
    },
    "japan": {
        "name": "Japan",
        "aliases": ["japan", "japanese", "tokyo", "nikkei"],
        "indices": {"^N225": "Nikkei 225", "^TPX": "TOPIX"},
        "top_stocks": ["7203.T", "6758.T", "9984.T", "6861.T", "8306.T"],
        "news_tickers": ["7203.T", "6758.T", "9984.T"],
    },
    "china": {
        "name": "China & Hong Kong",
        "aliases": ["china", "chinese", "hong kong", "hk", "hang seng", "shanghai", "asia"],
        "indices": {"^HSI": "Hang Seng (HK)", "000001.SS": "Shanghai Composite"},
        "top_stocks": ["BABA", "JD", "BIDU", "0700.HK", "9988.HK"],
        "news_tickers": ["BABA", "JD", "BIDU"],
    },
}

_regional_cache: dict[str, dict[str, Any]] = {}


def normalize_region(text: str) -> str | None:
    """Map free-text region mentions to a canonical region key."""
    padded = f" {text.lower().strip()} "
    for region, meta in REGIONS.items():
        for alias in meta["aliases"]:
            if alias in padded:
                return region
    return None


def _get_index_snapshot(indices_map: dict[str, str]) -> list[dict[str, Any]]:
    """Snapshot for an arbitrary index map: {symbol: display name}."""
    results = []
    for symbol, name in indices_map.items():
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="5d")
            if hist.empty or len(hist) < 2:
                continue
            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            results.append({"name": name, "value": round(last, 2), "pct_change": _pct_change(prev, last)})
        except Exception as exc:  # noqa: BLE001
            logger.debug("index %s failed: %s", symbol, exc)
    return results


def _get_ticker_changes(tickers: list[str]) -> list[dict[str, Any]]:
    """Daily % change for a list of tickers via one batched request."""
    rows: list[dict[str, Any]] = []
    if not tickers:
        return rows
    try:
        batch = yf.download(tickers, period="5d", progress=False, auto_adjust=True, group_by="ticker", threads=True, timeout=25)
    except Exception as exc:  # noqa: BLE001
        logger.debug("regional tickers download failed: %s", exc)
        return rows
    if batch is None or batch.empty:
        return rows
    multiindex = hasattr(batch.columns, "get_level_values")
    for ticker in tickers:
        try:
            if multiindex:
                if ticker not in batch.columns.get_level_values(0):
                    continue
                closes = batch[ticker]["Close"].dropna()
            else:
                closes = batch["Close"].dropna()
            if len(closes) < 2:
                continue
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            if not prev or last <= 0:
                continue
            rows.append({"ticker": ticker, "price": round(last, 2), "pct_change": round(((last - prev) / prev) * 100, 2)})
        except Exception:  # noqa: BLE001
            continue
    return rows


def _get_regional_news(tickers: list[str], limit: int = 6) -> list[dict[str, Any]]:
    """Aggregate recent headlines across a region's key tickers (keyless)."""
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ticker in tickers[:4]:
        for n in get_news(ticker, days_back=2)[:4]:
            title = n.get("title", "")
            if not title or title in seen:
                continue
            seen.add(title)
            items.append(n)
        if len(items) >= limit:
            break
    return items[:limit]


def get_regional_market_data(region: str) -> dict[str, Any]:
    """Regional market snapshot: indices + top movers + headlines (cached ~10 min)."""
    import time

    region_key = normalize_region(region) or region.lower().strip()
    meta = REGIONS.get(region_key)
    if not meta:
        return {
            "error": f"Unsupported region '{region}'. Try 'us', 'india', 'europe', 'japan', or 'china'.",
        }

    cached = _regional_cache.get(region_key)
    if cached and (time.time() - cached["ts"]) < 600:
        return cached["data"]

    indices = _get_index_snapshot(meta["indices"])
    movers = sorted(_get_ticker_changes(meta["top_stocks"]), key=lambda r: r["pct_change"], reverse=True)
    news = _get_regional_news(meta["news_tickers"])

    data = {
        "region": meta["name"],
        "indices": indices,
        "movers": movers[:5],
        "gainers": [m for m in movers if m["pct_change"] >= 0][:3],
        "losers": [m for m in movers if m["pct_change"] < 0][:3],
        "news": news,
    }
    _regional_cache[region_key] = {"ts": time.time(), "data": data}
    return data


def get_finnhub_news(ticker: str) -> list[dict[str, Any]]:
    """Finnhub company news — used when a key is available."""
    if not settings.finnhub_api_key:
        return []
    try:
        to_date = date.today().isoformat()
        from_date = (date.today() - timedelta(days=7)).isoformat()
        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": ticker, "from": from_date, "to": to_date, "token": settings.finnhub_api_key}
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        items = []
        for n in resp.json()[:10]:
            items.append({
                "title": n.get("headline", ""),
                "publisher": n.get("source", ""),
                "link": n.get("url", ""),
                "time": datetime.fromtimestamp(n.get("datetime", 0)).isoformat() if n.get("datetime") else None,
                "summary": n.get("summary", "")[:300],
            })
        return items
    except Exception as exc:  # noqa: BLE001
        logger.debug("finnhub news failed: %s", exc)
        return []