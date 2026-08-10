"""Assistant orchestration — the brain that connects memory, tools, and AI.

This is the core product logic:
1. Detect intent from user text (using LLM when available, heuristics otherwise).
2. Resolve entities (tickers), run relevant tools.
3. Synthesize a concise, personalized natural-language answer.
4. Persist conversation + learn memories.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models import Conversation, Memory, Message, User
from app.services import tools
from app.services.ai import get_reply, rule_based_reply

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Memory management
# ─────────────────────────────────────────────────────────────
def save_memory(db: Session, user: User, key: str, value: str, source: str = "conversation") -> None:
    """Upsert a long-term memory about the user."""
    existing = db.query(Memory).filter(Memory.user_id == user.id, Memory.key == key).first()
    if existing:
        existing.value = value
        existing.source = source
    else:
        db.add(Memory(user_id=user.id, key=key, value=value, source=source))
    db.commit()


def get_memories(db: Session, user: User) -> dict[str, str]:
    """All stored memories keyed by memory key."""
    rows = db.query(Memory).filter(Memory.user_id == user.id).all()
    return {m.key: m.value for m in rows}


def get_or_create_user(db: Session, telegram_id: int, username: str | None = None, first_name: str | None = None) -> User:
    """Fetch or create a user by Telegram id."""
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    if not user:
        user = User(telegram_id=telegram_id, username=username, first_name=first_name)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if username and user.username != username:
            user.username = username
            db.commit()
    return user


# ─────────────────────────────────────────────────────────────
# Conversation persistence
# ─────────────────────────────────────────────────────────────
def ensure_conversation(db: Session, user: User) -> Conversation:
    """Get (or create) the user's active conversation."""
    conv = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .first()
    )
    if not conv:
        conv = Conversation(user_id=user.id, title="General")
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return conv


def get_conversation_context(db: Session, user: User, limit: int = 10) -> str:
    """Recent conversation history formatted as context text for the LLM."""
    conv = ensure_conversation(db, user)
    db.refresh(conv)
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.id.desc())
        .limit(limit)
        .all()
    )
    messages.reverse()
    parts = []
    for m in messages:
        prefix = "User" if m.role == "user" else "Assistant"
        parts.append(f"{prefix}: {m.content[:800]}")
    return "\n".join(parts)


def add_message(db: Session, conv: Conversation, role: str, content: str, meta: dict | None = None) -> Message:
    msg = Message(conversation_id=conv.id, role=role, content=content, meta=meta or {})
    db.add(msg)
    conv.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(msg)
    return msg


def build_user_profile(db: Session, user: User, memories: dict[str, str]) -> str:
    """Compose a compact user profile for the LLM system prompt."""
    parts = []
    if user.role:
        parts.append(f"Role: {user.role}")
    if user.watchlist:
        parts.append(f"Watchlist: {', '.join(user.watchlist)}")
    if user.interests:
        parts.append(f"Interests: {', '.join(user.interests)}")
    if user.insights_preferences:
        parts.append(f"Wants insights on: {', '.join(user.insights_preferences)}")
    if user.briefing_time and user.briefing_enabled:
        parts.append(f"Daily briefing at {user.briefing_time}")
    if user.verticals:
        parts.append(f"Other interests: {', '.join(user.verticals)}")
    for k, v in memories.items():
        if k not in ("watchlist", "role", "interests"):
            parts.append(f"{k.replace('_', ' ').title()}: {v[:120]}")
    if not parts:
        return "New user — still getting to know them."
    return "; ".join(parts)


# ─────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────

INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    # ── Google integration intents first — their keywords (Gmail, Sheets,
    #    Drive, calendar, connect) are far more distinctive than generic
    #    financial words like "q1/q3" that would otherwise steal the match.
    ("connect_google", ["connect my", "connect your", "link my", "link your", "connect gmail", "connect calendar", "connect sheets", "connect drive", "sign in", "authorize", "oauth", "connect google"]),
    ("gmail", ["emails", "email", "mail", "inbox", "gmail", "conversations related", "search my emails", "summarize my emails"]),
    ("calendar_events", ["upcoming events", "what's on my calendar", "whats on my calendar", "today's schedule", "my schedule", "this week's schedule", "week ahead", "calendar"]),
    ("calendar_meeting", ["schedule a meeting", "schedule meeting", "set up a meeting", "book a meeting", "meeting with", "add a meeting", "create a meeting", "remind me at", "reminder at"]),
    ("google_sheets", ["google sheets", "spreadsheet", "analyze my sheet", "analyze the sheet", "summarize this sheet", "summarize the sheet", "financial model", "anomalies", "unusual trends", "sheet"]),
    ("google_drive", ["google drive", "my documents", "my files", "drive", "find the", "search drive"]),
    ("integrations", ["gmail", "google sheets", "google drive", "google calendar", "email"]),
    ("onboarding", ["start", "begin", "setup", "get started", "onboard"]),
    # ── Financial intents
    ("regional_markets", ["market in", "markets in", "how is india", "how is europe", "how is japan", "how is china", "how is the us", "hows india", "how are european", "how are asian", "news from", "economy news", "world markets", "global markets", "market news from", "indian market", "european market", "japanese market", "chinese market"]),
    ("research_guide", ["before investing", "before buying", "before i buy", "before i invest", "things to look for", "what to look for", "what should i look", "what to research", "what should i research", "how to start investing", "should i invest", "stock market tips", "how to pick stocks", "how to choose stocks", "investment guide", "investing checklist", "things to know before"]),
    ("price", ["price", "stock price", "quote", "how much is", "what is the value", "how is trading", "how is doing", "doing today", "how did do today", "what is the stock"]),
    ("news", ["news", "headlines", "what happened", "latest", "why did", "announcement"]),
    ("filings", ["filing", "sec", "10-k", "10-q", "8-k", "form 4", "insider"]),
    ("market_overview", ["market overview", "how is the market", "market today", "market doing", "indices", "index today", "top movers", "movers", "gainers", "losers", "top gaining", "top losing", "best performing", "worst performing"]),
    ("earnings", ["earnings", "eps", "quarterly results", "q1", "q2", "q3", "q4", "results"]),
    ("profile", ["profile", "overview of", "about the company", "tell me about", "fundamentals", "financials"]),
    ("history", ["history", "trend", "past 3 months", "past year", "performance over", "chart"]),
    ("compare", ["compare", "versus", "vs", "difference between"]),
    ("company_search", ["who is", "what company", "ticker for", "symbol for"]),
    ("briefing", ["briefing", "daily brief", "morning brief"]),
    ("alert", ["alert", "notify", "watch", "monitor", "track", "remind"]),
    ("document", ["document", "pdf", "report", "annual report", "quarterly report", "10-k file"]),
    ("greeting", ["hello", "hi", "hey", "good morning", "good evening"]),
    ("help", ["help", "what can you do", "features"]),
]


def detect_intent(text: str, user: User | None = None) -> str:
    """Detect the primary intent from user text using heuristics."""
    lowered = text.lower()
    # If the user is asking about movers specifically, prioritize that —
    # even when combined with research-advice phrasing like "before buying".
    if any(k in lowered for k in ("top movers", "movers", "gainers", "losers", "top gaining", "top losing")):
        return "market_overview"
    for intent, patterns in INTENT_PATTERNS:
        for pat in patterns:
            if pat in lowered:
                return intent
    return "general"


def extract_tickers(text: str, user: User | None = None) -> list[str]:
    """Extract candidate tickers from text: explicit tickers + company-name aliases."""
    candidates: list[str] = []
    # Pattern: uppercase 1-5 letter words (tickers) — but skip common words
    word_tokens = re.findall(r"\b[A-Z]{1,5}\b", text)
    stop = {"A", "I", "THE", "AND", "OR", "FOR", "NEW", "Q1", "Q2", "Q3", "Q4", "VS", "X", "FY"}
    for tok in word_tokens:
        if tok not in stop and tok not in candidates:
            candidates.append(tok)

    # Company names via alias map
    lowered = text.lower()
    # check aliases in text
    for name, ticker in _alias_tickers().items():
        if name in lowered and ticker not in candidates:
            candidates.append(ticker)

    # Fallback: user watchlist mentions
    if user and user.watchlist:
        for w in user.watchlist:
            if w.lower() in lowered and w.upper() not in candidates:
                candidates.append(w.upper())
    # Dedupe
    seen = set()
    result = []
    for c in candidates:
        u = c.upper()
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result[:3]


def _alias_tickers() -> dict[str, str]:
    """Small alias map used by extract_tickers."""
    return {
        "apple": "AAPL", "google": "GOOGL", "alphabet": "GOOGL", "microsoft": "MSFT",
        "nvidia": "NVDA", "tesla": "TSLA", "amazon": "AMZN", "meta": "META",
        "netflix": "NFLX", "amazon": "AMZN", "facebook": "META", "jpmorgan": "JPM",
        "goldman sachs": "GS", "visa": "V", "mastercard": "MA", "intel": "INTC",
        "amd": "AMD", "qualcomm": "QCOM", "broadcom": "AVGO", "salesforce": "CRM",
        "oracle": "ORCL", "ibm": "IBM", "walmart": "WMT", "costco": "COST",
        "mcdonalds": "MCD", "starbucks": "SBUX", "nike": "NKE", "uber": "UBER",
        "lyft": "LYFT", "airbnb": "ABNB", "paypal": "PYPL", "adobe": "ADBE",
        "zoom": "ZM", "shopify": "SHOP", "spotify": "SPOT", "palantir": "PLTR",
        "snowflake": "SNOW", "datadog": "DDOG", "crowdstrike": "CRWD",
        "cloudflare": "NET", "arm": "ARM", "asml": "ASML", "tsmc": "TSM",
        "reliance": "RELIANCE.NS", "infosys": "INFY.NS", "tcs": "TCS",
        "hdfc": "HDFCBANK.NS", "icici": "ICICIBANK.NS", "wipro": "WIPRO.NS",
        "tata motors": "TATAMOTORS.NS", "bharti airtel": "BHARTIARTL.NS",
        "pfizer": "PFE", "johnson & johnson": "JNJ", "merck": "MRK",
        "moderna": "MRNA", "lilly": "LLY", "unitedhealth": "UNH",
        "boeing": "BA", "caterpillar": "CAT", "ford": "F", "gm": "GM",
        "nvidia": "NVDA", "coca-cola": "KO", "pepsi": "PEP",
    }


# ─────────────────────────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────────────────────────
async def _build_tool_calls(
    intent: str,
    tickers: list[str],
    text: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Decide which tools to run based on intent + available tickers."""
    if not tickers:
        if intent in ("market_overview", "news"):
            if intent == "market_overview":
                return [("get_market_overview", {})]
            return [("get_news", {})]
        if intent == "earnings":
            return [("get_earnings_calendar", {"days": 7})]
        return []

    ticker = tickers[0]
    if intent == "price":
        return [("get_price", {"ticker": ticker})]
    if intent == "profile":
        return [("get_company_profile", {"ticker": ticker}), ("get_company_facts", {"ticker": ticker})]
    if intent == "news":
        return [("get_news", {"ticker": ticker})]
    if intent == "filings":
        return [("get_filings", {"ticker": ticker})]
    if intent == "history":
        return [("get_historical", {"ticker": ticker})]
    if intent == "earnings":
        return [("get_price", {"ticker": ticker}), ("get_earnings_calendar", {"days": 7})]
    if intent == "compare":
        calls = [("get_company_profile", {"ticker": tickers[0]})]
        if len(tickers) > 1:
            calls.append(("get_company_profile", {"ticker": tickers[1]}))
        return calls
    if intent == "market_overview":
        calls = [("get_market_overview", {})]
        if ticker:
            calls.append(("get_price", {"ticker": ticker}))
        return calls
    return [("get_company_profile", {"ticker": ticker})]


def _format_tool_data(name: str, data: Any) -> str:
    """Serialize tool output compactly for the LLM."""
    try:
        return json.dumps(data, default=str, indent=1)[:6000]
    except Exception:  # noqa: BLE001
        return str(data)[:6000]


async def generate_reply(
    db: Session,
    user: User,
    text: str,
    conversation_context: str | None = None,
) -> str:
    """Main entry: generate a reply for a user message, using tools + AI."""
    intent = detect_intent(text, user)
    tickers = extract_tickers(text, user)
    memories = get_memories(db, user)
    profile = build_user_profile(db, user, memories)
    context = conversation_context or get_conversation_context(db, user)

    # Onboarding flow takes priority
    if not user.onboarding_complete:
        return handle_onboarding(db, user, text)

    # Handle briefing / alert / integration requests directly
    if intent == "briefing":
        return _handle_briefing_request(user)
    if intent == "research_guide":
        return _handle_research_guide(db, user, text)
    if intent == "market_overview" and any(k in text.lower() for k in ("before buying", "before investing", "look for", "research", "things to")):
        return await _handle_movers_and_guide(db, user, text)
    if intent == "regional_markets":
        return await _handle_regional_markets(text, user, db)
    if intent == "alert":
        return _handle_alert_request(user, text, tickers)
    if intent == "connect_google":
        return await _handle_connect_google(user, text)
    if intent == "gmail":
        return await _handle_gmail_request(db, user, text)
    if intent == "calendar_events":
        return await _handle_calendar_request(db, user, text)
    if intent == "calendar_meeting":
        return await _handle_calendar_meeting_request(db, user, text)
    if intent == "google_sheets":
        return await _handle_sheets_request(db, user, text)
    if intent == "google_drive":
        return await _handle_drive_request(db, user, text)
    if intent == "integrations":
        return _handle_integration_request(user)
    if intent == "greeting" and not tickers:
        return _greeting(db, user, memories)
    if intent == "help" and not tickers:
        from app.services.ai import rule_based_reply
        return rule_based_reply("help")

    # Run tools
    tool_calls = await _build_tool_calls(intent, tickers, text)
    tool_results = []
    if tool_calls:
        for name, kwargs in tool_calls:
            try:
                result = await tools.run_tool(name, **kwargs)
                tool_results.append({"tool": name, "output": result})
            except Exception as exc:  # noqa: BLE001
                logger.warning("tool %s failed: %s", name, exc)
                tool_results.append({"tool": name, "error": str(exc)})

    prompt = _build_prompt(text, intent, tickers, tool_results)

    # Extra system context about user + tool catalog
    system = (
        f"USER PROFILE:\n{profile}\n\n"
        f"AVAILABLE DATA TOOLS AND WHAT THEY RETURN:\n{_tool_catalog_summary()}\n\n"
        "Use the tool data provided in the prompt as ground truth. "
        "NEVER invent numbers that are not in the tool data. "
        "If tool data is empty or errored, say you couldn't retrieve live data and offer alternatives."
    )

    try:
        reply = get_reply(prompt, system=system, context_text=context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("AI reply failed")
        reply = rule_based_reply(text)

    # Learn from the conversation (extract watchlist mentions for future memory)
    _learn_from_message(db, user, text, tickers, intent, reply)
    return reply


def _build_prompt(text: str, intent: str, tickers: list[str], tool_results: list[dict]) -> str:
    lines = [f"USER MESSAGE: {text}", f"DETECTED INTENT: {intent}"]
    if tickers:
        lines.append(f"RESOLVED TICKERS: {', '.join(tickers)}")
    if tool_results:
        lines.append("\nLIVE DATA RETRIEVED:")
        for tr in tool_results:
            if "output" in tr and tr["output"]:
                lines.append(f"\n--- Tool: {tr['tool']} ---")
                lines.append(_format_tool_data(tr["tool"], tr["output"]))
            elif "error" in tr:
                lines.append(f"\n--- Tool: {tr['tool']} ERROR: {tr['error']} ---")
    lines.append(
        "\n\nNow respond to the user naturally and concisely (as a financial analyst in Telegram). "
        "Format with short bullets and bold key numbers. Explain WHY the info matters. "
        "For comparisons, present a clear side-by-side structure. "
        "Stay under ~250 words. Do not mention 'tools' or 'API'. Just answer like a helpful analyst."
    )
    return "\n".join(lines)


def _tool_catalog_summary() -> str:
    from app.services.tools import TOOL_CATALOG

    lines = []
    for name, meta in TOOL_CATALOG.items():
        lines.append(f"- {name}: {meta['description']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Memorable learning
# ─────────────────────────────────────────────────────────────
def _learn_from_message(db: Session, user: User, text: str, tickers: list[str], intent: str, reply: str) -> None:
    """Update user memory based on the conversation turn."""
    lowered = text.lower()

    # Watchlist — when user says "track" / "monitor" / "follow" + ticker
    if intent == "alert" and tickers:
        current = list(user.watchlist or [])
        for t in tickers:
            if t not in current:
                current.append(t)
        user.watchlist = current
        db.commit()
        save_memory(db, user, "watchlist", ", ".join(current), source="conversation")

    # Explicit company preferences
    if any(k in lowered for k in ("i like", "i follow", "interested in", "my focus", "i track")):
        if tickers:
            current = list(user.watchlist or [])
            for t in tickers:
                if t not in current:
                    current.append(t)
            user.watchlist = current
            db.commit()
            save_memory(db, user, "watchlist", ", ".join(current), source="conversation")

    # Role / job hints
    if "i am an" in lowered or "i'm an" in lowered or "i am a" in lowered or "i'm a" in lowered:
        m = re.search(r"(?:i am an|i'm an|i am a|i'm a)\s+([a-z ]+?)(?:[.,!]|$)", lowered)
        if m and len(m.group(1).strip()) < 40:
            role = m.group(1).strip().title()
            user.role = role
            db.commit()
            save_memory(db, user, "role", role, source="conversation")


# ─────────────────────────────────────────────────────────────
# Onboarding
# ─────────────────────────────────────────────────────────────
ONBOARDING_STEPS = [
    ("role", "Great to meet you! Before we dive in — what best describes your role? (Investor, Analyst, Founder, Finance Professional, Student, etc.) — or just say **skip**."),
    ("interests", "Which companies, sectors, or markets do you actively follow? (e.g. *AI, semiconductors, NIFTY 50, Tesla*) — or **skip**."),
    ("watchlist", "Any stocks or companies you'd like me to monitor for you? (e.g. *Nvidia, Apple, Reliance*) — or **skip**."),
    ("insights", "What type of insights matter most to you? (Market news, earnings, SEC filings, analyst ratings, macro events) — or **skip**."),
    ("briefing", "Would you like a daily morning briefing? Just tell me a time (e.g. *8:30 AM*) or say **no**."),
    ("integrations", "One more thing — would you like to connect Gmail, Calendar, or Google Sheets so I can help with emails, meetings, and spreadsheets? (Say *yes*, *skip*, or tell me which one)."),
]


def start_onboarding(db: Session, user: User) -> str:
    user.onboarding_complete = False
    user.onboarding_step = "role"
    user.insights_preferences = user.insights_preferences or []
    user.watchlist = user.watchlist or []
    user.interests = user.interests or []
    db.commit()
    return (
        "👋 Welcome! I'm your AI financial assistant — part analyst, part executive assistant.\n\n"
        "I live here in Telegram so you can research companies, track markets, analyze documents, "
        "and stay on top of what matters — without switching apps.\n\n"
        + ONBOARDING_STEPS[0][1]
    )


def handle_onboarding(db: Session, user: User, text: str) -> str:
    """Process an onboarding response; advance through the flow."""
    lowered = text.strip().lower()
    step = user.onboarding_step or "role"

    if lowered in ("skip", "skip this", "no", "none", "nope", "not now", "later"):
        # skip current step
        return _advance_onboarding(db, user)

    if step == "role":
        user.role = text.strip()[:120]
        db.commit()
        save_memory(db, user, "role", user.role)
        return _advance_onboarding(db, user)

    if step == "interests":
        items = _parse_list(text)
        current = list(user.interests or [])
        for i in items:
            if i not in current:
                current.append(i)
        user.interests = current
        db.commit()
        save_memory(db, user, "interests", ", ".join(current))
        return _advance_onboarding(db, user)

    if step == "watchlist":
        items = _parse_list(text)
        # try to map names to tickers
        tickers = []
        for item in items:
            t = tools.resolve_ticker(item)
            if t and t not in tickers:
                tickers.append(t)
        if not tickers:
            tickers = items
        current = list(user.watchlist or [])
        for t in tickers:
            if t not in current:
                current.append(t)
        user.watchlist = current
        db.commit()
        save_memory(db, user, "watchlist", ", ".join(current))
        return _advance_onboarding(db, user)

    if step == "insights":
        items = _parse_list(text)
        user.insights_preferences = items
        db.commit()
        save_memory(db, user, "insights_preferences", ", ".join(items))
        return _advance_onboarding(db, user)

    if step == "briefing":
        time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", lowered)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            ampm = time_match.group(3)
            if ampm == "pm" and hour < 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            if not ampm:
                # assume 24h if hour >= 7, else morning
                if hour < 7:
                    hour += 12
            user.briefing_time = f"{hour:02d}:{minute:02d}"
            user.briefing_enabled = True
            db.commit()
            save_memory(db, user, "briefing_time", user.briefing_time)
            return _advance_onboarding(db, user)
        # default time
        from app.config import get_settings

        user.briefing_time = get_settings().default_briefing_time
        user.briefing_enabled = True
        db.commit()
        save_memory(db, user, "briefing_time", user.briefing_time)
        return _advance_onboarding(db, user)

    if step == "integrations":
        if "yes" in lowered or "connect" in lowered or "gmail" in lowered or "calendar" in lowered or "sheet" in lowered:
            user.onboarding_complete = True
            user.onboarding_step = None
            db.commit()
            lines = ["You can connect your Google accounts anytime — here are the quick links:\n"]
            from app.services.integrations import PROVIDER_LABELS, is_configured

            if is_configured():
                already = {i.provider for i in user.integrations}
                for p, label in PROVIDER_LABELS.items():
                    if p in already:
                        lines.append(f"✅ **{label}** — already connected!")
                    else:
                        link = _oauth_link(user, p)
                        if link:
                            lines.append(f"🔗 **{label}** — tap this link to connect:\n{link}")
            else:
                lines.append("⚠️ Google OAuth isn't configured on this deployment yet — just say *'connect my Gmail'* later once it is.")
            lines.append("\n🎉 You're all set! Try asking me:")
            lines.append("• *What's the biggest market-moving event today?*")
            lines.append("• *Compare Microsoft and Google*")
            lines.append("• *Track Tesla and alert me on SEC filings*")
            return "\n".join(lines)
        user.onboarding_complete = True
        user.onboarding_step = None
        db.commit()
        return (
            "🎉 You're all set! Here's what you can do now:\n\n"
            "📈 *Markets* — *What's the market doing today?*\n"
            "🏢 *Companies* — *Compare Apple and Microsoft*\n"
            "📰 *News* — *Any Tesla news this week?*\n"
            "📑 *Filings* — *Recent SEC filings for Meta?*\n"
            "🗂 *Documents* — upload a PDF and I'll summarize it\n"
            "⏰ *Daily Briefing* — I'll send you a morning brief\n\n"
            "Just chat naturally — no commands needed. I'm here all day."
        )

    # Unknown step — safely advance
    return _advance_onboarding(db, user)


def _advance_onboarding(db: Session, user: User) -> str:
    step_index = len(ONBOARDING_STEPS)
    steps = [s[0] for s in ONBOARDING_STEPS]
    if user.onboarding_step in steps:
        step_index = steps.index(user.onboarding_step) + 1
    if step_index >= len(ONBOARDING_STEPS):
        user.onboarding_complete = True
        user.onboarding_step = None
        db.commit()
        return (
            "🙌 You're all set! I've noted your preferences, and I'll keep learning as we chat.\n\n"
            "Try asking me:\n"
            "• *What should I know before markets open today?*\n"
            "• *Track Nvidia and alert me on any major news*\n"
            "• *Compare Microsoft and Google from an investment perspective*"
        )
    step_name, question = ONBOARDING_STEPS[step_index]
    user.onboarding_step = step_name
    db.commit()
    return question


def _parse_list(text: str) -> list[str]:
    """Parse a comma/and-separated list from text."""
    text = re.sub(r"\s+and\s+", ", ", text)
    parts = [p.strip().strip(".,!?") for p in re.split(r"[,;]", text)]
    return [p for p in parts if p]


# ─────────────────────────────────────────────────────────────
# Direct handlers for briefing / alerts / integrations
# ─────────────────────────────────────────────────────────────
def _handle_briefing_request(user: User) -> str:
    from app.config import get_settings

    settings = get_settings()
    if user.briefing_enabled:
        return (
            f"☀️ Your daily briefing is set for **{user.briefing_time or settings.default_briefing_time}**. "
            "I'll send a morning summary focused on your watchlist and interests.\n\n"
            "Want to change the time or topics? Just tell me, e.g. *'send my briefing at 7 AM and focus on AI stocks'*."
        )
    return (
        "I'd love to send you a daily briefing. Reply with a time you'd like it, "
        "e.g. *'8 AM'*, and I'll start preparing it."
    )


def _handle_alert_request(user: User, text: str, tickers: list[str]) -> str:
    """Set up an alert from natural language (price move / news / filing)."""
    from app.database import SessionLocal as db_factory
    from app.models import Alert as AlertModel
    lowered = text.lower()
    alert_type = "custom"

    if any(k in lowered for k in ("price", "move", "%", "percent", "drop", "surge", "fall", "rise")):
        alert_type = "price_move"
    elif any(k in lowered for k in ("news", "announcement")):
        alert_type = "news"
    elif any(k in lowered for k in ("filing", "sec", "8-k", "10-k")):
        alert_type = "filing"

    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", lowered)
    condition: dict[str, Any] = {}
    if pct_match:
        condition["pct_change"] = float(pct_match.group(1))
        condition["direction"] = "abs"
    if "drop" in lowered or "fall" in lowered:
        condition["direction"] = "down"
    if "rise" in lowered or "surge" in lowered or "move up" in lowered:
        condition["direction"] = "up"

    db = db_factory()
    try:
        for ticker in tickers[:3]:
            existing = (
                db.query(AlertModel)
                .filter(
                    AlertModel.user_id == user.id,
                    AlertModel.alert_type == alert_type,
                    AlertModel.target == ticker,
                    AlertModel.active.is_(True),
                )
                .first()
            )
            if existing:
                db.delete(existing)
                db.commit()

            alert = AlertModel(
                user_id=user.id,
                alert_type=alert_type,
                target=ticker,
                condition=condition,
                message=text,
                active=True,
            )
            db.add(alert)
        db.commit()
    finally:
        db.close()

    targets = ", ".join(tickers) if tickers else "your watchlist"
    if alert_type == "price_move":
        pct = condition.get("pct_change")
        detail = f" if it moves more than {pct}% in a day" if pct else ""
        return f"🔔 Done — I'll monitor **{targets}** for price moves{detail} and notify you."
    if alert_type == "news":
        return f"🔔 Done — I'll watch **{targets}** for major announcements and news."
    if alert_type == "filing":
        return f"🔔 Done — I'll notify you on new SEC filings for **{targets}**."
    return f"🔔 Done — I'll monitor **{targets}** and let you know when something important happens."


def _handle_research_guide(db: Session, user: User, text: str) -> str:
    """Concise expert checklist for what to research before investing."""
    watchlist = user.watchlist or []
    wl_note = ""
    if watchlist:
        wl_note = (
            f"\n\n🎯 **Your watchlist** — {', '.join(watchlist[:5])}. "
            "Want me to run this checklist on any of them? Just say e.g. *'analyze Nvidia on this checklist'*."
        )
    return (
        "Here's the checklist I use before putting money into a stock:\n\n"
        "**1️⃣ Market & Macro context**\n"
        "• Is the sector trending up or down? Rates, inflation, policy shifts?\n\n"
        "**2️⃣ Business quality**\n"
        "• Revenue growth, profit margins, and **free cash flow** — is growth real or borrowed?\n"
        "• Debt levels and interest coverage — can it survive a downturn?\n\n"
        "**3️⃣ Valuation**\n"
        "• P/E, P/B, PEG vs. **sector peers**, not just absolute numbers.\n"
        "• Is the market already pricing in the good news?\n\n"
        "**4️⃣ Signals & sentiment**\n"
        "• Recent price action & volume, earnings surprises, analyst revisions.\n"
        "• Red flags: insider selling, regulatory trouble, weak guidance.\n\n"
        "**5️⃣ Risk & position**\n"
        "• What's the downside if I'm wrong? Position size, stop level.\n\n"
        "💡 I can pull **live data** on any of this — ask e.g. *'what's Nvidia's revenue growth and P/E?'* "
        "or *'run this checklist on Tesla'*."
        + wl_note
    )


def _extract_region(text: str) -> str:
    """Extract a canonical region key from user text."""
    from app.services.market_data import normalize_region

    region = normalize_region(text)
    if region:
        return region
    lowered = text.lower()
    if any(k in lowered for k in ("india", "nifty", "sensex", "indian")):
        return "india"
    if any(k in lowered for k in ("europe", "uk", "london", "germany", "france", "ftse", "dax", "european", "britain")):
        return "europe"
    if any(k in lowered for k in ("japan", "tokyo", "nikkei", "japanese")):
        return "japan"
    if any(k in lowered for k in ("china", "hong kong", "hang seng", "shanghai", "chinese")):
        return "china"
    return "us"


async def _handle_movers_and_guide(db: Session, user: User, text: str) -> str:
    """Real top movers + a compact due-diligence checklist (combined answer)."""
    from app.services import tools as tools_registry

    data = await tools_registry.run_tool("get_market_overview")
    lines = ["📊 **Today's Top Movers**\n"]
    if isinstance(data, dict) and data.get("indices"):
        for idx in data["indices"][:4]:
            icon = "🟢" if (idx.get("pct_change") or 0) >= 0 else "🔴"
            lines.append(f"{icon} {idx['name']}: {idx['value']:,} ({idx['pct_change']:+.2f}%)")
    movers = []
    if isinstance(data, dict):
        movers = data.get("gainers", data.get("notable_movers", []))
    if movers:
        lines.append("\n**Top Gainers**")
        for m in movers[:5]:
            icon = "🟢" if (m.get("pct_change") or 0) >= 0 else "🔴"
            name = m.get("name", "")
            lines.append(f"{icon} {m['ticker']}{' (' + name + ')' if name else ''} — {m['pct_change']:+.2f}%")
    if isinstance(data, dict) and data.get("losers"):
        lines.append("\n**Top Losers**")
        for m in data["losers"][:3]:
            icon = "🔴"
            lines.append(f"{icon} {m['ticker']} — {m['pct_change']:+.2f}%")
    lines.append(
        "\n\n**💡 Before buying any of these, check:**\n"
        "1️⃣ Why did it move? News, earnings, or just momentum?\n"
        "2️⃣ Valuation vs. peers — is it still attractive after the run?\n"
        "3️⃣ Fundamentals — revenue growth, margins, debt, free cash flow.\n"
        "4️⃣ Risk — position size, stop level, downside if you're wrong."
    )
    lines.append("\nWant me to pull the fundamentals on any of these? Just name the ticker.")
    return "\n".join(lines)


async def _handle_regional_markets(text: str, user: User, db: Session) -> str:
    """Fetch and return a regional market snapshot (indices + movers + news)."""
    region = _extract_region(text)
    from app.services import tools as tools_registry

    data = await tools_registry.run_tool("get_regional_market_data", region=region)
    if not data or (isinstance(data, dict) and data.get("error")):
        err = data.get("error") if isinstance(data, dict) else "Unknown error"
        return f"⚠️ {err}"

    lines = [f"🌍 **{data['region']} — Market Snapshot**\n"]
    if data.get("indices"):
        for idx in data["indices"]:
            icon = "🟢" if (idx.get("pct_change") or 0) >= 0 else "🔴"
            lines.append(f"{icon} {idx['name']}: {idx['value']:,} ({idx['pct_change']:+.2f}%)")
    if data.get("movers"):
        lines.append("\n**Top Movers**")
        for m in data["movers"][:5]:
            icon = "🟢" if (m.get("pct_change") or 0) >= 0 else "🔴"
            lines.append(f"{icon} {m['ticker']} — {m['pct_change']:+.2f}%")
    if data.get("news"):
        lines.append("\n**Headlines**")
        for n in data["news"][:5]:
            lines.append(f"• {n.get('title', '')[:140]}")
    lines.append("\nWant me to dig into any of these companies, or set alerts on them?")
    return "\n".join(lines)


def _handle_integration_request(user: User) -> str:
    from app.services.integrations import describe_connection_status, is_configured

    status = describe_connection_status(user)
    if not is_configured():
        return (
            "I'd love to connect your Google accounts, but Google OAuth isn't configured "
            "on this deployment yet. My other features (markets, company research, filings, "
            "documents, briefings, alerts) all work without it.\n\n"
            f"Current status: {status}"
        )
    return (
        "Here are the integrations I offer:\n\n"
        "📧 **Gmail** — search and summarize emails about a company\n"
        "📅 **Google Calendar** — prep for meetings, find time, track events\n"
        "📊 **Google Sheets** — analyze financial models & detect anomalies\n"
        "🗂 **Google Drive** — find and summarize documents\n\n"
        "Just tell me which one to connect, e.g. *'connect my Gmail'*, and I'll walk you through it.\n\n"
        f"Current status: {status}"
    )


def _get_integration(user: User, provider: str):
    """Fetch a connected integration for a user (or None)."""
    return next((i for i in user.integrations if i.provider == provider), None)


def _oauth_link(user: User, provider: str) -> str:
    """Build a clickable OAuth connect link for the user."""
    from urllib.parse import urlencode

    from app.config import get_settings
    from app.services.integrations import is_configured

    settings = get_settings()
    if not is_configured():
        return ""
    # Derive the public base URL from the redirect URI host (port preserved).
    from urllib.parse import urlsplit

    redirect = settings.google_redirect_uri
    parts = urlsplit(redirect)
    base = f"{parts.scheme}://{parts.netloc}"
    # e.g. http://localhost:8000 → /oauth/google/connect
    params = urlencode({"telegram_id": user.telegram_id, "provider": provider})
    return f"{base}/oauth/google/connect?{params}"


async def _handle_connect_google(user: User, text: str) -> str:
    """Walk the user through connecting a Google account via OAuth."""
    from app.services.integrations import PROVIDER_LABELS, is_configured

    if not is_configured():
        return (
            "I'd love to help you connect Google, but OAuth isn't configured "
            "on this deployment yet. Everything else (markets, research, filings, "
            "documents, briefings, alerts) still works without it."
        )

    lowered = text.lower()
    providers = []
    if any(k in lowered for k in ("gmail", "email", "mail")):
        providers.append("gmail")
    if any(k in lowered for k in ("calendar", "meeting", "schedule")):
        providers.append("google_calendar")
    if any(k in lowered for k in ("sheet", "spreadsheet")):
        providers.append("google_sheets")
    if any(k in lowered for k in ("drive", "doc", "file")):
        providers.append("google_drive")
    if not providers:
        providers = ["gmail", "google_calendar", "google_sheets", "google_drive"]

    already = {i.provider for i in user.integrations}
    lines = []
    for p in providers:
        label = PROVIDER_LABELS.get(p, p)
        if p in already:
            lines.append(f"✅ **{label}** — already connected!")
        else:
            link = _oauth_link(user, p)
            if link:
                # Bare URL on its own line — Telegram auto-links it even in
                # plain text, so it stays clickable if Markdown ever fails.
                lines.append(f"🔗 **{label}** — tap this link to connect:\n{link}")
            else:
                lines.append(f"⚠️ **{label}** — setup link unavailable (check env config).")
    lines.append("\nAfter you click the link and authorize, you'll be redirected back and I'll confirm here.")
    lines.append("You can skip this anytime — just ask me about markets, companies, or documents instead.")
    return "\n".join(lines)


async def _handle_gmail_request(db: Session, user: User, text: str) -> str:
    """Search + summarize the user's Gmail for a company/topic."""
    from app.services.integrations import gmail_search

    integration = _get_integration(user, "gmail")
    if not integration:
        return (
            "I'd love to search your inbox, but Gmail isn't connected yet.\n"
            "Say *'connect my Gmail'* and I'll send you a link — takes ~30 seconds."
        )

    # extract a search topic: company name or after 'about'/'for'
    lowered = text.lower()
    search_query = ""
    m = re.search(r"(?:about|for|related to|on)\s+([a-zA-Z\.]+)", lowered)
    if m:
        topic = m.group(1)
        search_query = f'"{topic}"'

    results = await gmail_search(integration, query=search_query, max_results=8)
    if not results:
        return "I searched your inbox and didn't find relevant messages for that. Try a different company or phrase, or say *'connect my Gmail'* if it isn't linked."

    from app.services.ai import get_reply

    prompt = (
        "Here are the user's recent Gmail messages matching their request:\n"
        + json.dumps(results, default=str, indent=1)[:6000]
        + "\n\nSummarize these emails concisely for a busy finance professional. "
        "Group by topic, highlight action items, deadlines, or financial context. "
        "Keep it under 200 words. Do not invent details not present in the emails."
    )
    try:
        return await _run_ai_reply(prompt, user)
    except Exception:  # noqa: BLE001
        lines = ["📧 **Here's what I found in your inbox:**\n"]
        for r in results[:5]:
            lines.append(f"**{r.get('subject', 'No subject')}**")
            if r.get("from"):
                lines.append(f"From: {r['from']}")
            if r.get("snippet"):
                lines.append(f"_{r['snippet'][:120]}_")
            lines.append("")
        lines.append("Want me to summarize all of these or search for something specific?")
        return "\n".join(lines)


async def _handle_calendar_request(db: Session, user: User, text: str) -> str:
    """Return upcoming calendar events for the user."""
    from app.services.integrations import calendar_upcoming

    integration = _get_integration(user, "google_calendar")
    if not integration:
        return (
            "I can pull your schedule, but Calendar isn't connected yet.\n"
            "Say *'connect my calendar'* and I'll send you the link."
        )

    days = 7
    m = re.search(r"(\d+)\s*(day|week)", text.lower())
    if m:
        if "week" in m.group(2):
            days = int(m.group(1)) * 7
        else:
            days = int(m.group(1))

    events = await calendar_upcoming(integration, max_results=10, days=days)
    if not events:
        return f"Nothing on your calendar for the next {days} days. Enjoy the free time! ⏳"

    lines = [f"📅 **Your upcoming schedule ({days} days):**\n"]
    for ev in events[:8]:
        start = ev.get("start", "") or ""
        location = ev.get("location") or ""
        loc_note = f" — {location}" if location else ""
        lines.append(f"• **{ev.get('summary', 'Untitled')}** — {start}{loc_note}")
    lines.append("\nWant me to prep you for any of these meetings, or schedule something new?")
    return "\n".join(lines)


async def _handle_calendar_meeting_request(db: Session, user: User, text: str) -> str:
    """Create a calendar event from natural language."""
    from app.services.integrations import calendar_create_event

    integration = _get_integration(user, "google_calendar")
    if not integration:
        return (
            "I can schedule that, but Calendar isn't connected yet.\n"
            "Say *'connect my calendar'* and I'll send you the link."
        )

    # Parse meeting title — usually after 'meeting' / 'call' / 'huddle'
    title = "Meeting"
    m = re.search(r"(?:meeting|call|huddle|sync|catch-?up|reminder|appointment)(?: about| on| to)?\s+([^.]+)", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().strip('"').strip("'")
        # remove date/time phrases
        candidate = re.sub(r"\b(tomorrow|today|next week|at \d[\d:]*\s*(am|pm)?|at \d+|on [a-z]+day|at \d+:\d+)\b", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip(" ,-")
        if candidate:
            title = candidate

    # Parse date + time
    from datetime import datetime, timedelta

    now = datetime.now()
    start = now.replace(hour=11, minute=0, second=0, microsecond=0)
    if "tomorrow" in text.lower():
        start += timedelta(days=1)
    m_time = re.search(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text.lower())
    if m_time:
        hour = int(m_time.group(1))
        minute = int(m_time.group(2) or 0)
        ampm = m_time.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        if not ampm and hour < 7:
            hour += 12
        start = start.replace(hour=hour, minute=minute)
    end = start + timedelta(minutes=30)

    start_iso = start.isoformat()
    end_iso = end.isoformat()
    result = await calendar_create_event(integration, summary=title, start_dt=start_iso, end_dt=end_iso)
    if result.get("error"):
        return f"⚠️ Couldn't schedule that: {result['error']}"
    link = result.get("htmlLink", "")
    lines = [
        f"✅ Scheduled: **{result.get('summary', title)}**",
        f"🕐 {start.strftime('%A, %b %d at %I:%M %p')} — {end.strftime('%I:%M %p')}",
    ]
    if link:
        lines.append(f"🔗 [Open in Google Calendar]({link})")
    lines.append("\nWant me to add a reminder or invite someone?")
    return "\n".join(lines)


async def _handle_sheets_request(db: Session, user: User, text: str) -> str:
    """Find + analyze a Google Sheet by name."""
    from app.services.integrations import drive_find_by_name, drive_read_file_content, sheets_analyze

    # Searching Drive requires the Drive API; reading values needs Sheets API.
    # Accept either connection, but prefer Drive for file lookup.
    integration = _get_integration(user, "google_drive") or _get_integration(user, "google_sheets")
    if not integration:
        return (
            "I can analyze spreadsheets, but Drive/Sheets isn't connected yet.\n"
            "Say *'connect my sheets'* and I'll send you the link."
        )

    # Extract the sheet name to search for
    lowered = text.lower()
    sheet_name = ""
    m = re.search(r"(?:sheet|spreadsheet|model)\s+(?:called|named|titled)?\s*['\"]?([a-zA-Z0-9 ]+)", lowered)
    if m:
        sheet_name = m.group(1).strip()
    if not sheet_name:
        # fallback: words after 'analyze' or 'summarize'
        m2 = re.search(r"(?:analyze|summarize|review)\s+(?:the\s+)?(?:google\s+)?(?:sheet|spreadsheet|model)?\s*(?:called|named)?\s*['\"]?([a-zA-Z0-9 ]+)", lowered)
        if m2:
            sheet_name = m2.group(1).strip()
    if not sheet_name or len(sheet_name) < 2:
        sheet_name = "financial model"

    files = await drive_find_by_name(integration, sheet_name, max_results=3)
    if not files:
        return f"Couldn't find a spreadsheet matching *'{sheet_name}'* in your Drive. Try another name or say *'connect my sheets'*."

    # Pick the first sheet-like file
    file = next((f for f in files if "sheet" in (f.get("mimeType") or "")), files[0])
    content = await drive_read_file_content(integration, file["id"], mime_type=file.get("mimeType") or "")
    if content.get("error"):
        return f"⚠️ Couldn't read the file: {content['error']}"

    # Reconstruct 2D values from the tab-separated text representation
    raw_text = content.get("content", "")
    values = [line.split("\t") for line in raw_text.split("\n") if line.strip()]
    analysis = sheets_analyze(values, sheet_name=file.get("name", "Sheet"))
    lines = [f"📊 **{file.get('name')}**\n"]
    if analysis.get("error"):
        lines.append(analysis["error"])
    else:
        lines.append(f"Headers: {', '.join(str(h) for h in analysis['headers'][:10])}")
        lines.append(f"Rows: {analysis['row_count']}")
        if analysis.get("numeric_stats"):
            lines.append("\n**Key numeric columns:**")
            for col, stats in list(analysis["numeric_stats"].items())[:6]:
                lines.append(
                    f"• {col} — mean {stats['mean']}, range {stats['min']}–{stats['max']}"
                )
        if analysis.get("anomalies"):
            lines.append("\n⚠️ **Potential anomalies:**")
            for a in analysis["anomalies"][:5]:
                lines.append(f"• {a}")
        else:
            lines.append("\nNo major anomalies found in the sampled range.")
    lines.append("\nWant me to dig deeper into any column or compare this with another sheet?")
    return "\n".join(lines)


async def _handle_drive_request(db: Session, user: User, text: str) -> str:
    """Find + summarize a document in the user's Google Drive."""
    from app.services.integrations import drive_find_by_name, drive_read_file_content

    integration = _get_integration(user, "google_drive")
    if not integration:
        return (
            "I can search Drive, but it isn't connected yet.\n"
            "Say *'connect my drive'* and I'll send you the link."
        )

    lowered = text.lower()
    # Extract doc name: after 'the' / 'file' / 'document' / 'search for'
    doc_query = ""
    m = re.search(r"(?:find|search(?: for)?|read|summarize|analyze)\s+(?:the\s+)?(?:file|document|doc)?\s*['\"]?([a-zA-Z0-9 _\-]+)", lowered)
    if m:
        doc_query = m.group(1).strip(" '\"")
    if not doc_query or len(doc_query) < 2:
        doc_query = "financial"

    files = await drive_find_by_name(integration, doc_query, max_results=5)
    if not files:
        return f"Couldn't find a document matching *'{doc_query}'* in your Drive."

    lines = [f"🗂 **Found {len(files)} file(s) in Drive:**\n"]
    for f in files[:5]:
        lines.append(f"• **{f.get('name')}** — {f.get('mimeType', '').split('.')[-1]} (modified {f.get('modifiedTime', '')[:10]})")
    lines.append("\nWant me to *read* one of these and give you a summary? Just tell me which one.")
    return "\n".join(lines)


async def _run_ai_reply(prompt: str, user: User) -> str:
    from app.services.ai import get_reply, rule_based_reply

    try:
        return await asyncio.to_thread(
            get_reply, prompt, system="You are a concise, helpful financial assistant on Telegram."
        )
    except Exception:  # noqa: BLE001
        return rule_based_reply(prompt)


def _greeting(db: Session, user: User, memories: dict[str, str]) -> str:
    name = user.first_name or "there"
    watchlist = user.watchlist or []
    if watchlist:
        wl = ", ".join(watchlist[:5])
        return (
            f"👋 Hey {name}! I've got your watchlist covered — {wl}.\n\n"
            "What would you like to look into today? For example:\n"
            "• *What's the market doing this morning?*\n"
            "• *Any news on my watchlist?*\n"
            "• *Show me upcoming earnings*"
        )
    return (
        f"👋 Hey {name}! Ready when you are.\n\n"
        "Try:\n"
        "• *What's the biggest market-moving event today?*\n"
        "• *Compare Microsoft and Google*\n"
        "• *Track Tesla and alert me on SEC filings*\n"
        "• Upload a PDF and ask me to analyze it"
    )