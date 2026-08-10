"""AI provider layer — OpenAI, Anthropic, Gemini, and a rule-based fallback.

The provider is selected automatically based on which API keys are present
in the environment (OpenAI → Anthropic → Gemini → rule-based fallback).

When no API key is configured, the rule-based fallback still works: it
parses the tool data that the orchestration layer already retrieved and
synthesizes a concise, deterministic financial answer from it.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Max conversation context to send (avoid blowing token limits)
MAX_CONTEXT_CHARS = 20000


def _format_system(system: str) -> str:
    return (
        "You are a senior financial analyst and executive assistant inside Telegram. "
        "You are concise, precise, and proactive. You explain WHY something matters, "
        "never just forward headlines. You admit uncertainty rather than inventing facts. "
        "You format responses for chat: short paragraphs, bullets, and bold headings — "
        "never long reports. You already know the user from previous conversations and "
        "use that context to personalize answers.\n\n"
        + system
    )


def _build_messages(
    prompt: str,
    system: str = "",
    context_messages: list[dict[str, str]] | None = None,
    context_text: str | None = None,
) -> list[dict[str, str]]:
    """Assemble system + optional conversation history + prompt."""
    messages: list[dict[str, str]] = [{"role": "system", "content": _format_system(system)}]

    if context_text:
        messages.append({"role": "system", "content": f"CONVERSATION CONTEXT:\n{context_text[:MAX_CONTEXT_CHARS]}"})

    if context_messages:
        for m in context_messages[-12:]:  # last 12 turns for tight context
            role = "assistant" if m.get("role") == "assistant" else "user"
            content = (m.get("content") or "")[:4000]
            if content.strip():
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt})
    return messages


def _openai_reply(messages: list[dict[str, str]], max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()


def _anthropic_reply(messages: list[dict[str, str]], max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    user_msgs = [m for m in messages if m["role"] in ("user", "assistant")]
    resp = client.messages.create(
        model=settings.anthropic_model,
        system=system,
        messages=user_msgs,
        max_tokens=max_tokens,
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    return text.strip()


def _gemini_reply(messages: list[dict[str, str]], max_tokens: int) -> str:
    """Gemini via the new `google-genai` SDK (falls back to legacy SDK)."""
    system_text = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    history = [m for m in messages[1:] if m["role"] in ("user", "assistant")]
    user_content = messages[-1]["content"] if messages else "Hello"

    try:
        # New SDK: google-genai (installed as `google.genai`)
        from google import genai as _genai
        from google.genai import types as _genai_types

        client = _genai.Client(api_key=settings.gemini_api_key)

        # Build a conversation array: history + current user turn.
        contents: list[dict[str, Any]] = []
        for m in history:
            contents.append({
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            })
        # The last history entry is the final user message; send it all together
        # to keep ordering consistent with OpenAI-style messages.
        contents.append({"role": "user", "parts": [{"text": user_content}]})

        config = _genai_types.GenerateContentConfig(
            system_instruction=system_text,
            max_output_tokens=max_tokens,
            temperature=0.4,
        )
        resp = client.models.generate_content(
            model=settings.gemini_model,
            contents=contents,
            config=config,
        )
        if resp.candidates and resp.candidates[0].content.parts:
            return "".join(p.text or "" for p in resp.candidates[0].content.parts).strip()
        return (resp.text or "").strip() if hasattr(resp, "text") else ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("new gemini sdk failed: %s", exc)
        try:
            # Legacy SDK: google-generativeai (may not be installed)
            import google.generativeai as genai
        except ModuleNotFoundError:
            # Legacy SDK not installed — surface the original error (e.g. quota)
            raise exc
        try:
            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel(settings.gemini_model, system_instruction=system_text if system_text else None)
            chat = model.start_chat(history=[])
            for m in history:
                role = "model" if m["role"] == "assistant" else "user"
                chat.history.append({"role": role, "parts": [m["content"]]})
            resp = chat.send_message(
                user_content,
                generation_config={"max_output_tokens": max_tokens, "temperature": 0.4},
            )
            return (resp.text or "").strip()
        except Exception:  # noqa: BLE001
            raise


def get_reply(
    prompt: str,
    system: str = "",
    context_messages: list[dict[str, str]] | None = None,
    context_text: str | None = None,
    max_tokens: int = 600,
) -> str:
    """Get a reply from the best available AI provider."""
    messages = _build_messages(prompt, system, context_messages, context_text)

    if settings.has_openai:
        try:
            return _openai_reply(messages, max_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI failed, trying next provider: %s", exc)
    if settings.has_anthropic:
        try:
            return _anthropic_reply(messages, max_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anthropic failed, trying next provider: %s", exc)
    if settings.has_gemini:
        try:
            return _gemini_reply(messages, max_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini failed, falling back to rule-based: %s", exc)

    return rule_based_reply(prompt, system, context_text)


# ─────────────────────────────────────────────────────────────
# Rule-based fallback (no API keys configured)
# ─────────────────────────────────────────────────────────────

def _extract_user_message(prompt: str) -> str:
    """Pull the actual user message out of the prompt (it starts with 'USER MESSAGE:')."""
    m = re.search(r"USER MESSAGE: (.*?)(?:\n|$)", prompt, re.DOTALL)
    return m.group(1).strip() if m else prompt


def _extract_tool_data(prompt: str) -> list[tuple[str, Any]]:
    """Extract (tool_name, parsed_json) pairs from the prompt's LIVE DATA section."""
    results: list[tuple[str, Any]] = []
    section = re.search(r"LIVE DATA RETRIEVED:(.*?)(?:\n\n\nNow respond|\Z)", prompt, re.DOTALL)
    if not section:
        return results
    block_pattern = re.compile(r"--- Tool: ([a-z_]+) ---\n(.*?)(?=\n--- Tool: |\Z)", re.DOTALL)
    for match in block_pattern.finditer(section.group(1)):
        name = match.group(1)
        payload = match.group(2).strip()
        try:
            data = json.loads(payload)
            results.append((name, data))
        except json.JSONDecodeError:
            continue
    return results


def _fmt_money(value: Any, currency: str = "") -> str:
    """Compact currency formatting (K/M/B/T) for large financial figures."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{currency}{value}" if value is not None else ""
    if abs(v) >= 1e12:
        return f"{currency}{v / 1e12:,.2f}T"
    if abs(v) >= 1e9:
        return f"{currency}{v / 1e9:,.2f}B"
    if abs(v) >= 1e6:
        return f"{currency}{v / 1e6:,.2f}M"
    return f"{currency}{v:,.2f}"


def _fmt_price(data: dict[str, Any]) -> str:
    if "error" in data:
        return f"⚠️ {data['error']}"
    icon = "🟢" if (data.get("pct_change") or 0) >= 0 else "🔴"
    name = data.get("name") or data.get("ticker") or ""
    ticker = data.get("ticker", "")
    cur = data.get("currency", "USD") or "USD"
    price = data.get("price")
    pct = data.get("pct_change") or 0.0
    lines = [f"{icon} **{name}** ({ticker}): {_fmt_money(price, cur)} ({pct:+.2f}%)"]
    detail = []
    if data.get("day_low") is not None and data.get("day_high") is not None:
        detail.append(f"Day {_fmt_money(data['day_low'], cur)}–{_fmt_money(data['day_high'], cur)}")
    if data.get("volume"):
        detail.append(f"Vol {int(data['volume']):,}")
    if data.get("marketCap"):
        detail.append(f"MCap {_fmt_money(data['marketCap'])}")
    if data.get("trailingPE"):
        try:
            detail.append(f"P/E {float(data['trailingPE']):.1f}")
        except (TypeError, ValueError):
            pass
    if data.get("fiftyTwoWeekLow") is not None and data.get("fiftyTwoWeekHigh") is not None:
        detail.append(
            f"52w {_fmt_money(data['fiftyTwoWeekLow'])}–{_fmt_money(data['fiftyTwoWeekHigh'])}"
        )
    if detail:
        lines.append("   " + " • ".join(detail))
    return "\n".join(lines)


def _fmt_profile(data: dict[str, Any]) -> str:
    if "error" in data:
        return f"⚠️ {data['error']}"
    ticker = data.get("ticker", "")
    name = data.get("longName") or ticker
    lines = [f"🏢 **{name}** ({ticker})"]
    sector = data.get("sector") or ""
    industry = data.get("industry") or ""
    if sector or industry:
        lines.append(f"   {(sector + ' / ' + industry).strip(' /')}")
    metrics = []
    if data.get("marketCap"):
        metrics.append(f"MCap {_fmt_money(data['marketCap'])}")
    if data.get("totalRevenue"):
        metrics.append(f"Revenue {_fmt_money(data['totalRevenue'])}")
    if data.get("trailingPE"):
        try:
            metrics.append(f"P/E {float(data['trailingPE']):.1f}")
        except (TypeError, ValueError):
            pass
    if data.get("dividendYield") is not None:
        try:
            metrics.append(f"Div {float(data['dividendYield']) * 100:.2f}%")
        except (TypeError, ValueError):
            pass
    if data.get("profitMargins") is not None:
        try:
            metrics.append(f"Margin {float(data['profitMargins']) * 100:.1f}%")
        except (TypeError, ValueError):
            pass
    if data.get("revenueGrowth") is not None:
        try:
            metrics.append(f"Rev growth {float(data['revenueGrowth']) * 100:.1f}%")
        except (TypeError, ValueError):
            pass
    if data.get("returnOnEquity") is not None:
        try:
            metrics.append(f"ROE {float(data['returnOnEquity']) * 100:.1f}%")
        except (TypeError, ValueError):
            pass
    if metrics:
        lines.append("   " + " • ".join(metrics[:6]))
    if data.get("recommendationKey"):
        lines.append(f"   Analyst: **{data['recommendationKey']}**")
    if data.get("targetMeanPrice"):
        lines.append(f"   Target: {_fmt_money(data['targetMeanPrice'])}")
    return "\n".join(lines)


def _fmt_company_facts(data: dict[str, Any]) -> str:
    """Format SEC company-facts data (revenue, net income, etc.)."""
    if "error" in data or not data:
        return ""
    lines = [f"📊 **{data.get('entity') or data.get('ticker', '')}** — latest reported figures"]
    for label in (
        "revenue", "net income", "cash", "total debt", "shareholders equity",
        "diluted eps", "free cash flow", "operating margin", "r&d",
    ):
        metric = data.get(label)
        if isinstance(metric, dict) and metric.get("value") is not None:
            val = metric["value"]
            as_of = metric.get("end_date") or "n/a"
            if isinstance(val, (int, float)):
                lines.append(f"• {label.title()}: {_fmt_money(val)} (as of {as_of})")
            else:
                lines.append(f"• {label.title()}: {val} (as of {as_of})")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _fmt_news(data: Any) -> str:
    if isinstance(data, dict) and "error" in data:
        return f"⚠️ {data['error']}"
    if not isinstance(data, list) or not data:
        return "I couldn't find recent headlines."
    lines = ["📰 **Latest Headlines**"]
    for n in data[:6]:
        title = n.get("title") or n.get("headline") or ""
        pub = n.get("publisher") or n.get("source") or ""
        ts = n.get("time") or ""
        date_part = ts[:10] if isinstance(ts, str) and ts else ""
        suffix = f" ({date_part})" if date_part else ""
        lines.append(f"• {title}{suffix}{' — ' + pub if pub else ''}")
    return "\n".join(lines)


def _fmt_filings(data: Any) -> str:
    if isinstance(data, dict) and "error" in data:
        return f"⚠️ {data['error']}"
    if not isinstance(data, list) or not data:
        return "No recent SEC filings found."
    lines = ["📑 **Recent SEC Filings**"]
    for f in data[:6]:
        form = f.get("form", "?")
        company = f.get("company") or f.get("ticker") or ""
        filed = f.get("filing_date") or "n/a"
        lines.append(f"• **{form}** — {company} ({filed})")
    return "\n".join(lines)


def _fmt_earnings(data: Any) -> str:
    if isinstance(data, dict) and "error" in data:
        return f"⚠️ {data['error']}"
    if not isinstance(data, list) or not data:
        return "No earnings released in the next week on the calendar."
    lines = ["🗓 **Upcoming Earnings**"]
    for e in data[:8]:
        lines.append(f"• {e.get('ticker')} — {e.get('earnings_date')} ({e.get('time') or 'TBD'})")
    return "\n".join(lines)


def _fmt_history(data: dict[str, Any]) -> str:
    if "error" in data:
        return f"⚠️ {data['error']}"
    ticker = data.get("ticker", "")
    period = data.get("period", "")
    change = data.get("change_pct")
    lines = [f"📈 **{ticker}** — {period} trend"]
    if change is not None:
        icon = "🟢" if change >= 0 else "🔴"
        lines.append(
            f"{icon} {change:+.2f}% over the period "
            f"({data.get('start_date')} → {data.get('end_date')})"
        )
    series = data.get("series", [])
    if series:
        first, last = series[0], series[-1]
        lines.append(f"   From {_fmt_money(first.get('close'))} → {_fmt_money(last.get('close'))}")
    return "\n".join(lines)


def rule_based_reply(prompt: str, system: str = "", context_text: str | None = None) -> str:
    """Deterministic financial assistant used when no AI key is configured.

    If live tool data was already retrieved by the orchestration layer, it is
    parsed and synthesized into a concise answer. Otherwise keyword heuristics
    on the actual user message (not the full prompt) drive the response.
    """
    user_msg = _extract_user_message(prompt)
    user_text = user_msg.lower()
    tool_data = _extract_tool_data(prompt)

    # 1) Synthesize from live tool data when present
    if tool_data:
        overview = next((d for name, d in tool_data if name == "get_market_overview"), None)
        if isinstance(overview, dict) and overview.get("indices") is not None:
            lines = ["📊 **Market Overview**\n"]
            for idx in overview.get("indices", []):
                icon = "🟢" if (idx.get("pct_change") or 0) >= 0 else "🔴"
                lines.append(f"{icon} {idx['name']}: {_fmt_money(idx['value'])} ({idx['pct_change']:+.2f}%)")
            movers = overview.get("notable_movers", [])
            if movers:
                lines.append("\n**Notable Movers**")
                for m in movers[:5]:
                    icon = "🟢" if (m.get("pct_change") or 0) >= 0 else "🔴"
                    lines.append(f"{icon} {m['ticker']} — {m['pct_change']:+.2f}%")
            return "\n".join(lines)

        parts: list[str] = []
        facts = next((d for name, d in tool_data if name == "get_company_facts"), None)
        facts_text = _fmt_company_facts(facts) if isinstance(facts, dict) else ""
        for name, data in tool_data:
            if name == "get_price":
                parts.append(_fmt_price(data))
            elif name == "get_company_profile":
                parts.append(_fmt_profile(data))
            elif name == "get_news":
                parts.append(_fmt_news(data))
            elif name == "get_filings":
                parts.append(_fmt_filings(data))
            elif name == "get_earnings_calendar":
                parts.append(_fmt_earnings(data))
            elif name == "get_historical":
                parts.append(_fmt_history(data))
        if facts_text:
            parts.append(facts_text)
        merged = "\n\n".join(p for p in parts if p).strip()
        if merged:
            return merged

    # 2) Keyword fallback when no live data was retrieved
    if any(k in user_text for k in ("market overview", "how is the market", "market doing", "market today", "indices")):
        try:
            from app.services.market_data import get_market_overview

            data = get_market_overview()
            lines = ["📊 **Market Overview**\n"]
            for idx in data.get("indices", []):
                icon = "🟢" if (idx.get("pct_change") or 0) >= 0 else "🔴"
                lines.append(f"{icon} {idx['name']}: {_fmt_money(idx['value'])} ({idx['pct_change']:+.2f}%)")
            movers = data.get("notable_movers", [])
            if movers:
                lines.append("\n**Notable Movers**")
                for m in movers:
                    icon = "🟢" if (m.get("pct_change") or 0) >= 0 else "🔴"
                    lines.append(f"{icon} {m['ticker']} — {m['pct_change']:+.2f}%")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return "I couldn't fetch the market overview right now."

    if "compare" in user_text or "versus" in user_text or " vs " in user_text:
        return (
            "I can compare companies side-by-side. Tell me which two companies you'd like "
            "compared — e.g. *'Compare Microsoft and Google from an investment perspective'*."
        )

    if re.search(r"\b(hello|hi|hey|good morning|good evening|good afternoon)\b", user_text):
        return (
            "👋 Hello! I'm your financial assistant.\n\n"
            "I can help you with:\n"
            "• Stock prices & company research (e.g. *What's the market doing today?*)\n"
            "• Earnings, SEC filings & news (e.g. *Any Tesla news this week?*)\n"
            "• Document analysis — upload a PDF and ask me anything about it\n"
            "• Daily briefings & alerts — tell me what to watch and when\n\n"
            "Just chat naturally — no commands needed."
        )

    if any(k in user_text for k in ("help", "what can you do", "features")):
        return (
            "Here's what I can do for you:\n\n"
            "📈 **Markets** — prices, indices, movers, history\n"
            "🏢 **Companies** — profiles, fundamentals, comparisons\n"
            "📰 **News** — latest headlines and why they matter\n"
            "📑 **Filings** — SEC 10-K/10-Q/8-K/Form 4 tracking\n"
            "🗂 **Documents** — upload PDFs; I summarize & answer questions\n"
            "⏰ **Daily Briefing** — a tailored morning summary\n"
            "🔔 **Alerts** — price moves, news, filings on your watchlist\n"
            "🧠 **Memory** — I remember your interests and watchlist\n\n"
            "Try: *What's the biggest market-moving event today?*"
        )

    if any(k in user_text for k in ("briefing", "daily brief")):
        return (
            "I can prepare a daily briefing for you. "
            "Tell me when you'd like it (e.g. *send the briefing at 8:30 AM*) "
            "or just ask *what's happening in the market today?*"
        )

    if any(k in user_text for k in ("alert", "notify", "monitor", "track")):
        return (
            "I can set up alerts for you. For example:\n"
            "• *Alert me if Nvidia moves more than 5% today*\n"
            "• *Track Tesla and notify me on any SEC filing*\n"
            "• *Notify me when Apple releases earnings*\n\n"
            "What would you like me to monitor?"
        )

    if any(k in user_text for k in ("who are you", "your name", "about you")):
        return (
            "I'm your AI financial assistant — part analyst, part executive assistant. "
            "I live inside Telegram so you can research companies, track markets, "
            "analyze documents, and stay on top of what matters without switching apps. "
            "I remember our conversations, so the more we chat, the more useful I get."
        )

    # Default helpful response
    return (
        "I'm here to help with financial research, market tracking, document analysis, "
        "and daily briefings.\n\n"
        "Try asking something like:\n"
        "• *Compare Microsoft and Google from an investment perspective*\n"
        "• *Why did Nvidia's stock move today?*\n"
        "• *Summarize Apple's latest earnings*\n"
        "• *Track Tesla and alert me on SEC filings*\n\n"
        "I'm in demo mode right now, so I can give you market data, company profiles, "
        "filings, and document summaries — but for richer analysis, add an AI API key "
        "to the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)."
    )


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from model output."""
    if not text:
        return None
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None