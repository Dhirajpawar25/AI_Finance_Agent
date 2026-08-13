"""Scheduler — daily briefings and alert monitoring via APScheduler."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

scheduler: BackgroundScheduler | None = None


def _build_briefing(user: Any, db: Any) -> str:
    """Compose a personalized daily briefing for a user."""
    from app.services import market_data
    from app.services.ai import get_reply

    watchlist = list(user.watchlist or [])[:5]
    interests = list(user.interests or [])[:5]
    insights = list(user.insights_preferences or [])[:4]

    # Gather market context
    try:
        overview = market_data.get_market_overview()
    except Exception as exc:  # noqa: BLE001
        logger.warning("briefing market data failed: %s", exc)
        overview = {}

    watchlist_data = []
    for ticker in watchlist:
        try:
            data = market_data.get_price(ticker)
            if "error" not in data:
                watchlist_data.append(data)
        except Exception as exc:  # noqa: BLE001
            logger.debug("briefing price failed for %s: %s", ticker, exc)

    # Top news
    try:
        news = market_data.get_top_market_news(days_back=1)[:5]
    except Exception as exc:  # noqa: BLE001
        logger.warning("briefing news failed: %s", exc)
        news = []

    prompt = (
        "Create a concise morning market briefing for a finance professional.\n\n"
        f"User's watchlist: {', '.join(watchlist) if watchlist else 'none specified'}\n"
        f"User's interests: {', '.join(interests) if interests else 'general market'}\n"
        f"User wants insights on: {', '.join(insights) if insights else 'market overview'}\n\n"
        f"MARKET OVERVIEW DATA:\n{overview}\n\n"
        f"WATCHLIST PRICES:\n{watchlist_data}\n\n"
        f"TOP NEWS HEADLINES:\n{news}\n\n"
        "Structure the briefing as:\n"
        "☀️ **Morning Brief** — <date>\n"
        "1. **Markets** — index levels and tone (2-3 lines)\n"
        "2. **Your Watchlist** — key moves with % change\n"
        "3. **What Matters Today** — 3-5 news items with WHY they matter\n"
        "4. **On the Radar** — upcoming events/earnings if relevant\n\n"
        "Keep it under 250 words. Bold key numbers. No filler."
    )

    try:
        return get_reply(prompt, system="You are a precise financial analyst preparing a morning briefing.", max_tokens=600)
    except Exception as exc:  # noqa: BLE001
        logger.exception("briefing generation failed")
        return _fallback_briefing(overview, watchlist_data, news, watchlist)


def _fallback_briefing(overview: dict, watchlist_data: list, news: list, watchlist: list) -> str:
    lines = [f"☀️ **Morning Brief — {datetime.now().strftime('%A, %B %d')}**\n"]
    indices = overview.get("indices", [])
    if indices:
        lines.append("**Markets**")
        for idx in indices:
            icon = "🟢" if (idx.get("pct_change") or 0) >= 0 else "🔴"
            lines.append(f"{icon} {idx['name']}: {idx['value']} ({idx['pct_change']:+.2f}%)")
        lines.append("")
    if watchlist_data:
        lines.append("**Your Watchlist**")
        for w in watchlist_data:
            icon = "🟢" if (w.get("pct_change") or 0) >= 0 else "🔴"
            lines.append(f"{icon} {w['ticker']}: {w['price']} ({w['pct_change']:+.2f}%)")
        lines.append("")
    if news:
        lines.append("**What Matters Today**")
        for n in news[:4]:
            lines.append(f"• {n.get('title', '')}")
    if not watchlist_data and not indices and not news:
        lines.append("Data is limited this morning, but I'm watching your interests. Ask me anything!")
    return "\n".join(lines)


async def _send_briefing_to_user(user: Any, db: Any) -> None:
    """Send a briefing via the Telegram bot."""
    from app.bot import send_message_to_user

    try:
        briefing = _build_briefing(user, db)
        await send_message_to_user(user.telegram_id, briefing)
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to send briefing to user %s", user.telegram_id)


def _send_briefings_job() -> None:
    """Job: send briefings to users whose configured time matches the current hour/minute."""
    from app.database import SessionLocal
    from app.models import User

    now = datetime.now()
    db = SessionLocal()
    try:
        users = db.query(User).filter(User.briefing_enabled.is_(True)).all()
        import asyncio

        for user in users:
            briefing_time = user.briefing_time or "08:00"
            try:
                b_hour, b_minute = (int(x) for x in briefing_time.split(":"))
            except (ValueError, AttributeError):
                continue
            if b_hour == now.hour and b_minute == now.minute:
                try:
                    asyncio.run(_send_briefing_to_user(user, db))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("briefing failed for user %s: %s", user.telegram_id, exc)
    finally:
        db.close()


def _monitor_alerts_job() -> None:
    """Job: check active price/news/filing alerts and fire notifications."""
    from app.database import SessionLocal
    from app.models import Alert, AlertEvent, User

    db = SessionLocal()
    try:
        alerts = db.query(Alert).filter(Alert.active.is_(True)).all()
        import asyncio

        for alert in alerts:
            try:
                asyncio.run(_check_alert(db, alert))
            except Exception as exc:  # noqa: BLE001
                logger.warning("alert check failed for alert %s: %s", alert.id, exc)
    finally:
        db.close()


async def _check_alert(db, alert: Alert) -> None:
    """Evaluate a single alert and fire if conditions are met."""
    from app.bot import send_message_to_user
    from app.services import market_data, sec_data
    from app.models import User

    user = db.query(User).filter(User.id == alert.user_id).first()
    if not user:
        return

    # Cooldown: don't alert more than once per 6 hours for the same alert
    from datetime import timedelta

    recent = (
        db.query(AlertEvent)
        .filter(AlertEvent.alert_id == alert.id)
        .order_by(AlertEvent.created_at.desc())
        .first()
    )
    if recent and recent.created_at and (datetime.utcnow() - recent.created_at) < timedelta(hours=6):
        return

    ticker = alert.target
    fired = False
    message = ""

    try:
        if alert.alert_type == "price_move":
            price = market_data.get_price(ticker)
            if "error" not in price:
                pct = abs(price.get("pct_change") or 0)
                threshold = (alert.condition or {}).get("pct_change") or 3.0
                direction = (alert.condition or {}).get("direction", "abs")
                price_pct = price.get("pct_change") or 0
                if direction == "abs" and pct >= threshold:
                    fired = True
                elif direction == "down" and price_pct <= -threshold:
                    fired = True
                elif direction == "up" and price_pct >= threshold:
                    fired = True
                if fired:
                    message = (
                        f"🔔 **Price alert — {ticker}**\n"
                        f"{price['name']} is at {price['price']} ({price['pct_change']:+.2f}% today). "
                        f"Yesterday's close: {price.get('price', 0) - price.get('change', 0):.2f}."
                    )

        elif alert.alert_type == "news":
            news = market_data.get_news(ticker, days_back=1)
            if news:
                top = news[0]
                fired = True
                message = (
                    f"📰 **News alert — {ticker}**\n"
                    f"**{top.get('title', '')}**\n"
                    f"{top.get('publisher', '')} — {top.get('link', '')}\n\n"
                    "Reason it matters:\nAsk me *'why does this matter for {ticker}?'* and I'll break it down."
                )

        elif alert.alert_type == "filing":
            filings = await sec_data.get_filings(ticker, form_types=["8-K", "10-K", "10-Q", "4"], limit=3)
            if filings:
                top = filings[0]
                form = top.get("form", "?")
                date_filed = top.get("filing_date", "?")
                url = top.get("url", "")
                fired = True
                message = (
                    f"📑 **Filing alert — {ticker}**\n"
                    f"A new **{form}** was filed on {date_filed}.\n"
                    f"Details: {url}\n\n"
                    f"Ask me to *'summarize the latest {form} for {ticker}'*."
                )

        if fired and message:
            await send_message_to_user(user.telegram_id, message)
            db.add(AlertEvent(alert_id=alert.id, title=message[:200]))
            db.commit()
            logger.info("Alert fired for user %s: %s", user.telegram_id, ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning("alert evaluation error for %s: %s", ticker, exc)


def start_scheduler() -> BackgroundScheduler:
    """Start the background scheduler with briefing + alert jobs."""
    global scheduler
    if scheduler and scheduler.running:
        return scheduler

    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

    # Daily briefings — check every minute; users have their own time preferences
    scheduler.add_job(
        _send_briefings_job,
        CronTrigger(hour="*", minute="*", timezone="Asia/Kolkata"),
        id="briefings",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # Alert monitoring every 5 minutes
    scheduler.add_job(
        _monitor_alerts_job,
        IntervalTrigger(minutes=5),
        id="alerts",
        replace_existing=True,
        misfire_grace_time=300,
    )

    scheduler.start()
    logger.info("Scheduler started with briefing + alert jobs.")
    return scheduler


def stop_scheduler() -> None:
    """Gracefully stop the scheduler."""
    global scheduler
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        scheduler = None