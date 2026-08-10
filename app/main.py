"""FastAPI application — serves the Telegram webhook + OAuth callbacks + admin routes."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.config import get_settings
from app.database import Base, SessionLocal, engine
from app.services import assistant
from app.services.integrations import build_auth_url, exchange_code, is_configured, save_integration

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    # Start scheduler for briefings + alerts
    if settings.enable_scheduler:
        from app.services.scheduler import start_scheduler

        start_scheduler()
        logger.info("Scheduler started.")

    yield

    if settings.enable_scheduler:
        from app.services.scheduler import stop_scheduler

        stop_scheduler()


app = FastAPI(title="AI Financial Assistant", version="1.0.0", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────
# Health & info
# ─────────────────────────────────────────────────────────────
@app.get("/")
async def root() -> dict:
    return {
        "service": "AI Financial Assistant",
        "status": "running",
        "version": "1.0.0",
        "telegram_mode": "webhook" if settings.use_webhook else "polling",
        "ai_provider": _active_provider(),
        "webhook_path": settings.webhook_path,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "scheduler": bool(settings.enable_scheduler), "google_oauth": is_configured()}


def _active_provider() -> str:
    if settings.has_openai:
        return "openai"
    if settings.has_anthropic:
        return "anthropic"
    if settings.has_gemini:
        return "gemini"
    return "rule-based (no API key configured)"


# ─────────────────────────────────────────────────────────────
# Telegram webhook
# ─────────────────────────────────────────────────────────────
@app.post(settings.webhook_path)
async def telegram_webhook(request: Request) -> dict:
    """Receive Telegram update via webhook and dispatch to the bot."""
    from telegram import Update
    from telegram.ext import Application

    from app.bot import get_application

    app_bot: Application = get_application()
    # PTB v21+ requires Application.initialize() before process_update() can run.
    # Called lazily so the first webhook request initializes once, then the flag
    # (_initialized) keeps subsequent requests fast.
    if not getattr(app_bot, "_initialized", False):
        await app_bot.initialize()
    json_data = await request.json()
    update = Update.de_json(json_data, app_bot.bot)
    if update is None:
        return {"ok": False, "error": "Invalid update"}

    await app_bot.process_update(update)
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# Google OAuth callbacks
# ─────────────────────────────────────────────────────────────
@app.get("/oauth/google/connect")
async def google_connect(telegram_id: int, provider: str = "gmail") -> dict:
    """Generate an OAuth connect URL for a user."""
    if not is_configured():
        raise HTTPException(status_code=503, detail="Google OAuth is not configured on this deployment.")
    url = build_auth_url(state=str(telegram_id), provider=provider)
    return {"auth_url": url}


@app.get("/oauth/google/callback")
async def google_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    """Handle the OAuth callback and persist tokens."""
    if error:
        return PlainTextResponse(f"Authorization failed: {error}. You can close this tab.")
    if not code:
        return PlainTextResponse("Missing authorization code. You can close this tab.")

    # state is provider:telegram_id
    provider = "gmail"
    telegram_id = state or ""
    if state and ":" in state:
        provider, telegram_id = state.split(":", 1)

    try:
        tokens = await exchange_code(code)
    except Exception as exc:  # noqa: BLE001
        logger.exception("OAuth token exchange failed")
        return PlainTextResponse(f"Could not complete OAuth: {exc}")

    from app.models import User

    db = SessionLocal()
    try:
        try:
            user = db.query(User).filter(User.telegram_id == int(telegram_id)).first()
        except (TypeError, ValueError):
            user = None
        if not user:
            return PlainTextResponse("User not found. Please open the bot and send /start first.")
        await save_integration(user, provider, tokens)
    finally:
        db.close()

    # Notify the user in Telegram
    try:
        from app.bot import send_message_to_user

        labels = {
            "gmail": "📧 Gmail",
            "google_calendar": "📅 Google Calendar",
            "google_sheets": "📊 Google Sheets",
            "google_drive": "🗂 Google Drive",
        }
        label = labels.get(provider, provider)
        await send_message_to_user(
            int(telegram_id),
            f"✅ Connected **{label}**! I can now help you with that.\n\n"
            "Try: *'Search my emails for anything about Apple'* or *'What's on my calendar this week?'*",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("OAuth success but Telegram notification failed: %s", exc)

    return PlainTextResponse("✅ Connected! You can close this tab and go back to Telegram.")


# ─────────────────────────────────────────────────────────────
# Demo / API routes (useful for judging & testing)
# ─────────────────────────────────────────────────────────────
class MessageRequest(BaseModel):
    telegram_id: int
    text: str
    username: str | None = None
    first_name: str | None = None


class DemoResponse(BaseModel):
    reply: str
    onboarding_complete: bool
    watchlist: list[str] = []
    interests: list[str] = []


@app.post("/api/message", response_model=DemoResponse)
async def send_message_demo(req: MessageRequest) -> DemoResponse:
    """Simulate a Telegram-style message for testing/demo purposes."""
    db = SessionLocal()
    try:
        user = assistant.get_or_create_user(
            db, req.telegram_id, username=req.username, first_name=req.first_name
        )
        conv = assistant.ensure_conversation(db, user)
        assistant.add_message(db, conv, "user", req.text)
        reply = await assistant.generate_reply(db, user, req.text)
        assistant.add_message(db, conv, "assistant", reply)
        db.refresh(user)
        return DemoResponse(
            reply=reply,
            onboarding_complete=bool(user.onboarding_complete),
            watchlist=list(user.watchlist or []),
            interests=list(user.interests or []),
        )
    finally:
        db.close()


@app.get("/api/user/{telegram_id}")
async def get_user_demo(telegram_id: int) -> dict:
    """Fetch stored user profile (for judging/demo)."""
    from app.models import User

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == telegram_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "telegram_id": user.telegram_id,
            "username": user.username,
            "first_name": user.first_name,
            "role": user.role,
            "interests": user.interests or [],
            "watchlist": user.watchlist or [],
            "insights_preferences": user.insights_preferences or [],
            "briefing_time": user.briefing_time,
            "briefing_enabled": user.briefing_enabled,
            "onboarding_complete": user.onboarding_complete,
            "integrations": [i.provider for i in user.integrations],
            "memory_count": len(user.memories),
        }
    finally:
        db.close()


@app.get("/docs/health")
async def docs_health() -> dict:
    return {"status": "ok", "message": "AI Financial Assistant is running."}