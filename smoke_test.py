"""End-to-end smoke test for the assistant core (no Telegram token required).

Verifies:
1. Database bootstrap
2. Config loading
3. Live market data (yfinance)
4. AI fallback path
5. Assistant orchestration: onboarding flow -> market query -> alert creation
6. FastAPI app boots and exposes expected routes
"""
import asyncio
import sys

# Windows terminals default to cp1252 which can't encode emoji — force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# IMPORTANT: models must be imported before create_all so the metadata is populated
import app.models  # noqa: F401
from app.database import Base, engine

Base.metadata.create_all(bind=engine)
print("[1/6] Database tables created OK")

from app.config import get_settings

settings = get_settings()
print(f"[2/6] Config OK — telegram_token_set={bool(settings.telegram_bot_token)}, has_ai_keys={settings.has_any_ai}")

# Live market data
from app.services.market_data import get_price, guess_ticker

ticker = guess_ticker("Nvidia") or "NVDA"
price = get_price(ticker)
assert "error" not in price, f"get_price failed: {price}"
print(f"[3/6] Market data OK — {ticker} @ {price['price']} ({price['pct_change']}%)")

# AI fallback
from app.services.ai import get_reply

reply = get_reply("hi, what can you do?")
assert isinstance(reply, str) and len(reply) > 10
print(f"[4/6] AI fallback OK — reply preview: {reply[:50]}...")

# Full assistant orchestration
from app.database import SessionLocal
from app.models import Alert
from app.services import assistant as asst

TELEGRAM_ID = 424242


async def main():
    db = SessionLocal()
    try:
        # --- Onboarding flow ---
        user = asst.get_or_create_user(db, telegram_id=TELEGRAM_ID, username="e2e_test", first_name="E2E")
        conv = asst.ensure_conversation(db, user)
        msg = asst.start_onboarding(db, user)
        assert "onboard" in msg.lower() or "welcome" in msg.lower(), f"unexpected onboarding msg: {msg}"

        # Answer onboarding questions until complete
        answers = [
            "I'm a professional investor",
            "AI, semiconductors, and technology",
            "NVDA, MSFT, TSLA",
            "Market news, earnings, SEC filings",
            "8:30 AM",
            "skip",
        ]
        for answer in answers:
            asst.add_message(db, conv, "user", answer)
            msg = await asst.generate_reply(db, user, answer)
            asst.add_message(db, conv, "assistant", msg)

        db.refresh(user)
        assert user.onboarding_complete, "onboarding did not complete"
        assert "NVDA" in user.watchlist, f"watchlist not captured: {user.watchlist}"
        print(f"[5/6] Onboarding OK — role={user.role!r}, watchlist={user.watchlist}, briefing_time={user.briefing_time}")

        # --- Market query after onboarding ---
        asst.add_message(db, conv, "user", "How is Nvidia doing today?")
        reply = await asst.generate_reply(db, user, "How is Nvidia doing today?")
        asst.add_message(db, conv, "assistant", reply)
        assert isinstance(reply, str) and len(reply) > 10
        print(f"[6/6] Assisted query OK — reply preview: {reply[:70]}...")

        # --- Alert creation ---
        asst.add_message(db, conv, "user", "Alert me if NVDA moves more than 5% in a day")
        alert_reply = await asst.generate_reply(db, user, "Alert me if NVDA moves more than 5% in a day")
        asst.add_message(db, conv, "assistant", alert_reply)
        alerts = db.query(Alert).filter(Alert.user_id == user.id).all()
        assert alerts, "alert was not created in DB"
        print(f"     Alert OK — created alert #{alerts[-1].id} ({alerts[-1].alert_type} on {alerts[-1].target})")

        print("\n✅ SMOKE TEST PASSED (onboarding + market query + alerts)")
    finally:
        db.close()


asyncio.run(main())

# FastAPI app boots and exposes routes
from app.main import app  # noqa: E402

routes = {r.path for r in app.routes}
expected = {"/", "/health", "/webhook/telegram", "/api/message", "/api/user/{telegram_id}"}
missing = expected - routes
assert not missing, f"missing routes: {missing}"
print(f"[bonus] FastAPI OK — {len(routes)} routes registered (health, webhook, /api/message present)")
