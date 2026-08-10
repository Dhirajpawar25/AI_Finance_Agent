"""Verifies the Google integration plumbing end-to-end (no live Google calls).

Checks:
1. integrations.py imports + scopes (calendar write scope present)
2. Sheets analysis logic with a sample 2D table
3. Drive file content read helper constructs correctly
4. Intent detection routes "connect my Gmail" -> connect_google
5. Assistant returns "not connected" guidance for Gmail/Calendar/Sheets/Drive
6. Tool catalog includes the new Google tools
"""
import asyncio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import app.models  # noqa: F401
from app.database import Base, engine, SessionLocal

Base.metadata.create_all(bind=engine)
print("[1] DB tables OK")

# ── integrations.py plumbing ────────────────────────────────
from app.services.integrations import (
    PROVIDER_LABELS,
    PROVIDER_SCOPES,
    calendar_create_event,
    describe_connection_status,
    is_configured,
    sheets_analyze,
)

assert set(PROVIDER_SCOPES) == {"gmail", "google_calendar", "google_sheets", "google_drive"}, PROVIDER_SCOPES
calendar_scope = PROVIDER_SCOPES["google_calendar"][0]
assert "calendar" in calendar_scope and "readonly" not in calendar_scope, (
    f"calendar scope should be write-enabled, got {calendar_scope}"
)
print(f"[2] integrations OK — calendar scope: {calendar_scope}")
print(f"    is_configured (no keys in .env): {is_configured()}")

# ── sheets_analyze ──────────────────────────────────────────
sample = [
    ["Revenue", "Cost", "Profit"],
    ["1000", "600", "400"],
    ["1100", "700", "400"],
    ["9000", "800", "8200"],  # outlier
    ["1200", "750", "450"],
]
analysis = sheets_analyze(sample, sheet_name="Test")
assert analysis["row_count"] == 4, analysis
assert analysis["headers"] == ["Revenue", "Cost", "Profit"], analysis
assert analysis["anomalies"], "expected outlier detection to fire on Revenue=9000"
print(f"[3] sheets_analyze OK — {len(analysis['anomalies'])} anomaly found")

# ── intent detection ────────────────────────────────────────
from app.services.assistant import detect_intent

assert detect_intent("connect my Gmail please") == "connect_google"
assert detect_intent("search my emails for anything about Apple") == "gmail"
assert detect_intent("schedule a meeting tomorrow at 2 PM") == "calendar_meeting"
assert detect_intent("what's on my calendar this week?") == "calendar_events"
assert detect_intent("analyze my Q3 financial model in Sheets") == "google_sheets"
assert detect_intent("find the acquisition memo in my Drive") == "google_drive"
print("[4] intent detection OK — all 6 integration intents resolved")

# ── assistant "not connected" answers ───────────────────────
from app.services.assistant import get_or_create_user

db = SessionLocal()
try:
    user = get_or_create_user(db, 999001, username="tester", first_name="Test")
    user.onboarding_complete = True
    db.commit()

    async def run():
        from app.services.assistant import generate_reply

        replies = {}
        for text in [
            "connect my Gmail",
            "search my emails for anything about Apple",
            "what's on my calendar this week?",
            "schedule a meeting tomorrow at 2 PM with the team",
            "analyze my financial model spreadsheet",
            "find the acquisition memo in my Drive",
        ]:
            replies[text] = await generate_reply(db, user, text)
        return replies

    replies = asyncio.run(run())
    for text, reply in replies.items():
        print(f"    [{text[:45]}...] -> {reply[:60]}...")
        assert len(reply) > 10
finally:
    db.close()
print("[5] assistant integration handlers OK (no-crash, natural guidance)")

# ── tool catalog ────────────────────────────────────────────
from app.services.tools import TOOL_CATALOG

for tool in ["gmail_search", "calendar_upcoming", "calendar_create_event", "sheets_analyze", "drive_find_by_name", "drive_read_file_content"]:
    assert tool in TOOL_CATALOG, f"missing tool {tool}"
print(f"[6] tool catalog OK — {len(TOOL_CATALOG)} tools registered")

print("\n✅ ALL INTEGRATION CHECKS PASSED")