"""Verify the fixes for the user's exact test questions."""
import asyncio

import app.models  # noqa: F401
from app.database import Base, engine, SessionLocal

Base.metadata.create_all(bind=engine)

from app.services import assistant as asst

TELEGRAM_ID = 777001


async def main():
    db = SessionLocal()
    try:
        user = asst.get_or_create_user(db, telegram_id=TELEGRAM_ID, username="fix_test", first_name="Fix")
        # Complete onboarding quickly
        conv = asst.ensure_conversation(db, user)
        asst.start_onboarding(db, user)
        for answer in ["Analyst", "AI, tech", "NVDA, MSFT", "Market news", "8:00 AM", "skip"]:
            asst.add_message(db, conv, "user", answer)
            reply = await asst.generate_reply(db, user, answer)
            asst.add_message(db, conv, "assistant", reply)

        # ── Question 1: research before investing ──
        print("\n" + "=" * 60)
        print("Q1: 'What thing should I know or research before investing in stock market?'")
        q1 = "What thing should I know or research before investing in stock market"
        intent1 = asst.detect_intent(q1, user)
        print(f"Detected intent: {intent1}")
        asst.add_message(db, conv, "user", q1)
        reply1 = await asst.generate_reply(db, user, q1)
        asst.add_message(db, conv, "assistant", reply1)
        print("Response:\n" + reply1[:600])

        # ── Question 2: top movers ──
        print("\n" + "=" * 60)
        print("Q2: 'What are the top movers which things to look for before buying stocks'")
        q2 = "What are the top movers which things to look for before buying stocks"
        intent2 = asst.detect_intent(q2, user)
        print(f"Detected intent: {intent2}")
        asst.add_message(db, conv, "user", q2)
        reply2 = await asst.generate_reply(db, user, q2)
        asst.add_message(db, conv, "assistant", reply2)
        print("Response:\n" + reply2[:600])

        # ── Bonus: regional markets ──
        print("\n" + "=" * 60)
        print("Q3: 'How is the Indian market doing today?'")
        q3 = "How is the Indian market doing today"
        intent3 = asst.detect_intent(q3, user)
        print(f"Detected intent: {intent3}")
        asst.add_message(db, conv, "user", q3)
        reply3 = await asst.generate_reply(db, user, q3)
        asst.add_message(db, conv, "assistant", reply3)
        print("Response:\n" + reply3[:600])

        print("\n✅ FIX TESTS COMPLETE")
    finally:
        db.close()


asyncio.run(main())