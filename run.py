# Run the AI Financial Assistant from `python run.py`.
# Convenience: uses polling mode so no public URL is required.
from __future__ import annotations

import logging
import threading

import uvicorn

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


def start_polling() -> None:
    """Run the Telegram bot in polling mode (no public URL needed)."""
    from app.bot import get_application
    from telegram.ext import Application

    app_bot: Application = get_application()
    logger.info("Starting Telegram bot in POLLING mode...")
    app_bot.run_polling(allowed_updates=["message"], drop_pending_updates=True)


def run_server() -> None:
    """Run the FastAPI server (health, webhook fallback, OAuth, demo routes)."""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    # Start FastAPI (with DB init + scheduler) in a background thread,
    # then block on the Telegram polling loop.
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    start_polling()