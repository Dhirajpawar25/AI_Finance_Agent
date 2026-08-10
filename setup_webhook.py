"""Register the FastAPI webhook with Telegram.

Usage:
    python setup_webhook.py https://your-domain.com
"""
from __future__ import annotations

import sys

import httpx

from app.config import get_settings

settings = get_settings()


def _print_safe(text: str) -> None:
    """Print text safely even on Windows consoles without emoji support (cp1252)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def register_webhook(public_url: str) -> None:
    """Set the Telegram bot webhook to the public URL."""
    if not settings.telegram_bot_token:
        print("TELEGRAM_BOT_TOKEN is not set. Add it to your .env file.")
        sys.exit(1)

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/setWebhook"
    webhook_url = f"{public_url.rstrip('/')}{settings.webhook_path}"
    resp = httpx.get(url, params={"url": webhook_url, "allowed_updates": '["message"]'})
    data = resp.json()
    if data.get("ok"):
        _print_safe(f"[OK] Webhook registered: {webhook_url}")
    else:
        _print_safe(f"[FAILED] {data}")
        sys.exit(1)

    info_resp = httpx.get(f"https://api.telegram.org/bot{settings.telegram_bot_token}/getWebhookInfo")
    print("Webhook info:", info_resp.json())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup_webhook.py https://your-domain.com")
        sys.exit(1)
    register_webhook(sys.argv[1])
