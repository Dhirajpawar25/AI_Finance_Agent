"""Telegram bot — the primary interface for the AI Financial Assistant.

Users interact using only text, voice, and images. No slash commands,
no buttons, no menus — everything is conversational.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
import uuid

from google import genai as _genai  # new SDK (installed as google-genai)
from google.genai import types as _genai_types

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from app.config import get_settings
from app.database import SessionLocal
from app.services import assistant, document_service

logger = logging.getLogger(__name__)
settings = get_settings()

_application: Application | None = None


# ─────────────────────────────────────────────────────────────
# Bot helpers
# ─────────────────────────────────────────────────────────────
def get_application() -> Application:
    global _application
    if _application is None:
        request = HTTPXRequest(connect_timeout=30, read_timeout=60)
        _application = (
            ApplicationBuilder()
            .token(settings.telegram_bot_token)
            .request(request)
            .build()
        )
        _register_handlers(_application)
    return _application


def _register_handlers(app: Application) -> None:
    """Register all message handlers (conversational only — no command-based UX)."""
    # Internal /start is allowed to reset onboarding (not advertising commands)
    app.add_handler(CommandHandler("start", start_handler))
    # Everything else is a natural conversation
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_handler))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_handler))
    app.add_handler(MessageHandler(filters.Document.PDF, pdf_handler))
    app.add_handler(MessageHandler(filters.Document.TXT, txt_handler))
    app.add_handler(MessageHandler(filters.ALL, fallback_handler))


def _get_user_data(update: Update) -> dict[str, str | int | None]:
    """Extract user metadata from a Telegram update."""
    user = update.effective_user
    return {
        "telegram_id": user.id if user else 0,
        "username": getattr(user, "username", None),
        "first_name": getattr(user, "first_name", None),
    }


def _strip_markdown(text: str) -> str:
    """Convert Markdown to plain text while keeping URLs clickable.

    Telegram auto-links bare URLs, so we unwrap [label](url) into two lines:
    'label — url'. Bold/italic/emoji markers are stripped.
    """
    # [label](url) -> "label — url" (url stays clickable as a bare link)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 — \2", text)
    # **bold** / *italic* / _italic_
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", text)
    # ``code`` / `code`
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


async def _reply(update: Update, text: str) -> None:
    """Send a reply message to the user (chunked for Telegram limits)."""
    if not text:
        return
    # Telegram limit is 4096 chars — split into multiple messages
    for i in range(0, len(text), 3900):
        chunk = text[i : i + 3900]
        try:
            await update.effective_message.reply_text(
                chunk,
                disable_web_page_preview=True,
                parse_mode="Markdown",
            )
        except Exception:
            # Fallback: strip Markdown so at least URLs stay clickable as
            # plain text (Telegram auto-links bare URLs).
            try:
                await update.effective_message.reply_text(
                    _strip_markdown(chunk),
                    disable_web_page_preview=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("failed to send reply: %s", exc)


# ─────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Onboarding entry point for new users."""
    data = _get_user_data(update)
    db = SessionLocal()
    try:
        user = assistant.get_or_create_user(
            db,
            data["telegram_id"],
            username=data["username"],
            first_name=data["first_name"],
        )
        reply = assistant.start_onboarding(db, user)
    finally:
        db.close()
    await _reply(update, reply)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all text messages conversationally."""
    text = (update.message.text or "").strip()
    if not text:
        return
    data = _get_user_data(update)
    db = SessionLocal()
    try:
        user = assistant.get_or_create_user(
            db, data["telegram_id"], username=data["username"], first_name=data["first_name"]
        )
        conv = assistant.ensure_conversation(db, user)
        # persist the user message
        assistant.add_message(db, conv, "user", text)

        # typing indicator
        await update.effective_chat.send_action(action="typing")

        reply = await assistant.generate_reply(db, user, text)
        # persist assistant message
        assistant.add_message(db, conv, "assistant", reply)
    except Exception as exc:  # noqa: BLE001
        logger.exception("text handler failed")
        reply = (
            "I hit a snag while processing that. Please try again in a moment — "
            "or rephrase your question."
        )
    finally:
        db.close()
    await _reply(update, reply)


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Transcribe voice messages and respond conversationally."""
    try:
        await update.effective_chat.send_action(action="typing")
        voice = update.effective_message.voice or update.effective_message.audio
        if not voice:
            await _reply(update, "Couldn't read that voice message. Try sending it again?")
            return

        # Download the audio file
        file = await voice.get_file()
        audio_bytes = io.BytesIO()
        await file.download_to_memory(audio_bytes)
        audio_bytes.seek(0)

        transcript = await _transcribe_audio(audio_bytes, voice.duration)
        if not transcript:
            await _reply(
                update,
                "I couldn't transcribe that audio — speech recognition isn't configured "
                "on this deployment (add GEMINI_API_KEY or an OpenAI key to enable it). "
                "Feel free to type your question instead!",
            )
            return

        # Treat the transcript as a text message
        update.effective_message.text = transcript
        await text_handler(update, context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("voice handler failed")
        await _reply(update, "I couldn't process that voice message. Please try again.")


async def _transcribe_audio(audio_bytes: io.BytesIO, duration: int | None) -> str | None:
    """Transcribe audio using OpenAI Whisper or Gemini (whichever is configured)."""
    if settings.has_openai:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key)
            audio_bytes.seek(0)
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=("voice.ogg", audio_bytes, "audio/ogg"),
            )
            return (transcript.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI transcription failed: %s", exc)

    if settings.has_gemini:
        try:
            audio_bytes.seek(0)
            response = _genai.Client(api_key=settings.gemini_api_key).models.generate_content(
                model=settings.gemini_model,
                contents=[
                    "Transcribe this audio verbatim.",
                    _genai_types.Part.from_bytes(
                        data=audio_bytes.getvalue(), mime_type="audio/ogg"
                    ),
                ],
            )
            return (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini transcription failed: %s", exc)
    return None


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze images (charts, reports, screenshots) with a vision-capable model."""
    try:
        await update.effective_chat.send_action(action="typing")
        photo = update.effective_message.photo[-1] if update.effective_message.photo else None
        caption = update.effective_message.caption or ""

        if not photo:
            await _reply(update, "I couldn't read that image. Try sending it again?")
            return

        file = await photo.get_file()
        image_bytes = io.BytesIO()
        await file.download_to_memory(image_bytes)
        image_bytes.seek(0)

        if settings.has_openai:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=settings.openai_api_key)
                image_bytes.seek(0)
                b64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this image. It is likely a financial chart, report, "
                                    "or screenshot. Extract the key information and explain what "
                                    f"matters. User caption: {caption or '(none)'}"
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ]
                resp = client.chat.completions.create(
                    model=settings.vision_model,
                    messages=messages,
                    max_tokens=500,
                )
                reply = (resp.choices[0].message.content or "").strip()
                await _reply(update, reply)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAI vision failed: %s", exc)

        if settings.has_gemini:
            try:
                image_bytes.seek(0)
                response = _genai.Client(api_key=settings.gemini_api_key).models.generate_content(
                    model=settings.gemini_model,
                    contents=[
                        "Analyze this image. It is likely a financial chart, report, or screenshot. "
                        "Extract the key information and explain what matters. "
                        f"User caption: {caption or '(none)'}",
                        _genai_types.Part.from_bytes(
                            data=image_bytes.getvalue(), mime_type="image/jpeg"
                        ),
                    ],
                )
                reply = (response.text or "").strip()
                await _reply(update, reply)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gemini vision failed: %s", exc)

        await _reply(
            update,
            "I can't analyze images on this deployment yet (vision requires an OpenAI or "
            "Gemini API key). I can still help with markets, company research, filings, "
            "and documents — just tell me what you need!",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("image handler failed")
        await _reply(update, "I couldn't process that image. Please try again.")


async def pdf_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process uploaded PDF documents — summarize and answer questions."""
    try:
        await update.effective_chat.send_action(action="typing")
        doc = update.effective_message.document
        if not doc:
            return

        # Download the PDF
        file = await doc.get_file()
        pdf_bytes = io.BytesIO()
        await file.download_to_memory(pdf_bytes)
        pdf_bytes.seek(0)

        # Extract text
        try:
            text = document_service.extract_pdf_text(pdf_bytes.getvalue(), doc.file_name or "document.pdf")
        except ValueError as exc:
            await _reply(update, str(exc))
            return

        if not text.strip():
            await _reply(
                update,
                "I couldn't extract text from this PDF (it may be a scanned document). "
                "If you have a text version, send it to me and I'll analyze it.",
            )
            return

        # Persist the document
        data = _get_user_data(update)
        db = SessionLocal()
        try:
            user = assistant.get_or_create_user(
                db, data["telegram_id"], username=data["username"], first_name=data["first_name"]
            )
            storage_path = _save_doc_file(doc.file_name or "document.pdf", pdf_bytes.getvalue())
            document_service.save_uploaded_document(user, storage_path, doc.file_name or "document.pdf", text)
        finally:
            db.close()

        # Acknowledge + summarize
        await _reply(
            update,
            f"📄 I've read **{doc.file_name}** ({len(text.split())} words). Generating a summary…",
        )
        summary = await _run_in_thread(
            document_service.summarize_document_text, text, doc.file_name or "document"
        )
        await _reply(update, summary)
        await _reply(
            update,
            "Ask me anything about this document — I have it in context. For example:\n"
            "• *What are the biggest risks?*\n"
            "• *Summarize the key financials*\n"
            "• *Compare this with my other reports*",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("pdf handler failed")
        await _reply(update, "I couldn't process that PDF. Please try again.")


async def txt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process uploaded text documents."""
    try:
        await update.effective_chat.send_action(action="typing")
        doc = update.effective_message.document
        if not doc:
            return
        file = await doc.get_file()
        content = (await file.download_as_bytearray()).decode("utf-8", errors="ignore")
        if not content.strip():
            await _reply(update, "That file appears to be empty.")
            return

        summary = await _run_in_thread(
            document_service.summarize_document_text, content, doc.file_name or "document"
        )
        await _reply(update, summary)
        await _reply(update, "Ask me anything about this document — I have it in context.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("txt handler failed")
        await _reply(update, "I couldn't process that file. Please try again.")


async def fallback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Graceful fallback for unsupported messages."""
    await _reply(
        update,
        "That message type isn't supported yet. I can work with:\n"
        "• Text — ask me anything about markets, companies, or filings\n"
        "• Voice — send a voice message and I'll transcribe + respond\n"
        "• Images — charts, reports, and screenshots\n"
        "• PDFs / text files — I'll summarize and answer questions",
    )


def _save_doc_file(filename: str, content: bytes) -> str:
    """Persist an uploaded file to the docs directory."""
    docs_dir = os.path.join(settings.data_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._-") or "doc.pdf"
    path = os.path.join(docs_dir, f"{uuid.uuid4().hex[:8]}_{safe_name}")
    with open(path, "wb") as f:
        f.write(content)
    return path


async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking call in a thread to avoid stalling the bot."""
    import asyncio

    return await asyncio.to_thread(fn, *args, **kwargs)


# ─────────────────────────────────────────────────────────────
# Public API used by scheduler & routes
# ─────────────────────────────────────────────────────────────
async def send_message_to_user(telegram_id: int, text: str) -> bool:
    """Send a proactive message (briefing/alert) to a user."""
    try:
        app = get_application()
        # Cap message length
        await app.bot.send_message(chat_id=telegram_id, text=text[:3900], disable_web_page_preview=True)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to send proactive message to %s: %s", telegram_id, exc)
        return False