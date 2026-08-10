"""SQLAlchemy ORM models for the financial assistant."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    """A Telegram user of the assistant."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Onboarding
    role: Mapped[str | None] = mapped_column(String(120), nullable=True)
    interests: Mapped[list] = mapped_column(JSON, default=list)  # sectors/topics
    watchlist: Mapped[list] = mapped_column(JSON, default=list)  # tickers/companies
    insights_preferences: Mapped[list] = mapped_column(JSON, default=list)
    briefing_time: Mapped[str | None] = mapped_column(String(5), nullable=True)
    briefing_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    onboarding_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    onboarding_step: Mapped[str | None] = mapped_column(String(120), nullable=True)
    timezone: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Optional extra verticals
    verticals: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    memories: Mapped[list["Memory"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    documents: Mapped[list["Document"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    integrations: Mapped[list["Integration"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    """A topic-scoped conversation thread."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="General")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    """A single turn in a conversation."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # user / assistant / system
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)  # attachments, tool calls, etc.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Memory(Base):
    """Long-term facts the assistant learns about the user."""

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    key: Mapped[str] = mapped_column(String(255), index=True)
    value: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship(back_populates="memories")


class Alert(Base):
    """User-defined alert that triggers proactive notifications."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    alert_type: Mapped[str] = mapped_column(String(60))  # price_move, news, filing, custom
    target: Mapped[str] = mapped_column(String(255))  # ticker, company, or query
    condition: Mapped[dict] = mapped_column(JSON, default=dict)  # e.g. {"pct_change": 5, "direction": "abs"}
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship(back_populates="alerts")
    events: Mapped[list["AlertEvent"]] = relationship(back_populates="alert", cascade="all, delete-orphan")


class Document(Base):
    """A financial document uploaded by the user."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(500))
    file_type: Mapped[str] = mapped_column(String(20))  # pdf
    storage_path: Mapped[str] = mapped_column(String(1000))
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship(back_populates="documents")


class Integration(Base):
    """A connected external service (Gmail, Calendar, Sheets, Drive)."""

    __tablename__ = "integrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    provider: Mapped[str] = mapped_column(String(60))  # gmail, google_calendar, google_sheets, google_drive
    credentials: Mapped[dict] = mapped_column(JSON, default=dict)
    connected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship(back_populates="integrations")


class AlertEvent(Base):
    """A fired alert notification (for cooldown & audit)."""

    __tablename__ = "alert_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id"), index=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    alert: Mapped["Alert"] = relationship(back_populates="events")


class BriefingLog(Base):
    """Tracks sent daily briefings to avoid duplicates."""

    __tablename__ = "briefing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    briefing_date: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    briefing_type: Mapped[str] = mapped_column(String(40), default="morning")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)