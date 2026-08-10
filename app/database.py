"""SQLAlchemy database setup."""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()


def clean_database_url(raw: str) -> str:
    """Sanitize a DATABASE_URL coming from env vars.

    Dashboards and copy/paste often add surrounding double/single quotes or
    trailing whitespace/newlines. SQLAlchemy cannot parse those, so strip them
    here to fail-fast on the actual connection instead of a parsing error.
    """
    if not raw:
        return raw
    url = raw.strip()
    if len(url) >= 2 and url[0] == url[-1] and url[0] in ('"', "'"):
        url = url[1:-1]
    return url.strip()


database_url = clean_database_url(settings.database_url)

connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}

engine = create_engine(
    database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Import models so they register with Base.metadata."""
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)