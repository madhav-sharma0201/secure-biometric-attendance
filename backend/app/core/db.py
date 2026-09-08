"""Database session management.

The engine is created lazily rather than at import time. Building it on import means
the module cannot be imported at all without a working driver and reachable
configuration — which breaks tests, tooling and any process that merely wants to
inspect the models. It also makes the connection URL impossible to override cleanly.
"""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import settings
from backend.app.models.db import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def set_engine(engine: Engine) -> None:
    """Override the engine (tests, or an alternate deployment target)."""
    global _engine, _SessionLocal
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    get_engine()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables for local/test use only.

    NOT used in deployment. create_all() silently skips tables that already exist, so
    a column added later never appears in an existing database and the failure shows up
    at runtime instead of at deploy time. Kubernetes runs `alembic upgrade head` in an
    init container before the app starts.
    """
    Base.metadata.create_all(get_engine())
