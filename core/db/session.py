"""
Engine + session factory for MySQL, built from core.config.settings.

Usage:
    from core.db.session import get_session

    with get_session() as session:
        session.add(some_model)
        # commit happens automatically on clean exit, rollback on exception
"""

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from core.config.settings import settings

engine = create_engine(
    settings.db.sqlalchemy_url,
    pool_size=settings.db.pool_size,
    pool_pre_ping=True,   # avoids "MySQL server has gone away" on long-idle connections
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
