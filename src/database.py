"""
Database configuration — Production-hardened.

Enforces PostgreSQL in production environments and logs warnings
for SQLite usage in non-production contexts. Configures connection
pooling for production workloads.
"""
import os
import logging
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger("hr-attrition-db")

# ── Environment Detection ──────────────────────────────────────────────
ENV = os.getenv("ENV", "development").lower()
IS_PRODUCTION = ENV == "production"

# Default to SQLite if DATABASE_URL is not set
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./hr_analytics.db"
)

_is_sqlite = SQLALCHEMY_DATABASE_URL.startswith("sqlite")

# ── Production Safety Gate ─────────────────────────────────────────────
# In production, SQLite is NOT acceptable for audit trail data that
# requires ACID compliance, WAL replication, and encryption at rest.
if IS_PRODUCTION and _is_sqlite:
    raise RuntimeError(
        "FATAL: SQLite is not permitted in production (ENV=production). "
        "Set DATABASE_URL to a PostgreSQL connection string. "
        "Example: DATABASE_URL=postgresql://user:pass@host:5432/hr_analytics"
    )

if _is_sqlite:
    logger.warning(
        "Using SQLite database — acceptable for development only. "
        "Set DATABASE_URL to PostgreSQL for production deployments."
    )

# ── Engine Configuration ───────────────────────────────────────────────
connect_args: dict = {}
engine_kwargs: dict = {}

if _is_sqlite:
    # Setting check_same_thread=False is needed for SQLite in FastAPI.
    connect_args["check_same_thread"] = False
else:
    # PostgreSQL connection pooling for production workloads
    engine_kwargs.update({
        "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
        "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
        "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "1800")),
        "pool_pre_ping": True,  # Verify connections before use
    })

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=connect_args,
    **engine_kwargs,
)

# ── SQLite WAL mode for better concurrent read performance ─────────────
if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator:
    """FastAPI Dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_info() -> dict:
    """Return database connection metadata for health checks."""
    return {
        "driver": engine.url.drivername,
        "is_production_ready": not _is_sqlite,
        "pool_size": engine_kwargs.get("pool_size", "N/A"),
        "environment": ENV,
    }
