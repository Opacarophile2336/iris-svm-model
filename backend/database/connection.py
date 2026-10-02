"""Database connection adapter for HMNC_PRO.

Dual Database Architecture:
1. PRODUCTION: PostgreSQL via DATABASE_URL (psycopg2) for Supabase / Render
2. LOCAL DEVELOPMENT: SQL Server via pyodbc (Windows Integrated Auth or SQL Auth)
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

import config

logger = logging.getLogger(__name__)


def build_sql_server_odbc_string() -> str:
    """Build the ODBC connection string for local SQL Server."""
    if config.DB_USER and config.DB_PASSWORD:
        # SQL Server Authentication (Cloud / Azure SQL / Remote Linux)
        return (
            f"DRIVER={{{config.DB_DRIVER}}};"
            f"SERVER={config.DB_SERVER},{config.DB_PORT};"
            f"DATABASE={config.DB_NAME};"
            f"UID={config.DB_USER};"
            f"PWD={config.DB_PASSWORD};"
            f"Encrypt={config.DB_ENCRYPT};"
            f"TrustServerCertificate={config.DB_TRUST_SERVER_CERTIFICATE};"
        )
    else:
        # Windows Integrated Authentication (Default for local development on Windows)
        return (
            f"DRIVER={{{config.DB_DRIVER}}};"
            f"SERVER={config.DB_SERVER};"
            f"DATABASE={config.DB_NAME};"
            f"Trusted_Connection={config.DB_TRUSTED_CONNECTION};"
            f"TrustServerCertificate={config.DB_TRUST_SERVER_CERTIFICATE};"
        )


def resolve_connection_url(raw_url: Optional[str] = None) -> Tuple[str, str]:
    """Resolve the database dialect and normalized connection URL.

    Args:
        raw_url: Optional raw connection URL (defaults to DATABASE_URL environment variable).

    Returns:
        Tuple of (dialect, normalized_url)
        where dialect is 'postgresql' or 'mssql'.
    """
    url = raw_url if raw_url is not None else os.getenv("DATABASE_URL")
    if not url and hasattr(config, "DATABASE_URL"):
        url = config.DATABASE_URL

    if url and (
        url.startswith("postgresql://")
        or url.startswith("postgres://")
        or url.startswith("postgresql+psycopg2://")
    ):
        # Normalize postgres:// and postgresql:// to postgresql+psycopg2:// for SQLAlchemy 2.0+
        if url.startswith("postgres://"):
            normalized = url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif url.startswith("postgresql://"):
            normalized = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        else:
            normalized = url
        return "postgresql", normalized

    # Fallback to local SQL Server
    odbc_str = build_sql_server_odbc_string()
    sql_server_url = (
        "mssql+pyodbc:///?odbc_connect="
        + quote_plus(odbc_str)
    )
    return "mssql", sql_server_url


# Backward compatibility export
ODBC_CONNECTION_STRING = build_sql_server_odbc_string()

# --- Active Engine & Session Initialization ---
DB_DIALECT, DATABASE_URL = resolve_connection_url()

try:
    engine: Optional[Engine] = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )
    if DB_DIALECT == "postgresql":
        logger.info("Initialized PostgreSQL database engine via DATABASE_URL.")
    else:
        logger.info(f"Initialized SQL Server database engine for {config.DB_NAME} via pyodbc.")
except Exception as e:
    logger.warning(f"Could not initialize {DB_DIALECT.upper()} database engine: {e}")
    engine = None
    SessionLocal = None


def get_db():
    """Provide a database session for FastAPI dependencies with safe offline fallback."""
    if SessionLocal is None:
        yield None
        return

    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session exception: {e}")
        raise
    finally:
        try:
            db.close()
        except Exception:
            pass