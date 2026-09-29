"""SQL Server database connection for HMNC_PRO."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus

from config import (
    DB_DRIVER,
    DB_SERVER,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_PORT,
    DB_TRUSTED_CONNECTION,
    DB_TRUST_SERVER_CERTIFICATE,
    DB_ENCRYPT,
)

import logging

logger = logging.getLogger(__name__)

# Build ODBC connection string based on authentication mode
if DB_USER and DB_PASSWORD:
    # SQL Server Authentication (Standard for Cloud / Azure SQL / Remote Linux)
    ODBC_CONNECTION_STRING = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={DB_SERVER},{DB_PORT};"
        f"DATABASE={DB_NAME};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD};"
        f"Encrypt={DB_ENCRYPT};"
        f"TrustServerCertificate={DB_TRUST_SERVER_CERTIFICATE};"
    )
else:
    # Windows Integrated Authentication (Default for local development on Windows)
    ODBC_CONNECTION_STRING = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_NAME};"
        f"Trusted_Connection={DB_TRUSTED_CONNECTION};"
        f"TrustServerCertificate={DB_TRUST_SERVER_CERTIFICATE};"
    )

DATABASE_URL = (
    "mssql+pyodbc:///?odbc_connect="
    + quote_plus(ODBC_CONNECTION_STRING)
)

try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )
except Exception as e:
    logger.warning(f"Could not initialize SQL Server engine: {e}")
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