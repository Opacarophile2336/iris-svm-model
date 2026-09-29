"""SQL Server database connection for HMNC_PRO."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus

from config import (
    DB_DRIVER,
    DB_SERVER,
    DB_NAME,
    DB_TRUSTED_CONNECTION,
    DB_TRUST_SERVER_CERTIFICATE,
)


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


def get_db():
    """Provide a database session for FastAPI dependencies."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()