"""Authentication service — registration, login, JWT creation."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

import config
from database.repositories import user_repository

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(username: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=config.JWT_EXPIRY_HOURS)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
    except JWTError:
        return None


# ---------- Registration ----------

class RegistrationError(Exception):
    pass


def register_user(
    db: Session,
    username: str,
    password: str,
    confirm_password: str,
) -> dict:
    """Validate inputs, hash password, create USER account in SQL Server."""
    username = username.strip()
    if not username:
        raise RegistrationError("Username cannot be empty.")
    if not password:
        raise RegistrationError("Password cannot be empty.")
    if not confirm_password:
        raise RegistrationError("Confirm password cannot be empty.")
    if password != confirm_password:
        raise RegistrationError("Passwords do not match.")

    # Reserve admin username
    if username == config.ADMIN_USERNAME:
        raise RegistrationError("Username is not available.")

    if db is None:
        raise RegistrationError("Chức năng đăng ký tài khoản yêu cầu kết nối SQL Server. Database hiện đang ngoại tuyến.")

    existing = user_repository.find_user(db, username)
    if existing:
        raise RegistrationError("Username already exists.")

    pw_hash = hash_password(password)
    try:
        user = user_repository.create_user(db, username, pw_hash)
        db.commit()
        return user
    except Exception:
        db.rollback()
        raise


# ---------- Authentication ----------

class AuthenticationError(Exception):
    pass


def authenticate(
    db: Session,
    username: str,
    password: str,
) -> dict:
    """Returns dict with {username, role} or raises AuthenticationError."""
    username = username.strip()

    # Check admin first (credentials stored in .env, never in database)
    if username == config.ADMIN_USERNAME:
        if password == config.ADMIN_PASSWORD:
            return {"username": username, "role": "ADMIN"}
        raise AuthenticationError("Invalid username or password.")

    if db is None:
        raise AuthenticationError(
            "Cơ sở dữ liệu SQL Server đang ngoại tuyến. Vui lòng đăng nhập bằng tài khoản Quản trị viên (ADMIN) hoặc cấu hình database."
        )

    # Check normal USER in database
    try:
        user = user_repository.find_user(db, username)
    except Exception as e:
        raise AuthenticationError(f"Không thể kết nối đến cơ sở dữ liệu SQL Server: {e}")

    if not user:
        raise AuthenticationError("Invalid username or password.")
    if not verify_password(password, user["password_hash"]):
        raise AuthenticationError("Invalid username or password.")
    if user.get("status") != "active":
        raise AuthenticationError("Account is not active.")

    try:
        user_repository.update_last_login(db, username)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {"username": username, "role": user["role"]}
