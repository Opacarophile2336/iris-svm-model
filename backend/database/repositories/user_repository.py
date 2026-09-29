"""Repository for User database operations."""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from database.models import User


def find_user(db: Session, username: str) -> Optional[dict]:
    """Find a user by username."""
    user = (
        db.query(User)
        .filter(User.Username == username)
        .first()
    )

    if user is None:
        return None

    return {
        "user_id": user.UserID,
        "username": user.Username,
        "password_hash": user.PasswordHash,
        "role": user.Role,
        "status": user.Status,
        "created_at": user.CreatedAt,
        "last_login": user.LastLogin,
    }


def create_user(
    db: Session,
    username: str,
    password_hash: str,
) -> dict:
    """Create a normal USER account."""
    user = User(
        Username=username,
        PasswordHash=password_hash,
        Role="USER",
        Status="active",
        CreatedAt=datetime.utcnow(),
        LastLogin=None,
    )

    db.add(user)
    db.flush()
    #db.commit()
    db.refresh(user)

    return {
        "user_id": user.UserID,
        "username": user.Username,
        "password_hash": user.PasswordHash,
        "role": user.Role,
        "status": user.Status,
        "created_at": user.CreatedAt,
        "last_login": user.LastLogin,
    }


def update_last_login(db: Session, username: str) -> bool:
    """Update the last login timestamp for a user."""
    user = (
        db.query(User)
        .filter(User.Username == username)
        .first()
    )

    if user is None:
        return False

    user.LastLogin = datetime.utcnow()
    db.flush()
    #db.commit()

    return True