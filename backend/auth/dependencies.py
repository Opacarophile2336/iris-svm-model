"""FastAPI dependency functions for authentication and authorization."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from auth.service import decode_token

bearer = HTTPBearer()


def _get_payload(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    token = credentials.credentials
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def get_current_user(payload: dict = Depends(_get_payload)) -> dict:
    """Returns {username, role} for any authenticated user."""
    return {"username": payload["sub"], "role": payload["role"]}


def require_user(current: dict = Depends(get_current_user)) -> dict:
    """Allow any authenticated user (USER or ADMIN)."""
    return current


def require_admin(current: dict = Depends(get_current_user)) -> dict:
    """Allow only ADMIN role."""
    if current["role"] != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current
