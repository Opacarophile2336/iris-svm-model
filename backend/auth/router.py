"""Auth API router — /auth/register, /auth/login, /auth/me."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import service
from auth.dependencies import get_current_user
from database.connection import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    req: RegisterRequest,
    db: Session = Depends(get_db),
):
    try:
        user = service.register_user(
            db,
            req.username,
            req.password,
            req.confirm_password,
        )
        return {
            "message": "Registration successful.",
            "username": user["username"],
        }
    except service.RegistrationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login", response_model=TokenResponse)
def login(
    req: LoginRequest,
    db: Session = Depends(get_db),
):
    try:
        user_info = service.authenticate(
            db,
            req.username,
            req.password,
        )
    except service.AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )

    token = service.create_access_token(
        user_info["username"],
        user_info["role"],
    )

    return TokenResponse(
        access_token=token,
        role=user_info["role"],
        username=user_info["username"],
    )


@router.get("/me")
def me(current: dict = Depends(get_current_user)):
    return {
        "username": current["username"],
        "role": current["role"],
    }