"""Checkpoint 19.9: Verification of Authentication and Users in SQL Server."""
import time
from datetime import datetime
import pytest
from fastapi.testclient import TestClient

from main import app
from database.connection import SessionLocal
from database.models import User, Prediction
import config

client = TestClient(app)

TEST_USERNAME = "__sql_auth_test_user__"
TEST_PASSWORD = "AuthTestPassword123!"


def setup_function():
    # Ensure clean state for test user before test starts
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.Username == TEST_USERNAME).first()
        if user:
            db.query(Prediction).filter(Prediction.UserID == user.UserID).delete()
            db.delete(user)
            db.commit()
    finally:
        db.close()


def teardown_function():
    # Cleanup after test
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.Username == TEST_USERNAME).first()
        if user:
            db.query(Prediction).filter(Prediction.UserID == user.UserID).delete()
            db.delete(user)
            db.commit()
    finally:
        db.close()


def test_full_auth_flow_sql_server():
    db = SessionLocal()
    try:
        # 1. Password mismatch rejection
        r_mismatch = client.post("/auth/register", json={
            "username": TEST_USERNAME,
            "password": TEST_PASSWORD,
            "confirm_password": "WrongPassword999!",
        })
        assert r_mismatch.status_code == 400
        assert "do not match" in r_mismatch.json()["detail"].lower()
        # Verify not in DB
        assert db.query(User).filter(User.Username == TEST_USERNAME).first() is None

        # 2. Successful Registration
        r_reg = client.post("/auth/register", json={
            "username": TEST_USERNAME,
            "password": TEST_PASSWORD,
            "confirm_password": TEST_PASSWORD,
        })
        assert r_reg.status_code == 201
        assert r_reg.json()["username"] == TEST_USERNAME

        # 3. Query SQL Server Users table directly
        user_db = db.query(User).filter(User.Username == TEST_USERNAME).first()
        assert user_db is not None
        assert user_db.Username == TEST_USERNAME
        assert user_db.Role == "USER"
        assert user_db.Status == "active"
        assert isinstance(user_db.CreatedAt, datetime)
        assert user_db.LastLogin is None  # Initial state is NULL
        # Verify password hash (never plaintext)
        assert user_db.PasswordHash != TEST_PASSWORD
        assert user_db.PasswordHash.startswith("$2b$") or user_db.PasswordHash.startswith("$2a$")

        # 4. Duplicate Registration rejection
        r_dup = client.post("/auth/register", json={
            "username": TEST_USERNAME,
            "password": TEST_PASSWORD,
            "confirm_password": TEST_PASSWORD,
        })
        assert r_dup.status_code == 400
        assert "already exists" in r_dup.json()["detail"].lower()
        # Ensure only 1 row exists
        count = db.query(User).filter(User.Username == TEST_USERNAME).count()
        assert count == 1

        # 5. Login
        time.sleep(0.05)
        r_login = client.post("/auth/login", json={
            "username": TEST_USERNAME,
            "password": TEST_PASSWORD,
        })
        assert r_login.status_code == 200
        login_data = r_login.json()
        assert login_data["username"] == TEST_USERNAME
        assert login_data["role"] == "USER"
        user_token = login_data["access_token"]
        assert len(user_token) > 20

        # 6. Verify LastLogin updated in SQL Server
        db.refresh(user_db)
        assert user_db.LastLogin is not None
        assert isinstance(user_db.LastLogin, datetime)

        # 7. Check /auth/me with User token
        r_me = client.get("/auth/me", headers={"Authorization": f"Bearer {user_token}"})
        assert r_me.status_code == 200
        assert r_me.json()["username"] == TEST_USERNAME
        assert r_me.json()["role"] == "USER"

        # 8. Check ADMIN Login and /auth/me
        r_admin_login = client.post("/auth/login", json={
            "username": config.ADMIN_USERNAME,
            "password": config.ADMIN_PASSWORD,
        })
        assert r_admin_login.status_code == 200
        admin_data = r_admin_login.json()
        assert admin_data["username"] == config.ADMIN_USERNAME
        assert admin_data["role"] == "ADMIN"
        admin_token = admin_data["access_token"]

        r_admin_me = client.get("/auth/me", headers={"Authorization": f"Bearer {admin_token}"})
        assert r_admin_me.status_code == 200
        assert r_admin_me.json()["username"] == config.ADMIN_USERNAME
        assert r_admin_me.json()["role"] == "ADMIN"

        # 9. Verify ADMIN is not in Users table
        admin_in_db = db.query(User).filter(User.Username == config.ADMIN_USERNAME).first()
        assert admin_in_db is None

        # 10. Check Authorization (USER cannot access ADMIN endpoints)
        r_user_admin_ep = client.get("/ml/evaluation", headers={"Authorization": f"Bearer {user_token}"})
        assert r_user_admin_ep.status_code == 403

        r_admin_admin_ep = client.get("/ml/evaluation", headers={"Authorization": f"Bearer {admin_token}"})
        assert r_admin_admin_ep.status_code == 200

        # 11. Cleanup user test
        db.delete(user_db)
        db.commit()

        # 12. Verify cleanup
        assert db.query(User).filter(User.Username == TEST_USERNAME).first() is None
    finally:
        db.close()
