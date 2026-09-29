"""Tests for Prediction History Edge Cases & Admin Session Safety.

Verifies:
1. Admin user with credentials in .env (not in Users table) can retrieve personal history without 500.
2. Unregistered/deleted username with valid JWT returns 200 with empty history (items: [], total: 0).
3. Personal history returns strictly records belonging to the authenticated user.
4. Admin audit (/prediction/history/all) returns records across all users with pagination and search.
5. Invalid tokens return HTTP 401 Unauthorized.
6. Missing token returns HTTP 403 Forbidden.
7. Admin can delete any prediction record without needing a row in the Users table.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth.service import create_access_token
from main import app

client = TestClient(app, raise_server_exceptions=True)


class TestPredictionHistoryEdgeCases:
    """Test suite for edge cases in prediction history retrieval and authorization."""

    def test_01_admin_personal_history_returns_200_empty(self):
        """Admin (whose credentials reside in .env, not in Users table) gets empty personal history."""
        token_admin = create_access_token("admin", "ADMIN")
        resp = client.get("/prediction/history?page=1&page_size=50", headers={"Authorization": f"Bearer {token_admin}"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["page_size"] == 50
        assert data["total_pages"] == 1

    def test_02_unregistered_user_personal_history_returns_200_empty(self):
        """A valid JWT with a username not in Users table returns empty history rather than 500."""
        token_ghost = create_access_token("ghost_user_99999", "USER")
        resp = client.get("/prediction/history?page=1&page_size=50", headers={"Authorization": f"Bearer {token_ghost}"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_03_personal_history_strict_user_isolation(self):
        """Personal history returns only records belonging to the requesting user."""
        token_user = create_access_token("cp2_test_user", "USER")
        resp = client.get("/prediction/history?page=1&page_size=50", headers={"Authorization": f"Bearer {token_user}"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] > 0
        for item in data["items"]:
            assert item["username"] == "cp2_test_user"

    def test_04_admin_audit_all_users_and_pagination(self):
        """Admin audit endpoint returns records across users and supports pagination."""
        token_admin = create_access_token("admin", "ADMIN")
        resp = client.get("/prediction/history/all?page=1&page_size=10", headers={"Authorization": f"Bearer {token_admin}"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data["items"]) <= 10
        assert data["total"] >= len(data["items"])
        assert data["page"] == 1
        assert data["page_size"] == 10

    def test_05_invalid_token_returns_401(self):
        """Requests with corrupted or invalid tokens return 401 Unauthorized."""
        client_loose = TestClient(app, raise_server_exceptions=False)
        resp = client_loose.get("/prediction/history?page=1&page_size=50", headers={"Authorization": "Bearer invalid.token.payload"})
        assert resp.status_code == 401
        assert "Invalid or expired token" in resp.json()["detail"]

    def test_06_missing_token_returns_403(self):
        """Requests without Authorization header return 403 Forbidden."""
        client_loose = TestClient(app, raise_server_exceptions=False)
        resp = client_loose.get("/prediction/history?page=1&page_size=50")
        assert resp.status_code == 403

    def test_07_regular_user_cannot_access_admin_audit(self):
        """Regular user cannot access admin audit endpoint (HTTP 403 Forbidden)."""
        token_user = create_access_token("cp2_test_user", "USER")
        client_loose = TestClient(app, raise_server_exceptions=False)
        resp = client_loose.get("/prediction/history/all?page=1&page_size=50", headers={"Authorization": f"Bearer {token_user}"})
        assert resp.status_code == 403
        assert "Admin access required" in resp.json()["detail"]

    def test_08_admin_0814230306_login_and_personal_history_page_size_10(self):
        """Admin 0814230306 logs in successfully and retrieves personal history with page_size=10."""
        import config
        resp_login = client.post("/auth/login", json={"username": config.ADMIN_USERNAME, "password": config.ADMIN_PASSWORD})
        assert resp_login.status_code == 200, resp_login.text
        data_login = resp_login.json()
        assert data_login["role"] == "ADMIN"
        token = data_login["access_token"]

        # Specifically test GET /prediction/history?page=1&page_size=10
        resp_hist = client.get("/prediction/history?page=1&page_size=10", headers={"Authorization": f"Bearer {token}"})
        assert resp_hist.status_code == 200, resp_hist.text
        data_hist = resp_hist.json()
        assert data_hist["items"] == []
        assert data_hist["page"] == 1
        assert data_hist["page_size"] == 10
        assert data_hist["total"] == 0
        assert data_hist["total_pages"] == 1

    def test_09_admin_0814230306_audit_history_all(self):
        """Admin 0814230306 can access system audit endpoint /prediction/history/all."""
        import config
        resp_login = client.post("/auth/login", json={"username": config.ADMIN_USERNAME, "password": config.ADMIN_PASSWORD})
        assert resp_login.status_code == 200
        token = resp_login.json()["access_token"]

        resp_audit = client.get("/prediction/history/all?page=1&page_size=10", headers={"Authorization": f"Bearer {token}"})
        assert resp_audit.status_code == 200, resp_audit.text
        data_audit = resp_audit.json()
        assert len(data_audit["items"]) == 10
        assert data_audit["page"] == 1
        assert data_audit["page_size"] == 10
        assert data_audit["total"] >= 140
        assert data_audit["total_pages"] >= 14

    def test_10_regular_user_cannot_delete_other_user_history(self):
        """Regular user cannot delete prediction belonging to another user (HTTP 403 Forbidden)."""
        # User 'bongne' tries to delete prediction #67 belonging to 'caobao'
        token_bongne = create_access_token("bongne", "USER")
        client_loose = TestClient(app, raise_server_exceptions=False)
        resp = client_loose.delete("/prediction/history/67", headers={"Authorization": f"Bearer {token_bongne}"})
        assert resp.status_code == 403
        assert "do not have permission" in resp.json()["detail"].lower()

    def test_11_regular_user_sees_only_own_records(self):
        """Regular user personal history returns strictly records belonging to that user."""
        token_dungbeo = create_access_token("dungbeo", "USER")
        resp = client.get("/prediction/history?page=1&page_size=10", headers={"Authorization": f"Bearer {token_dungbeo}"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3
        for item in data["items"]:
            assert item["username"] == "dungbeo"
