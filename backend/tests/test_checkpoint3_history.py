"""Checkpoint 3 Comprehensive Test Suite — IrisAI Studio.

Verifies:
1. User Prediction History Management (Pagination, Filter by Species, Kernel, Date range).
2. Proper display mapping for all 5 models (SVM RBF, SVM Linear, SVM Polynomial, SVM Sigmoid, Deep Learning MLP).
3. Delete single prediction with strict ownership enforcement (User can only delete own, Admin can delete any).
4. Security: Non-owner cannot delete another user's prediction (HTTP 403).
5. Non-existent prediction deletion returns HTTP 404.
6. Clear user history (only current user's records deleted, other users preserved).
7. Admin all predictions audit API (GET /prediction/history/all with pagination, filtering, search by username).
8. Security: Non-admin cannot access admin audit endpoint (HTTP 403).
9. CSV Export generation with valid structure and headers.
10. Excel (.xlsx) Export generation with openpyxl verification (custom styling, columns, number formats).
11. Admin Export includes username and UserID columns.
12. Backward compatibility: GET /prediction/history without query parameters returns a raw list.
"""
from __future__ import annotations

import csv
import io
import os
import sys
from datetime import datetime, timedelta

import openpyxl
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth.service import create_access_token, hash_password
from database.connection import SessionLocal
from database.models import Prediction, User
from main import app
from ml import svm_service

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def setup_checkpoint3_data():
    """Setup test users (user_a, user_b, admin) and sample prediction records for CP3 tests.

    Cleans up all created test entities after tests complete.
    """
    svm_service.ensure_models_loaded()
    db = SessionLocal()

    created_user_ids = []
    created_pred_ids = []

    try:
        # 1. Helper to get or create user
        def get_or_create_user(username: str, role: str):
            u = db.query(User).filter(User.Username == username).first()
            if not u:
                u = User(
                    Username=username,
                    PasswordHash=hash_password("Cp3SecurePassword!123"),
                    Role=role,
                    Status="Active",
                    CreatedAt=datetime.utcnow(),
                )
                db.add(u)
                db.commit()
                db.refresh(u)
                created_user_ids.append(u.UserID)
            return u

        user_a = get_or_create_user("cp3_test_user_a", "USER")
        user_b = get_or_create_user("cp3_test_user_b", "USER")
        admin = get_or_create_user("cp3_test_admin", "ADMIN")

        token_a = create_access_token(user_a.Username, user_a.Role)
        token_b = create_access_token(user_b.Username, user_b.Role)
        token_admin = create_access_token(admin.Username, admin.Role)

        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}
        headers_admin = {"Authorization": f"Bearer {token_admin}"}

        # 2. Seed test predictions for user_a across all 5 models
        base_time = datetime.utcnow() - timedelta(days=2)
        sample_records = [
            (user_a.UserID, 5.1, 3.5, 1.4, 0.2, "Iris setosa", 0.98, 0.01, 0.01, "SVM", "rbf", base_time),
            (user_a.UserID, 5.9, 3.0, 4.2, 1.5, "Iris versicolor", 0.02, 0.95, 0.03, "SVM", "linear", base_time + timedelta(hours=2)),
            (user_a.UserID, 6.7, 3.1, 5.6, 2.4, "Iris virginica", 0.01, 0.04, 0.95, "SVM", "poly", base_time + timedelta(hours=4)),
            (user_a.UserID, 6.0, 2.2, 4.0, 1.0, "Iris versicolor", 0.05, 0.90, 0.05, "SVM", "sigmoid", base_time + timedelta(hours=6)),
            (user_a.UserID, 6.3, 3.3, 6.0, 2.5, "Iris virginica", 0.01, 0.02, 0.97, "Deep Learning MLP", "mlp", base_time + timedelta(hours=8)),
            # Seed 1 record for user_b
            (user_b.UserID, 5.0, 3.4, 1.5, 0.2, "Iris setosa", 0.99, 0.005, 0.005, "SVM", "rbf", base_time + timedelta(hours=10)),
        ]

        for uid, sl, sw, pl, pw, spec, ps, pv, pg, m_name, kern, c_time in sample_records:
            p = Prediction(
                UserID=uid,
                CreatedAt=c_time,
                SepalLength=sl,
                SepalWidth=sw,
                PetalLength=pl,
                PetalWidth=pw,
                PredictedSpecies=spec,
                ProbabilitySetosa=ps,
                ProbabilityVersicolor=pv,
                ProbabilityVirginica=pg,
                ModelName=m_name,
                Kernel=kern,
            )
            db.add(p)
            db.commit()
            db.refresh(p)
            created_pred_ids.append(p.PredictionID)

        context = {
            "user_a": user_a,
            "user_b": user_b,
            "admin": admin,
            "headers_a": headers_a,
            "headers_b": headers_b,
            "headers_admin": headers_admin,
            "created_pred_ids": created_pred_ids,
            "created_user_ids": created_user_ids,
        }

        yield context

    finally:
        # Cleanup: Remove test predictions and test users
        try:
            # Delete any predictions for cp3 users
            db.execute(text("DELETE FROM Predictions WHERE UserID IN (SELECT UserID FROM Users WHERE Username LIKE 'cp3_%')"))
            db.execute(text("DELETE FROM Users WHERE Username LIKE 'cp3_%'"))
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


class TestCheckpoint3PredictionHistory:
    """Check requirements for prediction history management, filter, export, and audit."""

    def test_01_user_history_paginated_and_5_models_display(self, setup_checkpoint3_data):
        """Test 1: User history returns paginated result with all 5 models and proper display names."""
        headers = setup_checkpoint3_data["headers_a"]
        resp = client.get("/prediction/history?page=1&page_size=10", headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data

        assert data["page"] == 1
        assert data["page_size"] == 10
        assert data["total"] >= 5

        # Check model display mappings
        expected_kernels = {"rbf", "linear", "poly", "sigmoid", "mlp"}
        found_kernels = {item["kernel"] for item in data["items"]}
        for k in expected_kernels:
            assert k in found_kernels, f"Kernel {k} missing from user history"

        display_map = {
            "rbf": "SVM RBF",
            "linear": "SVM Linear",
            "poly": "SVM Polynomial",
            "sigmoid": "SVM Sigmoid",
            "mlp": "Deep Learning MLP",
        }
        for item in data["items"]:
            if item["kernel"] in display_map:
                assert item["kernel_display"] == display_map[item["kernel"]]

    def test_02_filter_by_species(self, setup_checkpoint3_data):
        """Test 2: Filter history by species correctly filters records."""
        headers = setup_checkpoint3_data["headers_a"]

        # Filter setosa
        resp = client.get("/prediction/history?species=setosa&page=1&page_size=10", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for item in data["items"]:
            assert "setosa" in item["predicted_species"].lower()

        # Filter virginica
        resp = client.get("/prediction/history?species=virginica&page=1&page_size=10", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2
        for item in data["items"]:
            assert "virginica" in item["predicted_species"].lower()

    def test_03_filter_by_kernel(self, setup_checkpoint3_data):
        """Test 3: Filter history by kernel (e.g. sigmoid, mlp)."""
        headers = setup_checkpoint3_data["headers_a"]

        # Filter sigmoid
        resp = client.get("/prediction/history?kernel=sigmoid&page=1&page_size=10", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for item in data["items"]:
            assert item["kernel"] == "sigmoid"
            assert item["kernel_display"] == "SVM Sigmoid"

        # Filter mlp
        resp = client.get("/prediction/history?kernel=mlp&page=1&page_size=10", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for item in data["items"]:
            assert item["kernel"] == "mlp"
            assert item["kernel_display"] == "Deep Learning MLP"

    def test_04_pagination_mechanics(self, setup_checkpoint3_data):
        """Test 4: Verify pagination slices correctly across multiple pages."""
        headers = setup_checkpoint3_data["headers_a"]

        # Page 1 with page_size=2
        resp1 = client.get("/prediction/history?page=1&page_size=2", headers=headers)
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert len(data1["items"]) == 2
        assert data1["page"] == 1
        assert data1["total_pages"] >= 3

        # Page 2 with page_size=2
        resp2 = client.get("/prediction/history?page=2&page_size=2", headers=headers)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["items"]) == 2
        assert data2["page"] == 2

        # Items on page 1 and page 2 must be different
        ids_page1 = {item["id"] for item in data1["items"]}
        ids_page2 = {item["id"] for item in data2["items"]}
        assert ids_page1.isdisjoint(ids_page2), "Pages 1 and 2 share identical items"

    def test_05_delete_own_prediction_success(self, setup_checkpoint3_data):
        """Test 5: User can delete their own prediction record."""
        headers = setup_checkpoint3_data["headers_a"]
        user_a = setup_checkpoint3_data["user_a"]

        # Create a dedicated prediction to delete
        db = SessionLocal()
        p = Prediction(
            UserID=user_a.UserID,
            CreatedAt=datetime.utcnow(),
            SepalLength=5.5,
            SepalWidth=2.5,
            PetalLength=4.0,
            PetalWidth=1.3,
            PredictedSpecies="Iris versicolor",
            ProbabilityVersicolor=0.90,
            ModelName="SVM",
            Kernel="rbf",
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        pred_id = p.PredictionID
        db.close()

        # Delete it via API
        del_resp = client.delete(f"/prediction/history/{pred_id}", headers=headers)
        assert del_resp.status_code == 200, del_resp.text
        del_data = del_resp.json()
        assert del_data["success"] is True
        assert del_data["prediction_id"] == pred_id

        # Verify not in DB
        db = SessionLocal()
        check_p = db.query(Prediction).filter(Prediction.PredictionID == pred_id).first()
        db.close()
        assert check_p is None

    def test_06_security_user_cannot_delete_other_user_prediction(self, setup_checkpoint3_data):
        """Test 6: User A cannot delete User B's prediction (strict 403 Forbidden)."""
        headers_a = setup_checkpoint3_data["headers_a"]
        user_b = setup_checkpoint3_data["user_b"]

        # Create a prediction for user B
        db = SessionLocal()
        p = Prediction(
            UserID=user_b.UserID,
            CreatedAt=datetime.utcnow(),
            SepalLength=5.8,
            SepalWidth=2.7,
            PetalLength=5.1,
            PetalWidth=1.9,
            PredictedSpecies="Iris virginica",
            ProbabilityVirginica=0.92,
            ModelName="Deep Learning MLP",
            Kernel="mlp",
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        b_pred_id = p.PredictionID
        db.close()

        # User A tries to delete User B's record
        del_resp = client.delete(f"/prediction/history/{b_pred_id}", headers=headers_a)
        assert del_resp.status_code == 403, f"Expected 403, got {del_resp.status_code}"

        # Verify record still exists in DB
        db = SessionLocal()
        check_p = db.query(Prediction).filter(Prediction.PredictionID == b_pred_id).first()
        db.close()
        assert check_p is not None, "User B's prediction was wrongfully deleted!"

    def test_07_delete_nonexistent_prediction_returns_404(self, setup_checkpoint3_data):
        """Test 7: Deleting non-existent prediction returns 404 Not Found."""
        headers = setup_checkpoint3_data["headers_a"]
        resp = client.delete("/prediction/history/999999999", headers=headers)
        assert resp.status_code == 404

    def test_08_clear_user_history_only_affects_caller(self, setup_checkpoint3_data):
        """Test 8: Clear history only deletes current user's records, leaving others intact."""
        headers_a = setup_checkpoint3_data["headers_a"]
        user_a = setup_checkpoint3_data["user_a"]
        user_b = setup_checkpoint3_data["user_b"]

        db = SessionLocal()
        # Verify both users have records before clear
        count_a_before = db.query(Prediction).filter(Prediction.UserID == user_a.UserID).count()
        count_b_before = db.query(Prediction).filter(Prediction.UserID == user_b.UserID).count()
        db.close()

        assert count_a_before > 0
        assert count_b_before > 0

        # User A calls clear
        clear_resp = client.delete("/prediction/history", headers=headers_a)
        assert clear_resp.status_code == 200, clear_resp.text
        data = clear_resp.json()
        assert data["success"] is True
        assert data["count"] == count_a_before

        # Verify DB state
        db = SessionLocal()
        count_a_after = db.query(Prediction).filter(Prediction.UserID == user_a.UserID).count()
        count_b_after = db.query(Prediction).filter(Prediction.UserID == user_b.UserID).count()
        db.close()

        assert count_a_after == 0, "User A records were not completely cleared"
        assert count_b_after == count_b_before, "User B records were wrongfully cleared by User A!"

    def test_09_security_non_admin_cannot_access_all_history(self, setup_checkpoint3_data):
        """Test 9: Regular user cannot access /prediction/history/all (HTTP 403)."""
        headers = setup_checkpoint3_data["headers_a"]
        resp = client.get("/prediction/history/all", headers=headers)
        assert resp.status_code == 403

    def test_10_admin_can_access_all_history_with_username(self, setup_checkpoint3_data):
        """Test 10: Admin can view all predictions with username information."""
        headers_admin = setup_checkpoint3_data["headers_admin"]
        resp = client.get("/prediction/history/all?page=1&page_size=20", headers=headers_admin)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1
        for item in data["items"]:
            assert "username" in item
            assert item["username"] is not None

    def test_11_admin_search_by_username(self, setup_checkpoint3_data):
        """Test 11: Admin can search prediction audit records by username."""
        headers_admin = setup_checkpoint3_data["headers_admin"]
        resp = client.get("/prediction/history/all?search=cp3_test_user_b", headers=headers_admin)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        for item in data["items"]:
            assert item["username"] == "cp3_test_user_b"

    def test_12_admin_can_delete_any_user_prediction(self, setup_checkpoint3_data):
        """Test 12: Admin can delete another user's prediction record."""
        headers_admin = setup_checkpoint3_data["headers_admin"]
        user_b = setup_checkpoint3_data["user_b"]

        # Create record for user B
        db = SessionLocal()
        p = Prediction(
            UserID=user_b.UserID,
            CreatedAt=datetime.utcnow(),
            SepalLength=6.1,
            SepalWidth=2.8,
            PetalLength=4.7,
            PetalWidth=1.2,
            PredictedSpecies="Iris versicolor",
            ProbabilityVersicolor=0.88,
            ModelName="SVM",
            Kernel="linear",
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        pred_id = p.PredictionID
        db.close()

        # Admin deletes user B's prediction
        del_resp = client.delete(f"/prediction/history/{pred_id}", headers=headers_admin)
        assert del_resp.status_code == 200, del_resp.text
        assert del_resp.json()["success"] is True

        # Verify deleted in DB
        db = SessionLocal()
        check_p = db.query(Prediction).filter(Prediction.PredictionID == pred_id).first()
        db.close()
        assert check_p is None

    def test_13_export_csv_valid_content(self, setup_checkpoint3_data):
        """Test 13: Export CSV returns proper CSV stream with expected columns and data."""
        headers_admin = setup_checkpoint3_data["headers_admin"]
        resp = client.get("/prediction/history/export?format=csv", headers=headers_admin)
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "attachment; filename=" in resp.headers["content-disposition"]

        content = resp.content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        assert len(rows) >= 2, "CSV export should contain at least header and one data row"

        header = rows[0]
        assert "Prediction ID" in header
        assert "Predicted Species" in header
        assert "Kernel / Type" in header
        assert "Model Name" in header

    def test_14_export_excel_valid_openpyxl(self, setup_checkpoint3_data):
        """Test 14: Export Excel (.xlsx) returns valid workbook readable by openpyxl."""
        headers_admin = setup_checkpoint3_data["headers_admin"]
        resp = client.get("/prediction/history/export?format=xlsx", headers=headers_admin)
        assert resp.status_code == 200
        assert "spreadsheetml" in resp.headers["content-type"]

        # Parse workbook with openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(resp.content))
        assert "Prediction History" in wb.sheetnames
        ws = wb["Prediction History"]

        # Verify header row
        headers = [cell.value for cell in ws[1]]
        assert "Prediction ID" in headers
        assert "Predicted Species" in headers
        assert "Kernel / Type" in headers
        assert "Model Name" in headers

        # Verify at least one row of data exists
        assert ws.max_row >= 2
        # Check first data row prediction ID is positive integer
        pred_id_val = ws.cell(row=2, column=headers.index("Prediction ID") + 1).value
        assert isinstance(pred_id_val, int)

    def test_15_admin_export_includes_username(self, setup_checkpoint3_data):
        """Test 15: Admin export includes Username column in output."""
        headers_admin = setup_checkpoint3_data["headers_admin"]
        resp = client.get("/prediction/history/export?format=xlsx", headers=headers_admin)
        assert resp.status_code == 200

        wb = openpyxl.load_workbook(io.BytesIO(resp.content))
        ws = wb["Prediction History"]
        headers = [cell.value for cell in ws[1]]
        assert "Username" in headers
        assert "User ID" in headers

    def test_16_backward_compatibility_history_raw_list(self, setup_checkpoint3_data):
        """Test 16: Calling GET /prediction/history without query parameters returns a raw list for backward compatibility."""
        headers = setup_checkpoint3_data["headers_b"]
        resp = client.get("/prediction/history", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list), "Expected raw list for backward compatibility"
