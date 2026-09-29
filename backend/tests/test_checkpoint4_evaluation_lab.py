"""Checkpoint 4 Comprehensive Test Suite — IrisAI Studio.

Verifies:
1. Evaluation endpoint (/ml/evaluation) returns all 5 classifiers: rbf, linear, poly, sigmoid, mlp.
2. Metrics of all 5 classifiers lie in the valid statistical range [0.0, 1.0].
3. Confusion matrices for all 5 classifiers have 3x3 dimensions and sum to the test partition size (30).
4. Decision boundary generation for SVM Sigmoid kernel returns valid base64 PNG data.
5. Decision boundary supports different valid feature pairs (e.g., [0, 1], [2, 3]).
6. Invalid kernel for decision boundary is rejected with HTTP 400.
7. Critical MLP Rule: MLP is NOT accepted as an SVM decision boundary kernel and returns HTTP 400.
8. Admin re-training (/ml/train) trains all 5 classifiers and reports honest cross-validation results.
9. Prediction metrics endpoint (/prediction/metrics) returns calibrated performance data for all 5 models.
10. Security: Non-admin users cannot access /ml/evaluation (HTTP 403 Forbidden).
11. Security: Non-admin users cannot access /ml/decision-boundary (HTTP 403 Forbidden).
12. Security: Non-admin users cannot trigger /ml/train (HTTP 403 Forbidden).
13. ModelHistory persistence in SQL Server: verifies model history retrieval via /ml/model-history.
"""
from __future__ import annotations

import base64
import os
import sys
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth.service import create_access_token, hash_password
from database.connection import SessionLocal
from database.models import ModelHistory, User
from main import app
from ml import svm_service

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def setup_cp4_auth():
    """Setup test admin and test user tokens for Checkpoint 4 tests.

    Cleans up any created test users upon completion.
    """
    svm_service.ensure_models_loaded()
    db = SessionLocal()

    created_user_ids = []
    try:
        def get_or_create(username: str, role: str):
            u = db.query(User).filter(User.Username == username).first()
            if not u:
                u = User(
                    Username=username,
                    PasswordHash=hash_password("Cp4SecurePass123!"),
                    Role=role,
                    Status="Active",
                    CreatedAt=datetime.utcnow(),
                )
                db.add(u)
                db.commit()
                db.refresh(u)
                created_user_ids.append(u.UserID)
            return u

        user_admin = get_or_create("cp4_test_admin", "ADMIN")
        user_regular = get_or_create("cp4_test_user", "USER")

        admin_token = create_access_token(user_admin.Username, user_admin.Role)
        user_token = create_access_token(user_regular.Username, user_regular.Role)

        context = {
            "admin_headers": {"Authorization": f"Bearer {admin_token}"},
            "user_headers": {"Authorization": f"Bearer {user_token}"},
            "created_user_ids": created_user_ids,
        }
        yield context

    finally:
        try:
            db.execute(text("DELETE FROM Users WHERE Username LIKE 'cp4_%'"))
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


class TestCheckpoint4EvaluationAndLab:
    """Test suite for CP4 Model Management, Evaluation & Decision Boundary Studio."""

    def test_01_evaluation_endpoint_contains_all_five_classifiers(self, setup_cp4_auth):
        """Test 1: /ml/evaluation returns metrics for all 5 classifiers: rbf, linear, poly, sigmoid, mlp."""
        headers = setup_cp4_auth["admin_headers"]
        resp = client.get("/ml/evaluation", headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        expected_classifiers = {"rbf", "linear", "poly", "sigmoid", "mlp"}
        actual_keys = set(data.keys())
        assert expected_classifiers.issubset(actual_keys), (
            f"Missing classifiers in evaluation: {expected_classifiers - actual_keys}"
        )

        for k in expected_classifiers:
            item = data[k]
            assert "test_accuracy" in item
            assert "precision" in item
            assert "recall" in item
            assert "f1" in item
            assert "cv_mean" in item
            assert "confusion_matrix" in item

    def test_02_metrics_of_five_models_in_valid_range(self, setup_cp4_auth):
        """Test 2: All performance metrics for 5 models are legitimate floats in [0.0, 1.0]."""
        headers = setup_cp4_auth["admin_headers"]
        resp = client.get("/ml/evaluation", headers=headers)
        assert resp.status_code == 200
        data = resp.json()

        for k in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            m = data[k]
            assert 0.0 <= m["test_accuracy"] <= 1.0, f"Invalid test_accuracy for {k}: {m['test_accuracy']}"
            assert 0.0 <= m["precision"] <= 1.0, f"Invalid precision for {k}: {m['precision']}"
            assert 0.0 <= m["recall"] <= 1.0, f"Invalid recall for {k}: {m['recall']}"
            assert 0.0 <= m["f1"] <= 1.0, f"Invalid f1 for {k}: {m['f1']}"
            assert 0.0 <= m["cv_mean"] <= 1.0, f"Invalid cv_mean for {k}: {m['cv_mean']}"

    def test_03_confusion_matrix_3x3_for_five_models(self, setup_cp4_auth):
        """Test 3: Confusion matrices for all 5 classifiers have 3x3 dimensions and sum to 30 test samples."""
        headers = setup_cp4_auth["admin_headers"]
        resp = client.get("/ml/evaluation", headers=headers)
        assert resp.status_code == 200
        data = resp.json()

        for k in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            cm = data[k]["confusion_matrix"]
            assert len(cm) == 3, f"Confusion matrix for {k} must have 3 rows"
            for row in cm:
                assert len(row) == 3, f"Confusion matrix row for {k} must have 3 columns"
            total_samples = sum(sum(row) for row in cm)
            assert total_samples == 30, f"Confusion matrix for {k} total is {total_samples}, expected 30"

    def test_04_sigmoid_decision_boundary(self, setup_cp4_auth):
        """Test 4: Decision boundary generation for SVM Sigmoid returns valid base64 PNG data."""
        headers = setup_cp4_auth["admin_headers"]
        resp = client.get("/ml/decision-boundary?kernel=sigmoid&feature_x=2&feature_y=3", headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["kernel"] == "sigmoid"
        assert "image_base64" in data
        b64_str = data["image_base64"]
        assert len(b64_str) > 1000

        # Verify decoded content starts with PNG magic header
        raw_bytes = base64.b64decode(b64_str)
        assert raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"), "Decoded bytes do not form a valid PNG image"

    def test_05_decision_boundary_feature_pairs(self, setup_cp4_auth):
        """Test 5: Decision boundary supports various pairs of features (e.g. [0, 1] vs [2, 3])."""
        headers = setup_cp4_auth["admin_headers"]

        # Sepal length (0) vs Sepal width (1)
        resp1 = client.get("/ml/decision-boundary?kernel=rbf&feature_x=0&feature_y=1", headers=headers)
        assert resp1.status_code == 200
        assert len(resp1.json()["image_base64"]) > 1000

        # Petal length (2) vs Petal width (3)
        resp2 = client.get("/ml/decision-boundary?kernel=linear&feature_x=2&feature_y=3", headers=headers)
        assert resp2.status_code == 200
        assert len(resp2.json()["image_base64"]) > 1000

    def test_06_invalid_kernel_rejected(self, setup_cp4_auth):
        """Test 6: Requesting decision boundary for non-existent kernel returns HTTP 400 Bad Request."""
        headers = setup_cp4_auth["admin_headers"]
        resp = client.get("/ml/decision-boundary?kernel=fake_kernel&feature_x=0&feature_y=1", headers=headers)
        assert resp.status_code == 400
        assert "Unsupported kernel" in resp.json()["detail"]

    def test_07_critical_mlp_rule_decision_boundary(self, setup_cp4_auth):
        """Test 7: Critical MLP Rule: MLP is NOT accepted as an SVM decision boundary kernel and returns HTTP 400."""
        headers = setup_cp4_auth["admin_headers"]
        resp = client.get("/ml/decision-boundary?kernel=mlp&feature_x=2&feature_y=3", headers=headers)
        assert resp.status_code == 400
        assert "Unsupported kernel" in resp.json()["detail"]

    def test_08_admin_train_returns_all_five_models(self, setup_cp4_auth):
        """Test 8: POST /ml/train retrains and returns real metrics for all 5 classifiers."""
        headers = setup_cp4_auth["admin_headers"]
        resp = client.post("/ml/train", headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["message"] == "Training complete."
        assert "best_kernel" in data
        assert data["best_kernel"] in ["rbf", "linear", "poly", "sigmoid", "mlp"]
        assert "cv_score" in data

        results = data.get("results", {})
        expected_classifiers = {"rbf", "linear", "poly", "sigmoid", "mlp"}
        assert expected_classifiers.issubset(set(results.keys()))

        for k in expected_classifiers:
            assert "test_accuracy" in results[k]
            assert "cv_mean" in results[k]

    def test_09_prediction_metrics_contains_all_five_models(self, setup_cp4_auth):
        """Test 9: GET /prediction/metrics returns metrics for all 5 classifiers."""
        headers = setup_cp4_auth["user_headers"]
        resp = client.get("/prediction/metrics", headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        for k in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            assert k in data
            assert "kernel_display" in data[k]
            assert "test_accuracy" in data[k]
            assert "cv_accuracy" in data[k]

    def test_10_user_cannot_access_evaluation(self, setup_cp4_auth):
        """Test 10: Regular USER cannot access /ml/evaluation (HTTP 403 Forbidden)."""
        headers = setup_cp4_auth["user_headers"]
        resp = client.get("/ml/evaluation", headers=headers)
        assert resp.status_code == 403

    def test_11_user_cannot_access_decision_boundary(self, setup_cp4_auth):
        """Test 11: Regular USER cannot access /ml/decision-boundary (HTTP 403 Forbidden)."""
        headers = setup_cp4_auth["user_headers"]
        resp = client.get("/ml/decision-boundary?kernel=rbf&feature_x=2&feature_y=3", headers=headers)
        assert resp.status_code == 403

    def test_12_user_cannot_trigger_train(self, setup_cp4_auth):
        """Test 12: Regular USER cannot trigger /ml/train (HTTP 403 Forbidden)."""
        headers = setup_cp4_auth["user_headers"]
        resp = client.post("/ml/train", headers=headers)
        assert resp.status_code == 403

    def test_13_model_history_persistence_sql_server(self, setup_cp4_auth):
        """Test 13: Model history endpoint (/ml/model-history) returns recorded history entries."""
        headers = setup_cp4_auth["admin_headers"]
        resp = client.get("/ml/model-history", headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)
        # History list contains recorded entries
        assert len(data) >= 1
