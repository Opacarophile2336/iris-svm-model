"""Backend test suite.

Uses FastAPI's TestClient. Tests: registration, login, authorization, prediction, kernels, history.
"""
from __future__ import annotations

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Patch config paths before importing anything else
import config
config.USER_ACCOUNTS_PATH = config.PROJECT_ROOT / "data" / "test_user_accounts.xlsx"
config.PREDICTION_HISTORY_PATH = config.PROJECT_ROOT / "data" / "test_prediction_history.json"
config.TRAINED_MODEL_PATH = config.PROJECT_ROOT / "data" / "test_best_model.joblib"
config.MODEL_HISTORY_PATH = config.PROJECT_ROOT / "data" / "test_model_history.json"

import pytest
from fastapi.testclient import TestClient
from main import app
from storage.excel_repository import init_excel_file
from ml.svm_service import train_all_kernels, load_best_model

# Ensure fresh test storage and model are initialized
if config.USER_ACCOUNTS_PATH.exists():
    try:
        config.USER_ACCOUNTS_PATH.unlink()
    except Exception:
        pass
init_excel_file()

# Clean up transient test users from previous test runs in SQL Server
try:
    from database.connection import SessionLocal
    from database.models import User, Prediction
    _cleanup_db = SessionLocal()
    _test_usernames = [
        "testuser_001", "testuser_dup", "testuser_mm", "testuser_ep",
        "loginuser", "authuser", "preduser", "kerneluser",
        "user_a_iso", "user_b_iso", "mk_user", "batch_user", "insights_user"
    ]
    _matched_users = _cleanup_db.query(User).filter(User.Username.in_(_test_usernames)).all()
    _matched_ids = [u.UserID for u in _matched_users]
    if _matched_ids:
        _cleanup_db.query(Prediction).filter(Prediction.UserID.in_(_matched_ids)).delete(synchronize_session=False)
        _cleanup_db.query(User).filter(User.UserID.in_(_matched_ids)).delete(synchronize_session=False)
        _cleanup_db.commit()
    _cleanup_db.close()
except Exception:
    pass

if not load_best_model():
    train_all_kernels(save=True)

# Use httpx transport backend compatible with newer starlette
client = TestClient(app, raise_server_exceptions=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def register(username, password, confirm=None):
    return client.post("/auth/register", json={
        "username": username,
        "password": password,
        "confirm_password": confirm or password,
    })


def login(username, password):
    return client.post("/auth/login", json={"username": username, "password": password})


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}


# ──────────────────────────────────────────────────────────────────────────────
# Registration Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRegistration:
    def test_new_user_success(self):
        r = register("testuser_001", "SecurePass123")
        assert r.status_code == 201, r.text
        assert "Registration successful" in r.json()["message"]

    def test_duplicate_user_rejected(self):
        register("testuser_dup", "SecurePass123")
        r = register("testuser_dup", "SecurePass123")
        assert r.status_code == 400
        assert "already exists" in r.json()["detail"].lower()

    def test_password_mismatch_rejected(self):
        r = register("testuser_mm", "Pass123", "Different456")
        assert r.status_code == 400
        assert "do not match" in r.json()["detail"].lower()

    def test_empty_username_rejected(self):
        r = register("", "Pass123")
        assert r.status_code == 400

    def test_empty_password_rejected(self):
        r = register("testuser_ep", "")
        assert r.status_code == 400

    def test_admin_username_reserved(self):
        r = register("0814230306", "SomePass")
        assert r.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# Login Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLogin:
    def setup_method(self):
        register("loginuser", "TestPass123")

    def test_correct_credentials_success(self):
        r = login("loginuser", "TestPass123")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "access_token" in data
        assert data["role"] == "USER"

    def test_wrong_password_rejected(self):
        r = login("loginuser", "WrongPassword")
        assert r.status_code == 401

    def test_unknown_user_rejected(self):
        r = login("nonexistentuser99", "AnyPass")
        assert r.status_code == 401

    def test_admin_login_success(self):
        r = login("0814230306", "0814230306")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["role"] == "ADMIN"
        assert data["username"] == "0814230306"


# ──────────────────────────────────────────────────────────────────────────────
# Authorization Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAuthorization:
    def setup_method(self):
        register("authuser", "AuthPass123")
        user_r = login("authuser", "AuthPass123")
        self.user_token = user_r.json()["access_token"]
        admin_r = login("0814230306", "0814230306")
        self.admin_token = admin_r.json()["access_token"]

    def test_user_can_access_datasets(self):
        r = client.get("/datasets/iris", headers=auth_header(self.user_token))
        assert r.status_code == 200

    def test_user_cannot_train_model(self):
        r = client.post("/ml/train", headers=auth_header(self.user_token))
        assert r.status_code == 403

    def test_user_cannot_access_model_history(self):
        r = client.get("/ml/model-history", headers=auth_header(self.user_token))
        assert r.status_code == 403

    def test_user_cannot_access_evaluation(self):
        r = client.get("/ml/evaluation", headers=auth_header(self.user_token))
        assert r.status_code == 403

    def test_user_cannot_access_decision_boundary(self):
        r = client.get("/ml/decision-boundary", headers=auth_header(self.user_token))
        assert r.status_code == 403

    def test_admin_can_train_model(self):
        r = client.post("/ml/train", headers=auth_header(self.admin_token))
        assert r.status_code == 200

    def test_admin_can_access_model_history(self):
        r = client.get("/ml/model-history", headers=auth_header(self.admin_token))
        assert r.status_code == 200

    def test_admin_can_access_all_history(self):
        r = client.get("/prediction/history/all", headers=auth_header(self.admin_token))
        assert r.status_code == 200

    def test_user_cannot_access_all_history(self):
        r = client.get("/prediction/history/all", headers=auth_header(self.user_token))
        assert r.status_code == 403

    def test_unauthenticated_rejected(self):
        r = client.get("/datasets/iris")
        assert r.status_code in (401, 403)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPrediction:
    def setup_method(self):
        register("preduser", "PredPass123")
        r = login("preduser", "PredPass123")
        self.token = r.json()["access_token"]

    def test_valid_setosa_prediction(self):
        r = client.post("/prediction/predict", headers=auth_header(self.token), json={
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "predicted_species" in data
        assert "probabilities" in data
        assert "confidence" in data
        assert data["predicted_species"] == "Iris setosa"

    def test_valid_virginica_prediction(self):
        r = client.post("/prediction/predict", headers=auth_header(self.token), json={
            "sepal_length": 6.3, "sepal_width": 3.3,
            "petal_length": 6.0, "petal_width": 2.5,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["predicted_species"] == "Iris virginica"

    def test_probabilities_sum_to_100(self):
        r = client.post("/prediction/predict", headers=auth_header(self.token), json={
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        })
        data = r.json()
        probs = data["probabilities"]
        total = probs["setosa"] + probs["versicolor"] + probs["virginica"]
        assert abs(total - 100) < 1.0

    def test_negative_value_rejected(self):
        r = client.post("/prediction/predict", headers=auth_header(self.token), json={
            "sepal_length": -1.0, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        })
        assert r.status_code == 422

    def test_kernel_recorded_in_history(self):
        client.post("/prediction/predict", headers=auth_header(self.token), json={
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        })
        r = client.get("/prediction/history", headers=auth_header(self.token))
        assert r.status_code == 200
        history = r.json()
        assert len(history) > 0
        assert "kernel" in history[0]
        assert history[0]["kernel"] in ["rbf", "linear", "poly"]


# ──────────────────────────────────────────────────────────────────────────────
# Kernel Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestKernels:
    def setup_method(self):
        register("kerneluser", "KernelPass123")
        r = login("kerneluser", "KernelPass123")
        self.token = r.json()["access_token"]

    def test_supported_kernels(self):
        r = client.get("/ml/kernels", headers=auth_header(self.token))
        assert r.status_code == 200
        kernels = r.json()["kernels"]
        assert "sigmoid" in kernels
        assert "rbf" in kernels
        assert "linear" in kernels
        assert "poly" in kernels

    def test_decision_boundary_supports_sigmoid(self):
        admin_r = login("0814230306", "0814230306")
        admin_token = admin_r.json()["access_token"]
        r = client.get("/ml/decision-boundary?kernel=sigmoid", headers=auth_header(admin_token))
        assert r.status_code == 200
        assert "image_base64" in r.json()

    def test_decision_boundary_rejects_invalid_kernel(self):
        admin_r = login("0814230306", "0814230306")
        admin_token = admin_r.json()["access_token"]
        r = client.get("/ml/decision-boundary?kernel=invalid_kernel", headers=auth_header(admin_token))
        assert r.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# History Isolation Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHistoryIsolation:
    def setup_method(self):
        register("user_a_iso", "PassA123")
        register("user_b_iso", "PassB123")
        ra = login("user_a_iso", "PassA123")
        rb = login("user_b_iso", "PassB123")
        self.token_a = ra.json()["access_token"]
        self.token_b = rb.json()["access_token"]

    def test_user_sees_only_own_history(self):
        # user_a makes a prediction
        client.post("/prediction/predict", headers=auth_header(self.token_a), json={
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        })
        # user_b checks their history — should not see user_a's record
        r = client.get("/prediction/history", headers=auth_header(self.token_b))
        assert r.status_code == 200
        for record in r.json():
            assert record.get("username") != "user_a_iso"


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Kernel (RBF, Linear, Poly, Sigmoid, MLP) Prediction Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiKernelPrediction:
    def setup_method(self):
        register("mk_user", "MkPass123")
        r = login("mk_user", "MkPass123")
        self.token = r.json()["access_token"]

    def test_predict_all_returns_three_kernels(self):
        r = client.post("/prediction/predict-all", headers=auth_header(self.token), json={
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "results" in data
        assert "rbf" in data["results"]
        assert "linear" in data["results"]
        assert "poly" in data["results"]
        assert "sigmoid" in data["results"]
        assert "mlp" in data["results"]

        for k in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            res = data["results"][k]
            assert "predicted_species" in res
            assert "confidence" in res
            assert "accuracy" in res
            assert res["accuracy"] >= 0.0  # Real validated accuracy
            assert res["confidence"] > 0

    def test_predict_all_consensus(self):
        r = client.post("/prediction/predict-all", headers=auth_header(self.token), json={
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        })
        data = r.json()
        assert data["consensus"] in ["unanimous", "majority", "divergent"]
        assert "highest_confidence" in data
        assert data["highest_confidence"]["kernel"] in ["rbf", "linear", "poly", "sigmoid", "mlp"]


# ──────────────────────────────────────────────────────────────────────────────
# Batch Upload & Export Tests (.csv, .txt, .xlsx)
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchUploadAndExport:
    def setup_method(self):
        register("batch_user", "BatchPass123")
        r = login("batch_user", "BatchPass123")
        self.token = r.json()["access_token"]

    def test_upload_valid_csv(self):
        csv_content = (
            "Sepal Length,Sepal Width,Petal Length,Petal Width\n"
            "5.1,3.5,1.4,0.2\n"
            "6.2,2.9,4.3,1.3\n"
            "7.1,3.0,5.9,2.1\n"
        )
        files = {"file": ("test_samples.csv", csv_content.encode("utf-8"), "text/csv")}
        r = client.post("/prediction/batch-upload", headers=auth_header(self.token), files=files)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["total_rows"] == 3
        assert len(data["results"]) == 3
        # Verify predictions across all 3 kernels
        row0 = data["results"][0]
        assert "rbf_prediction" in row0
        assert "linear_prediction" in row0
        assert "poly_prediction" in row0
        assert "rbf_accuracy" in row0

    def test_upload_valid_txt(self):
        txt_content = (
            "sepal_length\tsepal_width\tpetal_length\tpetal_width\n"
            "5.0\t3.4\t1.5\t0.2\n"
            "6.0\t2.7\t5.1\t1.6\n"
        )
        files = {"file": ("test_samples.txt", txt_content.encode("utf-8"), "text/plain")}
        r = client.post("/prediction/batch-upload", headers=auth_header(self.token), files=files)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["total_rows"] == 2

    def test_upload_valid_xlsx(self):
        import io
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.append(["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"])
        ws.append([5.1, 3.5, 1.4, 0.2])
        ws.append([6.7, 3.1, 4.7, 1.5])
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        files = {"file": ("test_samples.xlsx", buf.read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        r = client.post("/prediction/batch-upload", headers=auth_header(self.token), files=files)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["total_rows"] == 2

    def test_upload_missing_column_rejected(self):
        csv_content = "Sepal Length,Sepal Width,Petal Length\n5.1,3.5,1.4\n"
        files = {"file": ("bad_samples.csv", csv_content.encode("utf-8"), "text/csv")}
        r = client.post("/prediction/batch-upload", headers=auth_header(self.token), files=files)
        assert r.status_code == 400
        assert "could not find required columns" in r.json()["detail"].lower()

    def test_upload_unsupported_format_rejected(self):
        files = {"file": ("test.pdf", b"%PDF-1.4...", "application/pdf")}
        r = client.post("/prediction/batch-upload", headers=auth_header(self.token), files=files)
        assert r.status_code == 400
        assert "unsupported file" in r.json()["detail"].lower()

    def test_export_xlsx(self):
        # First upload a file to populate batch cache
        csv_content = "Sepal Length,Sepal Width,Petal Length,Petal Width\n5.1,3.5,1.4,0.2\n"
        files = {"file": ("test.csv", csv_content.encode("utf-8"), "text/csv")}
        up_r = client.post("/prediction/batch-upload", headers=auth_header(self.token), files=files)
        batch_id = up_r.json()["batch_id"]

        export_r = client.post("/prediction/batch-export", headers=auth_header(self.token), json={
            "batch_id": batch_id,
            "format": "xlsx",
        })
        assert export_r.status_code == 200
        assert "spreadsheetml" in export_r.headers["content-type"]
        assert len(export_r.content) > 1000

    def test_export_csv(self):
        csv_content = "Sepal Length,Sepal Width,Petal Length,Petal Width\n5.1,3.5,1.4,0.2\n"
        files = {"file": ("test.csv", csv_content.encode("utf-8"), "text/csv")}
        up_r = client.post("/prediction/batch-upload", headers=auth_header(self.token), files=files)
        batch_id = up_r.json()["batch_id"]

        export_r = client.post("/prediction/batch-export", headers=auth_header(self.token), json={
            "batch_id": batch_id,
            "format": "csv",
        })
        assert export_r.status_code == 200
        assert "text/csv" in export_r.headers["content-type"]
        assert b"RBF Prediction" in export_r.content


# ──────────────────────────────────────────────────────────────────────────────
# Model Insights Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestModelInsights:
    def setup_method(self):
        register("insights_user", "InPass123")
        r = login("insights_user", "InPass123")
        self.token = r.json()["access_token"]

    def test_insights_generation(self):
        r = client.post("/prediction/insights", headers=auth_header(self.token), json={
            "sepal_length": 5.1, "sepal_width": 3.5,
            "petal_length": 1.4, "petal_width": 0.2,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "insights" in data
        ins = data["insights"]
        assert "species_alignment" in ins
        assert "discriminant_notes" in ins
        assert "closest_botanical_match" in ins
        assert ins["closest_botanical_match"] == "Iris setosa"

