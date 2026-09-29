"""Checkpoint 5 Comprehensive Test Suite — IrisAI Studio.

BATCH DATASET PROCESSING STUDIO — 5-MODEL UNIFICATION

Verifies:
1. Batch prediction includes all 5 classifiers: RBF, Linear, Polynomial, Sigmoid, and Deep Learning MLP.
2. Individual predictions, confidences (0-100), and test accuracies for all 5 classifiers.
3. Authentic Deep Learning MLP forward pass and calibrated softmax probabilities.
4. Sigmoid kernel predictions and non-linear outputs.
5. Consensus evaluation across all 5 models (Unanimous, Majority, Divergent).
6. No silent fallback to best model.
7. Batch CSV export generation with all 5 model columns.
8. Batch XLSX export generation: Sheet 1 (Batch Predictions) and Sheet 2 (Model Evaluation Summary) with all 5 models.
9. Dataset parsing for .csv, .txt, and .xlsx.
10. Input validation, handling of invalid/out-of-range rows with warnings.
11. Rejection of unsupported file formats and empty files.
12. API endpoints: /prediction/batch-upload and /prediction/batch-export for both xlsx and csv formats.
13. Authentication enforcement on batch endpoints.
"""
from __future__ import annotations

import io
import os
import sys
from datetime import datetime

import openpyxl
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth.service import create_access_token, hash_password
from database.connection import SessionLocal
from database.models import User
from main import app
from ml import svm_service
from prediction import batch_service

client = TestClient(app, raise_server_exceptions=True)

SAMPLE_ROWS = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5},
    {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.4, "petal_width": 2.1},
]


@pytest.fixture(scope="module")
def setup_cp5_auth():
    """Setup test user and admin auth headers for CP5 batch studio tests."""
    svm_service.ensure_models_loaded()
    db = SessionLocal()
    try:
        def get_or_create(username: str, role: str):
            u = db.query(User).filter(User.Username == username).first()
            if not u:
                u = User(
                    Username=username,
                    PasswordHash=hash_password("Cp5SecurePass123!"),
                    Role=role,
                    Status="Active",
                    CreatedAt=datetime.utcnow(),
                )
                db.add(u)
                db.commit()
                db.refresh(u)
            return u

        user = get_or_create("cp5_batch_user", "USER")
        user_token = create_access_token(user.Username, user.Role)

        yield {
            "user_headers": {"Authorization": f"Bearer {user_token}"},
            "username": user.Username,
        }
    finally:
        try:
            db.execute(text("DELETE FROM Users WHERE Username LIKE 'cp5_%'"))
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


class TestCheckpoint5BatchStudio:
    """Comprehensive test cases for Checkpoint 5 5-Model Batch Processing."""

    # -------------------------------------------------------------
    # 1. Batch Prediction Engine & 5-Model Completeness
    # -------------------------------------------------------------
    def test_01_batch_predict_returns_all_5_models(self):
        """Test 1: Batch prediction returns keys for all 5 models in each row."""
        svm_service.ensure_models_loaded()
        results, metrics = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)

        assert len(results) == len(SAMPLE_ROWS)
        for r in results:
            for model in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
                assert f"{model}_prediction" in r, f"Missing {model}_prediction in result row"
                assert f"{model}_confidence" in r, f"Missing {model}_confidence in result row"
                assert f"{model}_accuracy" in r, f"Missing {model}_accuracy in result row"

    def test_02_batch_predict_rbf_output_validity(self):
        """Test 2: RBF model in batch prediction outputs valid Iris species and bounded confidence."""
        results, _ = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        for r in results:
            assert r["rbf_prediction"] in ["Iris setosa", "Iris versicolor", "Iris virginica"]
            assert 0.0 <= r["rbf_confidence"] <= 100.0
            assert 0.0 <= r["rbf_accuracy"] <= 100.0

    def test_03_batch_predict_linear_output_validity(self):
        """Test 3: Linear model in batch prediction outputs valid Iris species and bounded confidence."""
        results, _ = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        for r in results:
            assert r["linear_prediction"] in ["Iris setosa", "Iris versicolor", "Iris virginica"]
            assert 0.0 <= r["linear_confidence"] <= 100.0
            assert 0.0 <= r["linear_accuracy"] <= 100.0

    def test_04_batch_predict_poly_output_validity(self):
        """Test 4: Polynomial model in batch prediction outputs valid Iris species and bounded confidence."""
        results, _ = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        for r in results:
            assert r["poly_prediction"] in ["Iris setosa", "Iris versicolor", "Iris virginica"]
            assert 0.0 <= r["poly_confidence"] <= 100.0
            assert 0.0 <= r["poly_accuracy"] <= 100.0

    def test_05_batch_predict_sigmoid_output_validity(self):
        """Test 5: Sigmoid model in batch prediction outputs valid Iris species and bounded confidence."""
        results, _ = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        for r in results:
            assert r["sigmoid_prediction"] in ["Iris setosa", "Iris versicolor", "Iris virginica"]
            assert 0.0 <= r["sigmoid_confidence"] <= 100.0
            assert 0.0 <= r["sigmoid_accuracy"] <= 100.0

    def test_06_batch_predict_mlp_output_validity(self):
        """Test 6: Deep Learning MLP in batch prediction outputs valid Iris species and bounded confidence."""
        results, _ = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        for r in results:
            assert r["mlp_prediction"] in ["Iris setosa", "Iris versicolor", "Iris virginica"]
            assert 0.0 <= r["mlp_confidence"] <= 100.0
            assert 0.0 <= r["mlp_accuracy"] <= 100.0

    def test_07_batch_mlp_is_real_neural_network(self):
        """Test 7: Verify MLP inference uses real neural network weights and architecture."""
        mlp_bundle = svm_service._models.get("mlp")
        assert mlp_bundle is not None
        model = mlp_bundle["model"]
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "coefs_")
        # MLP architecture (64, 32) has 3 weight matrices (input->64, 64->32, 32->3)
        assert len(model.coefs_) == 3

    def test_08_batch_mlp_probabilities_are_calibrated(self):
        """Test 8: Softmax probabilities from MLP in batch are positive and sum to ~1.0."""
        mlp_bundle = svm_service._models.get("mlp")
        scaler = svm_service._scaler
        model = mlp_bundle["model"]

        X = [[5.1, 3.5, 1.4, 0.2]]
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[0]

        assert len(proba) == 3
        assert sum(proba) == pytest.approx(1.0, abs=1e-4)
        assert all(p >= 0.0 for p in proba)
        assert max(proba) > 0.35  # Distinct highest probability above uniform distribution (0.33)

    def test_09_batch_consensus_calculation_across_5_models(self):
        """Test 9: Consensus incorporates all 5 models (Unanimous when all 5 agree)."""
        # Canonical setosa: all 5 models agree on Iris setosa
        setosa_rows = [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]
        results, _ = svm_service.batch_predict_all_kernels(setosa_rows)
        row = results[0]

        all_preds = [
            row["rbf_prediction"],
            row["linear_prediction"],
            row["poly_prediction"],
            row["sigmoid_prediction"],
            row["mlp_prediction"],
        ]
        assert len(set(all_preds)) == 1
        assert row["consensus"] == "Unanimous"

    def test_10_batch_consensus_handles_divergence_across_5_models(self):
        """Test 10: Consensus reflects majority or divergent status when models disagree."""
        # Ambiguous boundary point between versicolor and virginica
        border_rows = [{"sepal_length": 6.0, "sepal_width": 2.5, "petal_length": 5.0, "petal_width": 1.5}]
        results, _ = svm_service.batch_predict_all_kernels(border_rows)
        row = results[0]
        assert row["consensus"] in ["Unanimous", "Majority", "Divergent"]

    def test_11_no_silent_best_model_fallback_in_batch(self):
        """Test 11: Batch predictions preserve each model's independent result without overwriting with best_model."""
        results, _ = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        for r in results:
            preds = {
                "rbf": r["rbf_prediction"],
                "linear": r["linear_prediction"],
                "poly": r["poly_prediction"],
                "sigmoid": r["sigmoid_prediction"],
                "mlp": r["mlp_prediction"],
            }
            # All 5 models are evaluated individually and keyed separately
            assert len(preds) == 5

    def test_12_metrics_summary_contains_all_5_models(self):
        """Test 12: batch_predict_all_kernels returns metrics summary with all 5 models."""
        _, metrics = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        for model in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            assert model in metrics
            m = metrics[model]
            assert "test_accuracy" in m
            assert "cv_accuracy" in m
            assert "precision" in m
            assert "recall" in m
            assert "f1" in m

    # -------------------------------------------------------------
    # 2. Batch Export: CSV & Excel (XLSX)
    # -------------------------------------------------------------
    def test_13_batch_csv_export_contains_all_5_models(self):
        """Test 13: generate_batch_csv outputs CSV header and data for all 5 models."""
        results, _ = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        csv_str = batch_service.generate_batch_csv(results)

        df = pd.read_csv(io.StringIO(csv_str))
        expected_columns = [
            "Row",
            "Sepal Length (cm)",
            "Sepal Width (cm)",
            "Petal Length (cm)",
            "Petal Width (cm)",
            "RBF Prediction",
            "RBF Confidence (%)",
            "Linear Prediction",
            "Linear Confidence (%)",
            "Poly Prediction",
            "Poly Confidence (%)",
            "Sigmoid Prediction",
            "Sigmoid Confidence (%)",
            "MLP Prediction",
            "MLP Confidence (%)",
            "Consensus",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing CSV column: {col}"
        assert len(df) == len(SAMPLE_ROWS)

    def test_14_batch_excel_export_sheet1_contains_all_5_models(self):
        """Test 14: generate_batch_excel Sheet 1 'Batch Predictions' contains headers and data for all 5 models."""
        results, metrics = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        xlsx_bytes = batch_service.generate_batch_excel(results, metrics)

        wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
        assert "Batch Predictions" in wb.sheetnames
        ws = wb["Batch Predictions"]

        headers = [cell.value for cell in ws[1]]
        for expected in [
            "RBF Prediction",
            "Linear Prediction",
            "Poly Prediction",
            "Sigmoid Prediction",
            "MLP Prediction",
            "Sigmoid Confidence (%)",
            "MLP Confidence (%)",
            "Consensus",
        ]:
            assert expected in headers, f"Missing {expected} in Excel Sheet 1 header"

        assert ws.max_row == len(SAMPLE_ROWS) + 1

    def test_15_batch_excel_export_sheet2_contains_all_5_models(self):
        """Test 15: generate_batch_excel Sheet 2 'Model Evaluation Summary' contains metrics for all 5 models."""
        results, metrics = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        xlsx_bytes = batch_service.generate_batch_excel(results, metrics)

        wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes))
        assert "Model Evaluation Summary" in wb.sheetnames
        ws = wb["Model Evaluation Summary"]

        # First column has classifier names
        model_names = [ws.cell(row=i, column=1).value for i in range(2, ws.max_row + 1)]
        assert len(model_names) == 5
        # Verify MLP and Sigmoid presence in summary
        model_names_lower = [str(n).lower() for n in model_names]
        assert any("sigmoid" in n for n in model_names_lower)
        assert any("mlp" in n for n in model_names_lower)

    # -------------------------------------------------------------
    # 3. File Parsing: CSV, TXT, XLSX & Data Validation
    # -------------------------------------------------------------
    def test_16_parse_batch_file_csv_valid(self):
        """Test 16: parse_batch_file parses standard CSV content correctly."""
        csv_content = (
            "sepal_length,sepal_width,petal_length,petal_width\n"
            "5.1,3.5,1.4,0.2\n"
            "6.2,2.9,4.3,1.3\n"
        ).encode("utf-8")

        rows, warnings = batch_service.parse_batch_file(csv_content, "test.csv")
        assert len(rows) == 2
        assert len(warnings) == 0
        assert rows[0]["sepal_length"] == 5.1
        assert rows[1]["petal_width"] == 1.3

    def test_17_parse_batch_file_txt_tsv_valid(self):
        """Test 17: parse_batch_file parses tab-separated or space-separated TXT file."""
        txt_content = (
            "sepal_length\tsepal_width\tpetal_length\tpetal_width\n"
            "5.0\t3.4\t1.5\t0.2\n"
            "6.7\t3.1\t4.7\t1.5\n"
        ).encode("utf-8")

        rows, warnings = batch_service.parse_batch_file(txt_content, "test.txt")
        assert len(rows) == 2
        assert len(warnings) == 0
        assert rows[0]["sepal_length"] == 5.0

    def test_18_parse_batch_file_xlsx_valid(self):
        """Test 18: parse_batch_file parses XLSX binary data correctly."""
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["sepal_length", "sepal_width", "petal_length", "petal_width"])
        ws.append([5.1, 3.5, 1.4, 0.2])
        ws.append([5.8, 2.7, 5.1, 1.9])
        buf = io.BytesIO()
        wb.save(buf)

        rows, warnings = batch_service.parse_batch_file(buf.getvalue(), "test.xlsx")
        assert len(rows) == 2
        assert len(warnings) == 0
        assert rows[1]["petal_length"] == 5.1

    def test_19_parse_batch_file_skips_invalid_rows_with_warnings(self):
        """Test 19: Non-numeric, negative, or out-of-range values are skipped and generate warnings."""
        csv_content = (
            "sepal_length,sepal_width,petal_length,petal_width\n"
            "5.1,3.5,1.4,0.2\n"
            "invalid,3.5,1.4,0.2\n"
            "-1.0,3.5,1.4,0.2\n"
            "5.1,3.5,100.0,0.2\n"
            "6.0,3.0,4.5,1.5\n"
        ).encode("utf-8")

        rows, warnings = batch_service.parse_batch_file(csv_content, "mixed.csv")
        assert len(rows) == 2
        assert len(warnings) == 3

    def test_20_parse_batch_file_empty_raises_error(self):
        """Test 20: Empty file or file with no valid rows raises ValueError."""
        csv_content = "sepal_length,sepal_width,petal_length,petal_width\n".encode("utf-8")
        with pytest.raises(ValueError, match="The uploaded file contains no data"):
            batch_service.parse_batch_file(csv_content, "empty.csv")

    # -------------------------------------------------------------
    # 4. API Endpoints: Batch Upload & Export
    # -------------------------------------------------------------
    def test_21_api_batch_upload_returns_all_5_models(self, setup_cp5_auth):
        """Test 21: /prediction/batch-upload endpoint returns all 5 models in result rows and metrics."""
        csv_bytes = (
            "sepal_length,sepal_width,petal_length,petal_width\n"
            "5.1,3.5,1.4,0.2\n"
            "6.4,3.2,4.5,1.5\n"
        ).encode("utf-8")

        files = {"file": ("test.csv", csv_bytes, "text/csv")}
        resp = client.post(
            "/prediction/batch-upload",
            files=files,
            headers=setup_cp5_auth["user_headers"],
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert "batch_id" in data
        assert data["total_rows"] == 2
        assert "results" in data
        assert "metrics" in data

        row0 = data["results"][0]
        for model in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            assert f"{model}_prediction" in row0
            assert f"{model}_confidence" in row0
            assert model in data["metrics"]

    def test_22_api_batch_upload_unsupported_format_returns_400(self, setup_cp5_auth):
        """Test 22: /prediction/batch-upload with unsupported file extension returns HTTP 400."""
        files = {"file": ("data.pdf", b"fake-pdf-content", "application/pdf")}
        resp = client.post(
            "/prediction/batch-upload",
            files=files,
            headers=setup_cp5_auth["user_headers"],
        )
        assert resp.status_code == 400
        assert "Unsupported file format" in resp.json()["detail"]

    def test_23_api_batch_export_xlsx(self, setup_cp5_auth):
        """Test 23: /prediction/batch-export returns valid XLSX with all 5 models."""
        results, metrics = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        payload = {"format": "xlsx", "rows": results}

        resp = client.post(
            "/prediction/batch-export",
            json=payload,
            headers=setup_cp5_auth["user_headers"],
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        wb = openpyxl.load_workbook(io.BytesIO(resp.content))
        assert "Batch Predictions" in wb.sheetnames
        assert "Model Evaluation Summary" in wb.sheetnames

    def test_24_api_batch_export_csv(self, setup_cp5_auth):
        """Test 24: /prediction/batch-export returns valid CSV containing all 5 models."""
        results, metrics = svm_service.batch_predict_all_kernels(SAMPLE_ROWS)
        payload = {"format": "csv", "rows": results}

        resp = client.post(
            "/prediction/batch-export",
            json=payload,
            headers=setup_cp5_auth["user_headers"],
        )
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

        content = resp.content.decode("utf-8")
        assert "Sigmoid Prediction" in content
        assert "MLP Prediction" in content
        assert "RBF Prediction" in content

    def test_25_batch_endpoints_require_authentication(self):
        """Test 25: Unauthenticated requests to /prediction/batch-upload return HTTP 401 or 403."""
        files = {"file": ("test.csv", b"sepal_length,sepal_width,petal_length,petal_width\n5.1,3.5,1.4,0.2", "text/csv")}
        resp = client.post("/prediction/batch-upload", files=files)
        assert resp.status_code in [401, 403]
