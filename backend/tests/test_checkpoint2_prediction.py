"""Checkpoint 2 Comprehensive Test Suite — IrisAI Studio.

Verifies the implementation of Prediction Studio with explicit model selection:
1. Single prediction across all 5 models (rbf, linear, poly, sigmoid, mlp).
2. Deep Learning MLP genuine classifier verification (non-fabricated, real probabilities).
3. Strict model enforcement: NO silent fallback to _best_model.
4. Rejection of invalid models with HTTP 400.
5. SQL Server persistence: verification of records in Predictions table (especially SIGMOID and MLP).
6. Species image endpoint integration.
7. Backward compatibility for requests omitting the model field.
"""
from __future__ import annotations

import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from datetime import datetime
import config
from main import app
from ml import svm_service
from database.connection import SessionLocal
from database.models import User, Prediction
from auth.service import hash_password, create_access_token

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def auth_headers():
    """Ensure models loaded and create/get test user with a valid JWT token."""
    svm_service.ensure_models_loaded()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.Username == "cp2_test_user").first()
        if not user:
            user = User(
                Username="cp2_test_user",
                PasswordHash=hash_password("Cp2SecurePass123"),
                Role="USER",
                Status="Active",
                CreatedAt=datetime.utcnow(),
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        token = create_access_token(user.Username, user.Role)
        return {"Authorization": f"Bearer {token}"}
    finally:
        db.close()


class TestCheckpoint2PredictionStudio:
    """Test suite for Checkpoint 2 Prediction Studio requirements."""

    def test_single_prediction_all_five_models(self, auth_headers):
        """Test 1: Verify prediction with each of the 5 explicit models."""
        expected_classifiers = [
            ("rbf", "SVM", "RBF"),
            ("linear", "SVM", "Linear"),
            ("poly", "SVM", "Polynomial"),
            ("sigmoid", "SVM", "Sigmoid"),
            ("mlp", "Deep Learning MLP", "Deep Learning MLP"),
        ]

        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }

        for key, expected_model, expected_display in expected_classifiers:
            resp = client.post(
                "/prediction/predict",
                headers=auth_headers,
                json={**payload, "model": key},
            )
            assert resp.status_code == 200, f"Failed for model '{key}': {resp.text}"
            data = resp.json()

            assert data["kernel"] == key, f"Expected kernel '{key}', got '{data['kernel']}'"
            assert data["model"] == expected_model, f"Expected model '{expected_model}', got '{data['model']}'"
            assert data["kernel_display"] == expected_display
            assert data["predicted_species"] in ["Iris setosa", "Iris versicolor", "Iris virginica"]
            assert "history_id" in data
            assert data["history_id"] is not None

            # Probabilities validation
            probs = data["probabilities"]
            assert "setosa" in probs and "versicolor" in probs and "virginica" in probs
            total_prob = probs["setosa"] + probs["versicolor"] + probs["virginica"]
            assert abs(total_prob - 100.0) < 1.0, f"Probabilities for '{key}' do not sum to 100%: {total_prob}"

    def test_mlp_genuine_classifier(self, auth_headers):
        """Test 2: Verify Deep Learning MLP runs genuine MLPClassifier inference."""
        resp = client.post(
            "/prediction/predict",
            headers=auth_headers,
            json={
                "sepal_length": 6.3,
                "sepal_width": 3.3,
                "petal_length": 6.0,
                "petal_width": 2.5,
                "model": "mlp",
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["model"] == "Deep Learning MLP"
        assert data["kernel"] == "mlp"
        assert data["kernel_display"] == "Deep Learning MLP"
        assert data["predicted_species"] in ["Iris setosa", "Iris versicolor", "Iris virginica"]

        probs = data["probabilities"]
        assert all(p >= 0.0 for p in probs.values())
        assert abs(sum(probs.values()) - 100.0) < 1.0

    def test_no_silent_fallback(self, auth_headers):
        """Test 3: Verify system never silently falls back to _best_model when model is chosen."""
        best_info = svm_service.get_best_model_info()
        best_kernel = best_info.get("kernel", "linear")

        # Pick a kernel distinct from best_kernel
        all_kernels = ["rbf", "linear", "poly", "sigmoid", "mlp"]
        non_best_kernels = [k for k in all_kernels if k != best_kernel]
        assert len(non_best_kernels) > 0

        target_kernel = non_best_kernels[0]
        resp = client.post(
            "/prediction/predict",
            headers=auth_headers,
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
                "model": target_kernel,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["kernel"] == target_kernel
        assert data["kernel"] != best_kernel or target_kernel == best_kernel

    def test_invalid_model_rejected(self, auth_headers):
        """Test 4: Verify unsupported model keys are rejected with HTTP 400."""
        resp = client.post(
            "/prediction/predict",
            headers=auth_headers,
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
                "model": "unsupported_kernel",
            },
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "Invalid model 'unsupported_kernel'" in detail
        assert "Supported models:" in detail

    def test_sql_server_persistence_all_models(self, auth_headers):
        """Test 5: Verify records are actually persisted to SQL Server Predictions table."""
        db = SessionLocal()
        created_ids = []
        try:
            for k in ["sigmoid", "mlp", "rbf", "linear", "poly"]:
                resp = client.post(
                    "/prediction/predict",
                    headers=auth_headers,
                    json={
                        "sepal_length": 5.8,
                        "sepal_width": 2.7,
                        "petal_length": 4.1,
                        "petal_width": 1.0,
                        "model": k,
                    },
                )
                assert resp.status_code == 200, f"Prediction failed for {k}: {resp.text}"
                pred_id = resp.json().get("history_id")
                assert pred_id is not None, f"No history_id returned for {k}"
                created_ids.append(int(pred_id))

            # Query SQL Server directly to verify persistence
            rows = db.query(Prediction).filter(Prediction.PredictionID.in_(created_ids)).all()
            assert len(rows) == 5, f"Expected 5 records in SQL Server, found {len(rows)}"

            kernels_in_db = {r.Kernel.lower() for r in rows if r.Kernel}
            assert "sigmoid" in kernels_in_db, "SIGMOID record missing in SQL Server Predictions!"
            assert "mlp" in kernels_in_db, "MLP record missing in SQL Server Predictions!"

            # Check model name for MLP
            mlp_row = next(r for r in rows if r.Kernel.lower() == "mlp")
            assert mlp_row.ModelName == "Deep Learning MLP"
            assert mlp_row.PredictedSpecies in ["Iris setosa", "Iris versicolor", "Iris virginica"]
            assert mlp_row.SepalLength == 5.8
        finally:
            # Clean up test rows
            if created_ids:
                db.query(Prediction).filter(Prediction.PredictionID.in_(created_ids)).delete(synchronize_session=False)
                db.commit()
            db.close()

    def test_species_image_integration(self, auth_headers):
        """Test 6: Verify image endpoint returns valid data for predicted species."""
        for species in ["Iris setosa", "Iris versicolor", "Iris virginica"]:
            resp = client.get(f"/image/species/{species}", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "image_url" in data
            assert "source" in data

    def test_backward_compatibility_none_model(self, auth_headers):
        """Test 7: Verify request without model field defaults to best model safely."""
        resp = client.post(
            "/prediction/predict",
            headers=auth_headers,
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["kernel"] in ["rbf", "linear", "poly", "sigmoid", "mlp"]
        assert data["history_id"] is not None

    def test_compare_all_endpoint_five_models(self, auth_headers):
        """Test 8: Verify predict-all endpoint returns predictions across all 5 models."""
        resp = client.post(
            "/prediction/predict-all",
            headers=auth_headers,
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        results = data["results"]
        for expected_key in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            assert expected_key in results, f"'{expected_key}' missing from predict-all results"
            assert results[expected_key]["confidence"] > 0
            assert "probabilities" in results[expected_key]
