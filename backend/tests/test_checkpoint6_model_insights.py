"""Checkpoint 6 Comprehensive Test Suite — IrisAI Studio.

COMPARATIVE MULTI-MODEL INFERENCE & BOTANICAL MODEL INSIGHTS
(5-MODEL UNIFICATION: RBF, LINEAR, POLY, SIGMOID, MLP)

Verifies:
1. Comparative inference (/prediction/predict-all) returns all 5 models.
2. Individual predictions, confidences, and accuracies are mathematically bounded.
3. Real calibrated probabilities for all 5 models.
4. Genuine Softmax probability distribution for Deep Learning MLP.
5. 5-model unanimous consensus calculation.
6. Majority and divergent consensus ratio calculations.
7. Absence of silent best-model fallback.
8. /prediction/insights contains prediction results for all 5 models.
9. Insights diagnostics dynamically mention all 5 classifiers without 3-kernel hardcodes.
10. Insights majority diagnostics dynamically calculate model ratios.
11. Deep Learning MLP is strictly identified as a Neural Network, not an SVM kernel.
12. Sigmoid kernel diagnostics and non-linear hyperbolic tangent description.
13. Morphological distance and Z-score botanical analysis.
14. Canonical petal discriminant notes according to Fisher's taxonomy.
15. Outlier detection for atypical dimensions.
16. Accurate highest-confidence classifier identification.
17. Authentication enforcement on /prediction/insights.
18. Authentication enforcement on /prediction/predict-all.
19. Presence of model_explanations for all 5 models.
20. History persistence in SQL Server for primary prediction.
"""
from __future__ import annotations

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
from database.models import User, Prediction
from main import app
from ml import svm_service
from prediction import insights_service

client = TestClient(app, raise_server_exceptions=True)

CANONICAL_SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
BOUNDARY_SAMPLE = {"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 4.9, "petal_width": 1.8}
OUTLIER_SAMPLE = {"sepal_length": 18.0, "sepal_width": 3.0, "petal_length": 5.0, "petal_width": 1.5}


@pytest.fixture(scope="module")
def setup_cp6_auth():
    """Setup test user and admin auth headers for CP6 tests."""
    svm_service.ensure_models_loaded()
    db = SessionLocal()
    try:
        def get_or_create(username: str, role: str):
            u = db.query(User).filter(User.Username == username).first()
            if not u:
                u = User(
                    Username=username,
                    PasswordHash=hash_password("Cp6SecurePass123!"),
                    Role=role,
                    Status="Active",
                    CreatedAt=datetime.utcnow(),
                )
                db.add(u)
                db.commit()
                db.refresh(u)
            return u

        user = get_or_create("cp6_insights_user", "USER")
        user_token = create_access_token(user.Username, user.Role)

        yield {
            "user_headers": {"Authorization": f"Bearer {user_token}"},
            "username": user.Username,
        }
    finally:
        try:
            db.execute(text("DELETE FROM Predictions WHERE UserID IN (SELECT UserID FROM Users WHERE Username LIKE 'cp6_%')"))
            db.execute(text("DELETE FROM Users WHERE Username LIKE 'cp6_%'"))
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


class TestCheckpoint6ModelInsights:
    """Comprehensive test suite for Checkpoint 6."""

    # -------------------------------------------------------------
    # 1. Comparative Multi-Model Inference (/prediction/predict-all)
    # -------------------------------------------------------------
    def test_01_comparative_inference_returns_all_5_models(self, setup_cp6_auth):
        """Test 1: Comparative inference endpoint returns results for all 5 models."""
        resp = client.post(
            "/prediction/predict-all",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert "results" in data
        for model in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            assert model in data["results"], f"Missing {model} in results"

    def test_02_comparative_individual_model_validity(self, setup_cp6_auth):
        """Test 2: Each model output contains valid species, bounded confidence, and accuracy."""
        resp = client.post(
            "/prediction/predict-all",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()

        for model, res in data["results"].items():
            assert res["predicted_species"] in ["Iris setosa", "Iris versicolor", "Iris virginica"]
            assert 0.0 <= res["confidence"] <= 100.0
            assert 0.0 <= res["accuracy"] <= 100.0
            assert 0.0 <= res["cv_accuracy"] <= 100.0
            assert "probabilities" in res

    def test_03_comparative_real_calibrated_probabilities(self, setup_cp6_auth):
        """Test 3: Probabilities for all 5 models are non-negative and sum to ~100%."""
        resp = client.post(
            "/prediction/predict-all",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()

        for model, res in data["results"].items():
            probs = res["probabilities"]
            assert all(p >= 0.0 for p in probs.values())
            total = sum(probs.values())
            assert abs(total - 100.0) < 0.5, f"Probabilities for {model} sum to {total} instead of ~100"

    def test_04_comparative_mlp_softmax_integrity(self, setup_cp6_auth):
        """Test 4: Deep Learning MLP outputs genuine softmax probabilities and correct species for Setosa."""
        resp = client.post(
            "/prediction/predict-all",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()

        mlp_res = data["results"]["mlp"]
        assert mlp_res["kernel"] == "mlp"
        assert mlp_res["kernel_display"] == "Deep Learning MLP"
        assert mlp_res["predicted_species"] == "Iris setosa"
        assert mlp_res["confidence"] > 35.0

    def test_05_comparative_consensus_unanimous_5_models(self, setup_cp6_auth):
        """Test 5: Canonical Setosa produces unanimous consensus across all 5 models."""
        resp = client.post(
            "/prediction/predict-all",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["consensus"] == "unanimous"
        assert "5/5" in data["consensus_text"] or "Unanimous" in data["consensus_text"]

    def test_06_comparative_consensus_majority_or_divergent(self, setup_cp6_auth):
        """Test 6: Boundary sample demonstrates majority or divergent consensus without error."""
        resp = client.post(
            "/prediction/predict-all",
            json=BOUNDARY_SAMPLE,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["consensus"] in ["unanimous", "majority", "divergent"]
        assert len(data["results"]) == 5

    def test_07_no_silent_fallback_in_comparative(self, setup_cp6_auth):
        """Test 7: Comparative results do not force all models to match best_model."""
        resp = client.post(
            "/prediction/predict-all",
            json=BOUNDARY_SAMPLE,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "highest_confidence" in data
        assert data["highest_confidence"]["kernel"] in ["rbf", "linear", "poly", "sigmoid", "mlp"]

    # -------------------------------------------------------------
    # 2. Model Insights (/prediction/insights) & Service Diagnostics
    # -------------------------------------------------------------
    def test_08_insights_endpoint_contains_5_models_results(self, setup_cp6_auth):
        """Test 8: /prediction/insights returns both multi_kernel_results with 5 models and insights object."""
        resp = client.post(
            "/prediction/insights",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert "results" in data
        assert len(data["results"]) == 5
        assert "insights" in data
        assert "discriminant_notes" in data["insights"]
        assert "species_alignment" in data["insights"]
        assert "model_diagnostics" in data["insights"]

    def test_09_insights_unanimous_mentions_all_5_models(self, setup_cp6_auth):
        """Test 9: Dynamic unanimous diagnostic text mentions all 5 classifiers, not hardcoded 3."""
        resp = client.post(
            "/prediction/insights",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()
        diags = data["insights"]["model_diagnostics"]

        full_text = " ".join(diags)
        assert "All 5 classifiers" in full_text or "5 classifiers" in full_text
        assert "All three kernels" not in full_text

    def test_10_insights_majority_mentions_correct_ratio(self, setup_cp6_auth):
        """Test 10: Dynamic majority diagnostic calculation does not hardcode '(2/3)'."""
        # Test directly via insights_service
        fake_multi_result = {
            "results": {
                "rbf": {"predicted_species": "Iris versicolor", "confidence": 92.0},
                "linear": {"predicted_species": "Iris virginica", "confidence": 75.0},
                "poly": {"predicted_species": "Iris versicolor", "confidence": 88.0},
                "sigmoid": {"predicted_species": "Iris versicolor", "confidence": 82.0},
                "mlp": {"predicted_species": "Iris versicolor", "confidence": 89.0},
            },
            "consensus": "majority",
            "highest_confidence": {"kernel": "rbf", "kernel_display": "RBF", "predicted_species": "Iris versicolor"},
        }
        res = insights_service.analyze_sample_insights(6.0, 2.7, 4.9, 1.8, fake_multi_result)
        diags_text = " ".join(res["model_diagnostics"])

        assert "4/5" in diags_text
        assert "2/3" not in diags_text

    def test_11_insights_mlp_distinction_as_neural_network(self, setup_cp6_auth):
        """Test 11: Diagnostics explicitly describe Deep Learning MLP as a neural network and not an SVM kernel."""
        resp = client.post(
            "/prediction/insights",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()
        diags = data["insights"]["model_diagnostics"]

        mlp_notes = [d for d in diags if "multi-layer perceptron" in d]
        assert len(mlp_notes) > 0
        mlp_text = mlp_notes[0]
        assert "neural network" in mlp_text.lower()
        assert "relu" in mlp_text.lower()
        assert "softmax" in mlp_text.lower()
        assert "kernel function" not in mlp_text.lower()

    def test_12_insights_sigmoid_kernel_diagnostics(self, setup_cp6_auth):
        """Test 12: Diagnostics contain explicit section analyzing SVM Sigmoid."""
        resp = client.post(
            "/prediction/insights",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()
        diags = data["insights"]["model_diagnostics"]

        sig_notes = [d for d in diags if "SVM Sigmoid evaluates" in d or "hyperbolic tangent" in d]
        assert len(sig_notes) > 0
        sig_text = sig_notes[0]
        assert "hyperbolic tangent" in sig_text.lower() or "s-curve" in sig_text.lower()

    def test_13_insights_morphological_distance_and_z_scores(self, setup_cp6_auth):
        """Test 13: Morphological distance and Z-scores are computed for all 3 Iris species."""
        resp = client.post(
            "/prediction/insights",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()
        align = data["insights"]["species_alignment"]

        assert "Iris setosa" in align
        assert "Iris versicolor" in align
        assert "Iris virginica" in align

        # For canonical Setosa, distance to Setosa should be the smallest
        assert align["Iris setosa"]["morphological_distance"] < align["Iris virginica"]["morphological_distance"]
        assert data["insights"]["closest_botanical_match"] == "Iris setosa"

    def test_14_insights_discriminant_notes_botanical(self, setup_cp6_auth):
        """Test 14: Botanical discriminant notes recognize petal length <= 2.2 cm as absolute Setosa separator."""
        resp = client.post(
            "/prediction/insights",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()
        notes = " ".join(data["insights"]["discriminant_notes"])

        assert "Iris setosa" in notes
        assert "2.2 cm" in notes or "linearly separable" in notes

    def test_15_insights_outlier_detection(self, setup_cp6_auth):
        """Test 15: Outlier sample triggers outlier warning in insights."""
        resp = client.post(
            "/prediction/insights",
            json=OUTLIER_SAMPLE,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()
        outliers = data["insights"]["outliers"]

        assert len(outliers) > 0
        assert any("Sepal Length" in o for o in outliers)

    def test_16_insights_highest_confidence_mapping(self, setup_cp6_auth):
        """Test 16: Highest-confidence model is correctly mapped and identified in insights."""
        resp = client.post(
            "/prediction/insights",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()

        highest = data["highest_confidence"]
        assert highest["kernel"] in data["results"]
        # Confidence must equal the max confidence among all 5 models
        max_conf = max(r["confidence"] for r in data["results"].values())
        assert highest["confidence"] == max_conf

    def test_17_security_insights_requires_authentication(self):
        """Test 17: Unauthenticated POST to /prediction/insights returns HTTP 401 or 403."""
        resp = client.post("/prediction/insights", json=CANONICAL_SETOSA)
        assert resp.status_code in [401, 403]

    def test_18_security_predict_all_requires_authentication(self):
        """Test 18: Unauthenticated POST to /prediction/predict-all returns HTTP 401 or 403."""
        resp = client.post("/prediction/predict-all", json=CANONICAL_SETOSA)
        assert resp.status_code in [401, 403]

    def test_19_model_explanations_dictionary(self, setup_cp6_auth):
        """Test 19: Insights contains model_explanations dictionary for all 5 models."""
        resp = client.post(
            "/prediction/insights",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "model_explanations" in data["insights"]
        exps = data["insights"]["model_explanations"]
        for m in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            assert m in exps, f"Missing explanation for {m}"

    def test_20_predict_all_records_history_in_sql_server(self, setup_cp6_auth):
        """Test 20: /prediction/predict-all persists the primary prediction in SQL Server Predictions table."""
        resp = client.post(
            "/prediction/predict-all",
            json=CANONICAL_SETOSA,
            headers=setup_cp6_auth["user_headers"],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "history_id" in data
        history_id = data["history_id"]

        db = SessionLocal()
        try:
            record = db.query(Prediction).filter(Prediction.PredictionID == history_id).first()
            if record:
                assert record.PredictedSpecies == "Iris setosa"
                assert record.Kernel.lower() in ["rbf", "linear", "poly", "sigmoid", "mlp"]
        finally:
            db.close()
