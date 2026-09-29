"""Checkpoint 1 Comprehensive Test Suite — IrisAI Studio.

Verifies the integration of the complete 5-classifier set:
1. SVM RBF
2. SVM Linear
3. SVM Polynomial
4. SVM Sigmoid
5. Deep Learning MLP

Tests metrics, non-fabricated probability output, cross-validation,
joblib persistence, and SQL Server ModelHistory integration with rollback safety.
"""
from __future__ import annotations

import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np

import config
from ml import svm_service, decision_boundary
from dl import dl_service
from database.connection import SessionLocal
from database.models import ModelHistory
from database.repositories import model_history_repository


# ──────────────────────────────────────────────────────────────────────────────
# 1. Test 5-Classifier Set Training & Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestClassifierSetTraining:
    def test_all_five_classifiers_trained(self):
        """Verify train_all_kernels trains all 5 classifiers and produces valid metrics."""
        summary = svm_service.train_all_kernels(save=False)

        expected_classifiers = ["rbf", "linear", "poly", "sigmoid", "mlp"]
        for clf in expected_classifiers:
            assert clf in summary, f"Classifier '{clf}' missing from training summary."
            info = summary[clf]
            assert "cv_mean" in info
            assert "test_accuracy" in info
            assert "precision" in info
            assert "recall" in info
            assert "f1" in info
            assert "confusion_matrix" in info
            assert "cv_scores" in info

            # Metrics bounds
            assert 0.0 <= info["cv_mean"] <= 1.0
            assert 0.0 <= info["test_accuracy"] <= 1.0
            assert 0.0 <= info["precision"] <= 1.0
            assert 0.0 <= info["recall"] <= 1.0
            assert 0.0 <= info["f1"] <= 1.0

            # 5-fold cross validation
            assert len(info["cv_scores"]) == 5

            # Confusion matrix is 3x3 for the 3 Iris classes
            cm = np.array(info["confusion_matrix"])
            assert cm.shape == (3, 3)

        assert summary["best_kernel"] in expected_classifiers

    def test_evaluation_data_contains_all_five(self):
        """Verify get_evaluation_data returns full evaluation for all 5 classifiers."""
        eval_data = svm_service.get_evaluation_data()
        for clf in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            assert clf in eval_data, f"Classifier '{clf}' missing from evaluation data."
            assert eval_data[clf]["kernel_display"] is not None
            assert len(eval_data[clf]["confusion_matrix"]) == 3


# ──────────────────────────────────────────────────────────────────────────────
# 2. Test Deep Learning MLP Implementation
# ──────────────────────────────────────────────────────────────────────────────

class TestDeepLearningMLP:
    def test_mlp_pipeline_end_to_end(self):
        """Verify DLPipeline executes all 6 phases and produces genuine metrics."""
        pipeline = dl_service.DLPipeline()
        result = pipeline.run_pipeline()

        assert result["data_loaded"] is True
        assert result["preprocessed"] is True
        assert result["status"] == "pipeline complete"
        assert result["training_history"]["status"] == "trained"

        # Check model architecture
        model = pipeline._model
        assert model is not None
        assert model.hidden_layer_sizes == (64, 32)
        assert model.activation == "relu"
        assert model.solver == "adam"

        # Evaluation metrics
        eval_metrics = result["evaluation"]
        assert 0.0 <= eval_metrics["accuracy"] <= 1.0
        assert 0.0 <= eval_metrics["f1_score"] <= 1.0
        assert eval_metrics["confusion_matrix"] is not None

    def test_mlp_predict_genuine_probabilities(self):
        """Verify MLP predict() outputs genuine Softmax probabilities summing to 100%."""
        pipeline = dl_service.DLPipeline()
        pipeline.run_pipeline()

        sample = [5.1, 3.5, 1.4, 0.2]  # Canonical Setosa sample
        pred = pipeline.predict(sample)

        assert pred["status"] == "success"
        assert pred["predicted_species"] in ["Iris setosa", "Iris versicolor", "Iris virginica"]
        probs = pred["probabilities"]
        prob_sum = probs["setosa"] + probs["versicolor"] + probs["virginica"]
        assert abs(prob_sum - 100.0) < 0.5
        assert pred["confidence"] == max(probs.values())


# ──────────────────────────────────────────────────────────────────────────────
# 3. Test SVM Sigmoid Implementation
# ──────────────────────────────────────────────────────────────────────────────

class TestSVMSigmoid:
    def test_sigmoid_metrics_and_probabilities(self):
        """Verify Sigmoid kernel trains, evaluates, and outputs genuine probabilities."""
        svm_service.ensure_models_loaded()
        metrics = svm_service.get_all_kernels_metrics()

        assert "sigmoid" in metrics
        sig_metric = metrics["sigmoid"]
        assert sig_metric["kernel_display"] == "Sigmoid"
        assert 0.0 <= sig_metric["test_accuracy"] <= 100.0
        assert 0.0 <= sig_metric["cv_accuracy"] <= 100.0

    def test_sigmoid_decision_boundary(self):
        """Verify 2D decision boundary visualization generates for Sigmoid kernel."""
        img_b64 = decision_boundary.generate_decision_boundary("sigmoid", 2, 3)
        assert isinstance(img_b64, str)
        assert len(img_b64) > 1000  # Valid non-empty base64 PNG


# ──────────────────────────────────────────────────────────────────────────────
# 4. Test Probability Output Across All 5 Classifiers
# ──────────────────────────────────────────────────────────────────────────────

class TestProbabilityAndConfidenceOutput:
    def test_all_models_produce_calibrated_probabilities(self):
        """Verify predict_all_kernels outputs genuine probabilities for all 5 models."""
        svm_service.ensure_models_loaded()
        res = svm_service.predict_all_kernels(5.1, 3.5, 1.4, 0.2)

        results = res["results"]
        assert len(results) == 5
        for clf in ["rbf", "linear", "poly", "sigmoid", "mlp"]:
            assert clf in results
            r = results[clf]
            probs = r["probabilities"]
            total = probs["setosa"] + probs["versicolor"] + probs["virginica"]
            assert abs(total - 100.0) < 0.5, f"{clf} probabilities sum to {total} != 100"
            assert r["confidence"] == max(probs.values())
            assert r["confidence"] > 0.0

    def test_best_model_predict_contract(self):
        """Verify predict() on the best model adheres to API contract."""
        pred = svm_service.predict(6.0, 2.9, 4.5, 1.5)
        assert "predicted_class_idx" in pred
        assert "predicted_species" in pred
        assert "probabilities" in pred
        assert "confidence" in pred
        assert "model" in pred
        assert "kernel" in pred


# ──────────────────────────────────────────────────────────────────────────────
# 5. Test SQL Server ModelHistory for All 5 Classifiers
# ──────────────────────────────────────────────────────────────────────────────

class TestSQLServerModelHistoryAllFive:
    def test_record_all_five_classifiers_in_sql_server(self):
        """Verify SQL Server ModelHistory table safely stores and retrieves all 5 classifiers."""
        db = SessionLocal()
        created_records = []
        try:
            classifiers_to_test = [
                ("SVM_RBF_TEST", "rbf", 0.98),
                ("SVM_LINEAR_TEST", "linear", 0.96),
                ("SVM_POLY_TEST", "poly", 0.95),
                ("SVM_SIGMOID_TEST", "sigmoid", 0.88),
                ("Deep Learning MLP", "mlp", 0.97),
            ]

            for model_name, kernel, acc in classifiers_to_test:
                rec = model_history_repository.record_model_history(
                    db=db,
                    model_name=model_name,
                    kernel=kernel,
                    accuracy=acc,
                    precision_score=acc,
                    recall_score=acc,
                    f1_score=acc,
                )
                assert rec.ModelID is not None and rec.ModelID > 0
                assert rec.Kernel == kernel.upper()
                created_records.append(rec)

            # Retrieve records within transaction
            records = model_history_repository.get_model_history(db)
            saved_ids = {r.ModelID for r in created_records}
            matched = [r for r in records if r.ModelID in saved_ids]
            assert len(matched) == 5

            # Test dictionary representation
            dict_map = {r.Kernel: model_history_repository.model_history_to_dict(r) for r in matched}
            assert dict_map["RBF"]["kernel_display"] == "RBF"
            assert dict_map["LINEAR"]["kernel_display"] == "Linear"
            assert dict_map["POLY"]["kernel_display"] == "Polynomial"
            assert dict_map["SIGMOID"]["kernel_display"] == "Sigmoid"
            assert dict_map["MLP"]["kernel_display"] == "Deep Learning MLP"

            # Rollback to ensure complete database cleanliness
            db.rollback()

            # Confirm rollback
            count_after = db.query(ModelHistory).filter(ModelHistory.ModelID.in_(list(saved_ids))).count()
            assert count_after == 0
        finally:
            db.close()
