"""Checkpoint 19.8: Verification of ModelHistory in SQL Server."""
from datetime import datetime
import pytest
from fastapi.testclient import TestClient

from main import app
from database.connection import SessionLocal
from database.models import ModelHistory
from database.repositories.model_history_repository import (
    record_model_history,
    get_model_history,
    model_history_to_dict,
)
from auth.service import create_access_token
import config

client = TestClient(app)


def test_orm_schema_and_direct_crud():
    db = SessionLocal()
    try:
        # Check ORM columns
        cols = {c.name for c in ModelHistory.__table__.columns}
        expected = {"ModelID", "ModelName", "Kernel", "Accuracy", "PrecisionScore", "RecallScore", "F1Score", "CreatedAt"}
        assert expected.issubset(cols), f"Missing columns in ModelHistory: {expected - cols}"

        # 1. Test insertion in transaction
        rec = record_model_history(
            db=db,
            model_name="SVM_RBF_TEST",
            kernel="rbf",
            accuracy=0.98,
            precision_score=0.97,
            recall_score=0.98,
            f1_score=0.975,
            created_at=datetime(2026, 9, 17, 11, 0, 0),
        )
        assert rec.ModelID is not None and rec.ModelID > 0
        assert rec.ModelName == "SVM_RBF_TEST"
        assert rec.Kernel == "RBF"
        assert rec.Accuracy == 0.98
        assert rec.PrecisionScore == 0.97
        assert rec.RecallScore == 0.98
        assert rec.F1Score == 0.975
        assert rec.CreatedAt == datetime(2026, 9, 17, 11, 0, 0)

        # 2. Test reading record back
        records = get_model_history(db)
        found = [r for r in records if r.ModelID == rec.ModelID]
        assert len(found) == 1
        d = model_history_to_dict(found[0])
        assert d["model_id"] == rec.ModelID
        assert d["kernel"] == "RBF"
        assert d["best_kernel"] == "rbf"
        assert d["kernel_display"] == "RBF"
        assert d["accuracy"] == 0.98

        # 3. Test allowed classifiers: LINEAR, POLY, SIGMOID, MLP
        rec_lin = record_model_history(db, "SVM_LINEAR_TEST", "linear", 0.96)
        assert rec_lin.Kernel == "LINEAR"

        rec_poly = record_model_history(db, "SVM_POLY_TEST", "poly", 0.95)
        assert rec_poly.Kernel == "POLY"

        rec_sig = record_model_history(db, "SVM_SIGMOID_TEST", "sigmoid", 0.88)
        assert rec_sig.Kernel == "SIGMOID"

        rec_mlp = record_model_history(db, "Deep Learning MLP", "mlp", 0.95)
        assert rec_mlp.Kernel == "MLP"

        # 4. Strictly verify unsupported kernel rejection
        with pytest.raises(ValueError) as exc:
            record_model_history(db, "SVM_INVALID_TEST", "unsupported_kernel", 0.80)
        assert "Invalid or unsupported kernel" in str(exc.value)


        # 5. Rollback
        db.rollback()

        # 6. Verify database is clean
        count = db.query(ModelHistory).count()
        assert count == 0, f"Expected 0 records after rollback, got {count}"
    finally:
        db.close()


def test_api_model_history_authorization():
    # 1. Unauthenticated request -> Rejected
    r_unauth = client.get("/ml/model-history")
    assert r_unauth.status_code in (401, 403)

    # 2. Regular user token -> Forbidden
    user_token = create_access_token(username="__sql_test_user__", role="USER")
    r_user = client.get("/ml/model-history", headers={"Authorization": f"Bearer {user_token}"})
    assert r_user.status_code == 403

    # 3. Admin token -> Success
    admin_token = create_access_token(username=config.ADMIN_USERNAME, role="ADMIN")
    r_admin = client.get("/ml/model-history", headers={"Authorization": f"Bearer {admin_token}"})
    assert r_admin.status_code == 200
    assert isinstance(r_admin.json(), list)
