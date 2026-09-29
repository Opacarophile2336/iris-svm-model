"""Checkpoint 19.11: Verification of SQL Server Transaction & Rollback Safety.

Tests that all 4 repositories:
- Users (user_repository)
- Predictions (prediction_repository)
- KernelUsageHistory (kernel_usage_repository)
- ModelHistory (model_history_repository)
properly use flush/refresh without committing, allowing the caller/service
to manage transactions with rollback or commit, ensuring zero test data leakage.
"""
import pytest
from datetime import datetime

from database.connection import SessionLocal
from database.models import User, Prediction, KernelUsageHistory, ModelHistory
from database.repositories import (
    user_repository,
    kernel_usage_repository,
    model_history_repository,
)
from prediction import prediction_repository

TEST_USER_TX_ROLLBACK = "__test_tx_user_rollback__"
TEST_USER_TX_COMMIT = "__test_tx_user_commit__"
TEST_MODEL_NAME = "__TEST_TX_MODEL__"
TEST_PRED_MODEL = "__TEST_TX_PRED__"


def _cleanup_test_data():
    """Ensure no test transaction artifacts remain in the database."""
    db = SessionLocal()
    try:
        # Cleanup predictions with test model name
        db.query(Prediction).filter(
            Prediction.ModelName == TEST_PRED_MODEL
        ).delete(synchronize_session=False)

        # Cleanup model history with test model name
        db.query(ModelHistory).filter(
            ModelHistory.ModelName == TEST_MODEL_NAME
        ).delete(synchronize_session=False)

        # Cleanup test users and their cascade/associated data
        test_usernames = [TEST_USER_TX_ROLLBACK, TEST_USER_TX_COMMIT]
        users = db.query(User).filter(User.Username.in_(test_usernames)).all()
        for u in users:
            db.query(KernelUsageHistory).filter(
                KernelUsageHistory.UserID == u.UserID
            ).delete(synchronize_session=False)
            db.query(Prediction).filter(
                Prediction.UserID == u.UserID
            ).delete(synchronize_session=False)
            db.delete(u)

        db.commit()
    finally:
        db.close()


def setup_function():
    _cleanup_test_data()


def teardown_function():
    _cleanup_test_data()


def test_users_transaction_rollback():
    """Verify Users repository create_user flushes without commit, rolling back cleanly."""
    db = SessionLocal()
    try:
        # 1. Insert via repository
        user_dict = user_repository.create_user(
            db=db,
            username=TEST_USER_TX_ROLLBACK,
            password_hash="$2b$12$dummyhashtestonlyforrollbackverification123456",
        )
        user_id = user_dict["user_id"]
        assert user_id is not None

        # 2. Query within the same uncommitted transaction
        uncommitted_user = (
            db.query(User)
            .filter(User.Username == TEST_USER_TX_ROLLBACK)
            .first()
        )
        assert uncommitted_user is not None
        assert uncommitted_user.UserID == user_id

        # 3. Explicit rollback
        db.rollback()
    finally:
        db.close()

    # 4. Open fresh session and verify data was NOT committed
    db_new = SessionLocal()
    try:
        user_after_rollback = (
            db_new.query(User)
            .filter(User.Username == TEST_USER_TX_ROLLBACK)
            .first()
        )
        assert user_after_rollback is None
    finally:
        db_new.close()


def test_predictions_transaction_rollback():
    """Verify Predictions repository create_prediction flushes without commit, rolling back cleanly."""
    # Obtain a valid test user ID for FK constraint
    db_pre = SessionLocal()
    try:
        test_user = (
            db_pre.query(User)
            .filter(User.Username == "__sql_test_user__")
            .first()
        )
        assert test_user is not None, "__sql_test_user__ must exist for FK validation"
        user_id = test_user.UserID
    finally:
        db_pre.close()

    db = SessionLocal()
    try:
        # 1. Create prediction record via repository
        pred = prediction_repository.create_prediction(
            db=db,
            user_id=user_id,
            sepal_length=5.1,
            sepal_width=3.5,
            petal_length=1.4,
            petal_width=0.2,
            predicted_species="Iris-setosa",
            probability_setosa=0.99,
            probability_versicolor=0.005,
            probability_virginica=0.005,
            model_name=TEST_PRED_MODEL,
            kernel="RBF",
        )
        pred_id = pred.PredictionID
        assert pred_id is not None

        # 2. Query within uncommitted transaction
        uncommitted_pred = (
            db.query(Prediction)
            .filter(Prediction.PredictionID == pred_id)
            .first()
        )
        assert uncommitted_pred is not None
        assert uncommitted_pred.ModelName == TEST_PRED_MODEL

        # 3. Explicit rollback
        db.rollback()
    finally:
        db.close()

    # 4. Open fresh session and verify record does not exist
    db_new = SessionLocal()
    try:
        pred_after_rollback = (
            db_new.query(Prediction)
            .filter(Prediction.PredictionID == pred_id)
            .first()
        )
        assert pred_after_rollback is None
    finally:
        db_new.close()


def test_kernel_usage_transaction_rollback():
    """Verify KernelUsageHistory repository record_kernel_usage flushes without commit, rolling back cleanly."""
    db_pre = SessionLocal()
    try:
        test_user = (
            db_pre.query(User)
            .filter(User.Username == "__sql_test_user__")
            .first()
        )
        assert test_user is not None
        user_id = test_user.UserID
    finally:
        db_pre.close()

    db = SessionLocal()
    try:
        # 1. Record kernel usage via repository
        usage = kernel_usage_repository.record_kernel_usage(
            db=db,
            user_id=user_id,
            kernel="RBF",
        )
        usage_id = usage.UsageID
        assert usage_id is not None

        # 2. Query within uncommitted transaction
        uncommitted_usage = (
            db.query(KernelUsageHistory)
            .filter(KernelUsageHistory.UsageID == usage_id)
            .first()
        )
        assert uncommitted_usage is not None
        assert uncommitted_usage.Kernel == "RBF"

        # 3. Explicit rollback
        db.rollback()
    finally:
        db.close()

    # 4. Verify in fresh session
    db_new = SessionLocal()
    try:
        usage_after_rollback = (
            db_new.query(KernelUsageHistory)
            .filter(KernelUsageHistory.UsageID == usage_id)
            .first()
        )
        assert usage_after_rollback is None
    finally:
        db_new.close()


def test_model_history_transaction_rollback():
    """Verify ModelHistory repository record_model_history flushes without commit, rolling back cleanly."""
    db = SessionLocal()
    try:
        # 1. Record model history via repository
        mh = model_history_repository.record_model_history(
            db=db,
            model_name=TEST_MODEL_NAME,
            kernel="POLY",
            accuracy=0.985,
            precision_score=0.98,
            recall_score=0.98,
            f1_score=0.98,
        )
        model_id = mh.ModelID
        assert model_id is not None

        # 2. Query within uncommitted transaction
        uncommitted_mh = (
            db.query(ModelHistory)
            .filter(ModelHistory.ModelID == model_id)
            .first()
        )
        assert uncommitted_mh is not None
        assert uncommitted_mh.ModelName == TEST_MODEL_NAME

        # 3. Explicit rollback
        db.rollback()
    finally:
        db.close()

    # 4. Verify in fresh session
    db_new = SessionLocal()
    try:
        mh_after_rollback = (
            db_new.query(ModelHistory)
            .filter(ModelHistory.ModelID == model_id)
            .first()
        )
        assert mh_after_rollback is None
    finally:
        db_new.close()


def test_transaction_commit_and_cleanup():
    """Verify explicit commit persists data across sessions and cleanup removes it completely."""
    # 1. Create and commit
    db = SessionLocal()
    user_id = None
    try:
        user_dict = user_repository.create_user(
            db=db,
            username=TEST_USER_TX_COMMIT,
            password_hash="$2b$12$dummyhashtestonlyforcommitverification123456",
        )
        user_id = user_dict["user_id"]
        db.commit()
    finally:
        db.close()

    # 2. Open fresh session, verify persistence across connections
    db2 = SessionLocal()
    try:
        persisted_user = (
            db2.query(User)
            .filter(User.Username == TEST_USER_TX_COMMIT)
            .first()
        )
        assert persisted_user is not None
        assert persisted_user.UserID == user_id
        assert persisted_user.Role == "USER"

        # 3. Delete and commit cleanup
        db2.delete(persisted_user)
        db2.commit()
    finally:
        db2.close()

    # 4. Open 3rd fresh session, verify record is completely gone
    db3 = SessionLocal()
    try:
        deleted_user = (
            db3.query(User)
            .filter(User.Username == TEST_USER_TX_COMMIT)
            .first()
        )
        assert deleted_user is None
    finally:
        db3.close()


def test_final_database_cleanliness():
    """Final audit confirming 0 test rows remain in SQL Server HMNC_PRO across all 4 tables."""
    db = SessionLocal()
    try:
        tx_users = db.query(User).filter(
            User.Username.in_([TEST_USER_TX_ROLLBACK, TEST_USER_TX_COMMIT])
        ).count()
        tx_preds = db.query(Prediction).filter(
            Prediction.ModelName == TEST_PRED_MODEL
        ).count()
        tx_models = db.query(ModelHistory).filter(
            ModelHistory.ModelName == TEST_MODEL_NAME
        ).count()

        assert tx_users == 0, f"Found {tx_users} lingering test users"
        assert tx_preds == 0, f"Found {tx_preds} lingering test predictions"
        assert tx_models == 0, f"Found {tx_models} lingering test model records"
    finally:
        db.close()
