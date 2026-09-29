"""Repository for ModelHistory database operations in SQL Server."""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy.orm import Session

import config
from database.models import ModelHistory

KERNEL_DISPLAY_MAP = {
    "rbf": "RBF",
    "linear": "Linear",
    "poly": "Polynomial",
    "sigmoid": "Sigmoid",
    "mlp": "Deep Learning MLP",
}


def record_model_history(
    db: Session,
    model_name: str,
    kernel: str,
    accuracy: Optional[float] = None,
    precision_score: Optional[float] = None,
    recall_score: Optional[float] = None,
    f1_score: Optional[float] = None,
    created_at: Optional[datetime] = None,
) -> ModelHistory:
    """Record a model training entry in SQL Server.

    Validates that kernel/model is one of the supported classifiers (rbf, linear, poly, sigmoid, mlp).
    Uses db.flush() and db.refresh() without committing.
    """
    kernel_clean = kernel.strip().lower()
    allowed = [k.lower() for k in config.ALLOWED_KERNELS] + ["mlp"]
    if kernel_clean not in allowed:
        raise ValueError(
            f"Invalid or unsupported kernel '{kernel}'. Allowed: {allowed}"
        )

    record = ModelHistory(
        ModelName=model_name,
        Kernel=kernel_clean.upper(),
        Accuracy=accuracy,
        PrecisionScore=precision_score,
        RecallScore=recall_score,
        F1Score=f1_score,
        CreatedAt=created_at or datetime.utcnow(),
    )
    db.add(record)
    db.flush()
    db.refresh(record)
    return record


def get_model_history(
    db: Session,
    limit: Optional[int] = None,
) -> List[ModelHistory]:
    """Retrieve model training history records from SQL Server, newest first."""
    query = db.query(ModelHistory).order_by(ModelHistory.CreatedAt.desc())
    if limit is not None and limit > 0:
        query = query.limit(limit)
    return query.all()


def model_history_to_dict(record: ModelHistory) -> dict[str, Any]:
    """Convert a ModelHistory ORM instance to a dictionary compatible with API and frontend."""
    kernel_lower = (record.Kernel or "").lower()
    k_disp = KERNEL_DISPLAY_MAP.get(kernel_lower, (record.Kernel or "").upper())

    return {
        # Core SQL ModelHistory fields
        "model_id": record.ModelID,
        "model_name": record.ModelName,
        "kernel": record.Kernel,
        "accuracy": record.Accuracy,
        "precision_score": record.PrecisionScore,
        "recall_score": record.RecallScore,
        "f1_score": record.F1Score,
        "created_at": record.CreatedAt.isoformat() if record.CreatedAt else None,

        # Frontend compatibility fields
        "model_type": record.ModelName,
        "best_kernel": kernel_lower,
        "kernel_display": k_disp,
        "training_date": record.CreatedAt.isoformat() if record.CreatedAt else None,
        "cv_score": record.Accuracy,
        "test_accuracy": record.Accuracy,
    }
