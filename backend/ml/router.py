"""ML/SVM API router — all endpoints are ADMIN-only except GET /ml/model-info."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth.dependencies import require_admin, require_user
from database.connection import get_db
from database.repositories import model_history_repository
from ml import svm_service, decision_boundary
import config

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


# ---------- Schemas ----------

class TrainResponse(BaseModel):
    message: str
    best_kernel: str
    cv_score: float


# ---------- Endpoints ----------

@router.get("/model-info")
def model_info(current=Depends(require_user)):
    """Any authenticated user can see best model info (kernel, accuracy)."""
    return svm_service.get_best_model_info()


@router.post("/train")
def train(current=Depends(require_admin)):
    """Retrain all supported kernels, select best by CV accuracy."""
    try:
        results = svm_service.train_all_kernels(save=True)
        best = results["best_kernel"]
        return {
            "message": "Training complete.",
            "best_kernel": best,
            "kernel_display": svm_service.KERNEL_DISPLAY.get(best, best),
            "cv_score": results[best]["cv_mean"],
            "results": {k: v for k, v in results.items() if k != "best_kernel"},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation")
def evaluation(current=Depends(require_admin)):
    """Per-kernel evaluation metrics (accuracy, precision, recall, F1, CM)."""
    try:
        return svm_service.get_evaluation_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decision-boundary")
def decision_boundary_endpoint(
    kernel: str = Query("rbf", description="Kernel: rbf, linear, poly, sigmoid"),
    feature_x: int = Query(2, description="Feature index for X axis (0-3)"),
    feature_y: int = Query(3, description="Feature index for Y axis (0-3)"),
    current=Depends(require_admin),
):
    """Return base64-encoded PNG of 2D decision boundary."""
    if kernel not in config.ALLOWED_KERNELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported kernel '{kernel}'. Allowed: {config.ALLOWED_KERNELS}",
        )
    try:
        image_b64 = decision_boundary.generate_decision_boundary(kernel, feature_x, feature_y)
        return {"kernel": kernel, "image_base64": image_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-history")
def model_history(current=Depends(require_admin), db: Session = Depends(get_db)):
    """Return all model training history entries from SQL Server (with fallback)."""
    if db is not None:
        try:
            sql_records = model_history_repository.get_model_history(db)
            if sql_records:
                return [model_history_repository.model_history_to_dict(r) for r in sql_records]
        except Exception as e:
            logger.warning(f"Could not query model history from SQL Server: {e}")

    import json
    path = config.MODEL_HISTORY_PATH
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read model history: {e}")


@router.get("/kernels")
def get_kernels(current=Depends(require_user)):
    """Return list of supported kernels and classifiers."""
    return {
        "kernels": config.ALLOWED_KERNELS,
        "display": svm_service.KERNEL_DISPLAY,
    }

