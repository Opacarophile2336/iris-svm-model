"""Prediction API router — USER and ADMIN accessible."""
from __future__ import annotations

import io
import math
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from sqlalchemy.orm import Session

from auth.dependencies import require_user, require_admin
from database.connection import get_db
from ml import svm_service
from prediction import service as prediction_service, batch_service, insights_service, export_service

router = APIRouter(prefix="/prediction", tags=["Prediction"])

# In-memory temporary cache for batch download: batch_id -> (results, metrics)
_batch_cache: dict[str, tuple[list[dict], dict]] = {}


# ---------- Schemas ----------

class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, le=20, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, le=20, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, le=20, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, le=20, description="Petal width in cm")
    model: Optional[str] = Field(None, description="Classifier key: 'rbf', 'linear', 'poly', 'sigmoid', 'mlp', or None/'best'")

    @field_validator("sepal_length", "sepal_width", "petal_length", "petal_width", mode="before")
    @classmethod
    def no_nan_inf(cls, v: float) -> float:
        if isinstance(v, (float, int)) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Value must be a finite number.")
        return float(v)

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and isinstance(v, str):
            v_clean = v.strip().lower()
            return v_clean if v_clean else None
        return v


class BatchExportRequest(BaseModel):
    batch_id: Optional[str] = None
    format: str = "xlsx"  # 'xlsx' or 'csv'
    rows: Optional[list[dict]] = None


# ---------- Endpoints ----------

@router.post("/predict")
def predict(req: PredictRequest, current=Depends(require_user), db: Session = Depends(get_db)):
    """Predict Iris species using the selected model (or best model if not specified). Records to history."""
    valid_models = {"rbf", "linear", "poly", "sigmoid", "mlp", "best"}
    if req.model is not None and req.model not in valid_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{req.model}'. Supported models: rbf, linear, poly, sigmoid, mlp",
        )

    try:
        result = svm_service.predict(
            req.sepal_length,
            req.sepal_width,
            req.petal_length,
            req.petal_width,
            model_key=req.model,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {e}")

    # Save to history via prediction_service in SQL Server
    history_id = None
    try:
        probs = result.get("probabilities", {})
        p_setosa = probs.get("setosa", 0.0)
        p_versicolor = probs.get("versicolor", 0.0)
        p_virginica = probs.get("virginica", 0.0)
        record = prediction_service.create_prediction(
            db=db,
            user_id=current["username"],
            sepal_length=req.sepal_length,
            sepal_width=req.sepal_width,
            petal_length=req.petal_length,
            petal_width=req.petal_width,
            predicted_species=result["predicted_species"],
            probability_setosa=p_setosa / 100.0 if p_setosa > 1.0 else p_setosa,
            probability_versicolor=p_versicolor / 100.0 if p_versicolor > 1.0 else p_versicolor,
            probability_virginica=p_virginica / 100.0 if p_virginica > 1.0 else p_virginica,
            model_name=result.get("model", "SVM"),
            kernel=result.get("kernel", "rbf"),
        )
        db.commit()
        history_id = record.PredictionID
    except Exception:
        db.rollback()

    return {**result, "history_id": history_id}


@router.post("/predict-all")
def predict_all(req: PredictRequest, current=Depends(require_user), db: Session = Depends(get_db)):
    """Predict Iris species using all 5 classifiers: RBF, LINEAR, POLY, SIGMOID, MLP.

    Returns predictions, confidences, probabilities, and REAL validated accuracy for each model.
    """
    try:
        multi_result = svm_service.predict_all_kernels(
            req.sepal_length, req.sepal_width, req.petal_length, req.petal_width
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    primary_kernel = multi_result["highest_confidence"]["kernel"]
    primary_info = multi_result["results"][primary_kernel]

    # Save primary result to history via prediction_service in SQL Server
    history_id = str(uuid.uuid4())[:8].upper()
    try:
        probs = primary_info.get("probabilities", {})
        p_setosa = probs.get("setosa", 0.0)
        p_versicolor = probs.get("versicolor", 0.0)
        p_virginica = probs.get("virginica", 0.0)
        record = prediction_service.create_prediction(
            db=db,
            user_id=current["username"],
            sepal_length=req.sepal_length,
            sepal_width=req.sepal_width,
            petal_length=req.petal_length,
            petal_width=req.petal_width,
            predicted_species=primary_info["predicted_species"],
            probability_setosa=p_setosa / 100.0 if p_setosa > 1.0 else p_setosa,
            probability_versicolor=p_versicolor / 100.0 if p_versicolor > 1.0 else p_versicolor,
            probability_virginica=p_virginica / 100.0 if p_virginica > 1.0 else p_virginica,
            model_name="Deep Learning MLP" if primary_kernel == "mlp" else "SVM",
            kernel=primary_kernel,
        )
        db.commit()
        history_id = record.PredictionID
    except Exception:
        db.rollback()

    return {**multi_result, "history_id": history_id}


@router.post("/batch-upload")
async def batch_upload(file: UploadFile = File(...), current=Depends(require_user)):
    """Upload dataset file (.csv, .txt, .xlsx) and run predictions across all 5 classifiers: RBF, LINEAR, POLY, SIGMOID, MLP."""
    filename = file.filename or "dataset.csv"
    ext = filename.lower().split(".")[-1]
    if ext not in ["csv", "txt", "xlsx"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '.{ext}'. Supported formats: .csv, .txt, .xlsx",
        )

    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    try:
        valid_rows, warnings = batch_service.parse_batch_file(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File parsing error: {e}")

    try:
        results_rows, metrics_summary = svm_service.batch_predict_all_kernels(valid_rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    batch_id = str(uuid.uuid4())[:12]
    _batch_cache[batch_id] = (results_rows, metrics_summary)

    # Keep cache bounded
    if len(_batch_cache) > 50:
        oldest = next(iter(_batch_cache))
        _batch_cache.pop(oldest, None)

    return {
        "batch_id": batch_id,
        "filename": filename,
        "total_rows": len(results_rows),
        "warnings": warnings,
        "results": results_rows,
        "metrics": metrics_summary,
    }


@router.post("/batch-export")
def batch_export(req: BatchExportRequest, current=Depends(require_user)):
    """Download batch prediction results as an Excel (.xlsx) or CSV file."""
    results = None
    metrics = None

    if req.batch_id and req.batch_id in _batch_cache:
        results, metrics = _batch_cache[req.batch_id]
    elif req.rows:
        results = req.rows
        metrics = svm_service.get_all_kernels_metrics()

    if not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No batch data available for export. Please perform a batch prediction first.",
        )

    if req.format.lower() == "csv":
        csv_data = batch_service.generate_batch_csv(results)
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=hmnc_iris_batch_predictions.csv"},
        )
    else:
        # Default: XLSX
        xlsx_bytes = batch_service.generate_batch_excel(results, metrics or {})
        return StreamingResponse(
            io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=hmnc_iris_batch_predictions.xlsx"},
        )


@router.post("/insights")
def get_insights(req: PredictRequest, current=Depends(require_user)):
    """Deep botanical and machine learning diagnostics comparing sample with canonical Iris benchmarks."""
    try:
        multi_result = svm_service.predict_all_kernels(
            req.sepal_length, req.sepal_width, req.petal_length, req.petal_width
        )
        insights = insights_service.analyze_sample_insights(
            req.sepal_length, req.sepal_width, req.petal_length, req.petal_width, multi_result
        )
        return {**multi_result, "insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {e}")


@router.get("/metrics")
def get_model_metrics(current=Depends(require_user)):
    """Return real validated metrics for all classifiers."""
    return svm_service.get_all_kernels_metrics()


@router.get("/history")
def user_history(
    page: Optional[int] = Query(None, ge=1, description="Page number"),
    page_size: Optional[int] = Query(None, ge=1, le=100, description="Items per page"),
    species: Optional[str] = Query(None, description="Filter by species: setosa, versicolor, virginica"),
    kernel: Optional[str] = Query(None, description="Filter by kernel: rbf, linear, poly, sigmoid, mlp"),
    model: Optional[str] = Query(None, description="Filter by model name: SVM, Deep Learning MLP"),
    start_date: Optional[datetime] = Query(None, description="Filter from start date"),
    end_date: Optional[datetime] = Query(None, description="Filter to end date"),
    current=Depends(require_user),
    db: Session = Depends(get_db),
):
    """Return the current user's prediction history from SQL Server.

    When pagination or filter query params are provided, returns paginated object.
    When omitted, returns list format for full backward compatibility.
    """
    if any(param is not None for param in (page, page_size, species, kernel, model, start_date, end_date)):
        return prediction_service.get_user_history_paginated(
            db=db,
            user_identity=current["username"],
            page=page or 1,
            page_size=page_size or 20,
            species=species,
            kernel=kernel,
            model=model,
            start_date=start_date,
            end_date=end_date,
        )
    return prediction_service.get_user_history(db, current["username"])


@router.get("/history/all")
def all_history(
    page: Optional[int] = Query(None, ge=1, description="Page number"),
    page_size: Optional[int] = Query(None, ge=1, le=100, description="Items per page"),
    species: Optional[str] = Query(None, description="Filter by species"),
    kernel: Optional[str] = Query(None, description="Filter by kernel"),
    model: Optional[str] = Query(None, description="Filter by model"),
    start_date: Optional[datetime] = Query(None, description="Filter from start date"),
    end_date: Optional[datetime] = Query(None, description="Filter to end date"),
    search: Optional[str] = Query(None, description="Search by username"),
    current=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Return all users' prediction history from SQL Server (admin only).

    Supports pagination, filtering, and search by username.
    """
    if any(param is not None for param in (page, page_size, species, kernel, model, start_date, end_date, search)):
        return prediction_service.get_all_history_paginated(
            db=db,
            page=page or 1,
            page_size=page_size or 20,
            species=species,
            kernel=kernel,
            model=model,
            start_date=start_date,
            end_date=end_date,
            username_search=search,
        )
    return prediction_service.get_all_history(db)


@router.delete("/history/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    current=Depends(require_user),
    db: Session = Depends(get_db),
):
    """Delete a single prediction record with strict ownership enforcement.

    Users can only delete their own records. Admins can delete any record.
    """
    is_admin = (current.get("role") == "ADMIN")
    result = prediction_service.delete_prediction(
        db=db,
        prediction_id=prediction_id,
        user_identity=current["username"],
        is_admin=is_admin,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction #{prediction_id} not found.",
        )
    if result is False:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You do not have permission to delete prediction #{prediction_id}.",
        )

    db.commit()
    return {
        "success": True,
        "message": f"Prediction #{prediction_id} deleted successfully.",
        "prediction_id": prediction_id,
    }


@router.delete("/history")
def clear_history(
    current=Depends(require_user),
    db: Session = Depends(get_db),
):
    """Clear all predictions belonging to the currently logged in user.

    Admins calling this endpoint will only clear predictions created by the admin account,
    never clearing all system predictions.
    """
    deleted_count = prediction_service.clear_user_history(
        db=db,
        user_identity=current["username"],
    )
    db.commit()
    return {
        "success": True,
        "message": f"Successfully deleted {deleted_count} prediction record(s).",
        "count": deleted_count,
    }


@router.get("/history/export")
def export_history(
    format: str = Query("xlsx", pattern="^(xlsx|csv)$", description="Export format: xlsx or csv"),
    species: Optional[str] = Query(None),
    kernel: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    search: Optional[str] = Query(None),
    current=Depends(require_user),
    db: Session = Depends(get_db),
):
    """Export prediction history as Excel (.xlsx) or CSV file.

    Regular users export their own records; Admins can export all records.
    """
    is_admin = (current.get("role") == "ADMIN")
    records = prediction_service.get_history_for_export(
        db=db,
        user_identity=current["username"],
        is_admin=is_admin,
        species=species,
        kernel=kernel,
        model=model,
        start_date=start_date,
        end_date=end_date,
        username_search=search if is_admin else None,
    )

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if format.lower() == "csv":
        csv_data = export_service.generate_history_csv(records, is_admin=is_admin)
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=iris_prediction_history_{timestamp_str}.csv",
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )
    else:
        xlsx_bytes = export_service.generate_history_excel(records, is_admin=is_admin)
        return StreamingResponse(
            io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=iris_prediction_history_{timestamp_str}.xlsx",
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )
