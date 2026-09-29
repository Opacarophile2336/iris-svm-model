"""Prediction repository — SQLAlchemy ORM operations for the Predictions table."""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Union

from sqlalchemy.orm import Session

from database.models import Prediction, User

KERNEL_DISPLAY_MAP = {
    "rbf": "SVM RBF",
    "linear": "SVM Linear",
    "poly": "SVM Polynomial",
    "sigmoid": "SVM Sigmoid",
    "mlp": "Deep Learning MLP",
}


def create_prediction(
    db: Session,
    user_id: int,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    predicted_species: str,
    probability_setosa: Optional[float] = None,
    probability_versicolor: Optional[float] = None,
    probability_virginica: Optional[float] = None,
    model_name: Optional[str] = None,
    kernel: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> Prediction:
    """Create a new Prediction record in SQL Server using SQLAlchemy ORM.

    Uses db.flush() and db.refresh() to obtain generated IDs without committing.
    Transaction management (commit / rollback) is left to the calling service layer.
    """
    prediction = Prediction(
        UserID=user_id,
        CreatedAt=created_at or datetime.utcnow(),
        SepalLength=sepal_length,
        SepalWidth=sepal_width,
        PetalLength=petal_length,
        PetalWidth=petal_width,
        PredictedSpecies=predicted_species,
        ProbabilitySetosa=probability_setosa,
        ProbabilityVersicolor=probability_versicolor,
        ProbabilityVirginica=probability_virginica,
        ModelName=model_name,
        Kernel=kernel,
    )
    db.add(prediction)
    db.flush()
    db.refresh(prediction)
    return prediction


def delete_prediction(
    db: Session,
    prediction_id: int,
    user_id: int,
    is_admin: bool = False,
) -> Optional[bool]:
    """Delete a single prediction record with ownership enforcement.

    Returns:
        True: Successfully deleted.
        False: Record exists but belongs to another user (unauthorized for non-admin).
        None: Record not found.
    """
    prediction = db.query(Prediction).filter(Prediction.PredictionID == prediction_id).first()
    if not prediction:
        return None
    if not is_admin and prediction.UserID != user_id:
        return False
    db.delete(prediction)
    db.flush()
    return True


def clear_user_history(
    db: Session,
    user_id: int,
) -> int:
    """Delete all prediction records belonging to a specific user.

    Returns the count of deleted records.
    """
    count = db.query(Prediction).filter(Prediction.UserID == user_id).delete(synchronize_session=False)
    db.flush()
    return count


def query_predictions(
    db: Session,
    user_id: Optional[Union[int, str]] = None,
    species: Optional[str] = None,
    kernel: Optional[str] = None,
    model: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    username_search: Optional[str] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
) -> tuple[List[Prediction], int]:
    """Query prediction records with filtering, searching, and pagination."""
    query = db.query(Prediction).join(User, Prediction.UserID == User.UserID)

    if user_id is not None:
        if isinstance(user_id, str):
            query = query.filter(User.Username == user_id)
        else:
            query = query.filter(Prediction.UserID == user_id)

    if username_search:
        query = query.filter(User.Username.ilike(f"%{username_search.strip()}%"))

    if species:
        s_clean = species.strip().lower()
        query = query.filter(Prediction.PredictedSpecies.ilike(f"%{s_clean}%"))

    if kernel:
        k_clean = kernel.strip().lower()
        query = query.filter(Prediction.Kernel.ilike(k_clean))

    if model:
        m_clean = model.strip().lower()
        query = query.filter(Prediction.ModelName.ilike(f"%{m_clean}%"))

    if start_date:
        query = query.filter(Prediction.CreatedAt >= start_date)

    if end_date:
        query = query.filter(Prediction.CreatedAt <= end_date)

    total = query.count()
    query = query.order_by(Prediction.CreatedAt.desc())

    if page is not None and page_size is not None and page > 0 and page_size > 0:
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

    records = query.all()
    return records, total


def get_predictions_by_user(
    db: Session,
    user_id: Union[int, str],
    limit: Optional[int] = None,
) -> List[Prediction]:
    """Retrieve predictions for a specific user (by user_id or username), newest first."""
    if isinstance(user_id, str):
        query = (
            db.query(Prediction)
            .join(User, Prediction.UserID == User.UserID)
            .filter(User.Username == user_id)
            .order_by(Prediction.CreatedAt.desc())
        )
    else:
        query = (
            db.query(Prediction)
            .filter(Prediction.UserID == user_id)
            .order_by(Prediction.CreatedAt.desc())
        )

    if limit is not None and limit > 0:
        query = query.limit(limit)

    return query.all()


def get_all_predictions(
    db: Session,
    limit: Optional[int] = None,
) -> List[Prediction]:
    """Retrieve all prediction records for ADMIN view (newest first)."""
    query = db.query(Prediction).order_by(Prediction.CreatedAt.desc())

    if limit is not None and limit > 0:
        query = query.limit(limit)

    return query.all()


def prediction_to_dict(prediction: Prediction) -> dict[str, Any]:
    """Convert a Prediction ORM instance into a serializable dictionary compatible with SQL and API models."""
    p_setosa = prediction.ProbabilitySetosa or 0.0
    p_versicolor = prediction.ProbabilityVersicolor or 0.0
    p_virginica = prediction.ProbabilityVirginica or 0.0
    max_prob = max(p_setosa, p_versicolor, p_virginica)

    kernel_raw = (prediction.Kernel or "").lower().strip()
    k_disp = KERNEL_DISPLAY_MAP.get(kernel_raw, (prediction.Kernel or "").upper())

    return {
        # Core DB fields
        "prediction_id": prediction.PredictionID,
        "user_id": prediction.UserID,
        "username": prediction.user.Username if prediction.user else None,
        "created_at": prediction.CreatedAt.isoformat() if prediction.CreatedAt else None,
        "sepal_length": prediction.SepalLength,
        "sepal_width": prediction.SepalWidth,
        "petal_length": prediction.PetalLength,
        "petal_width": prediction.PetalWidth,
        "predicted_species": prediction.PredictedSpecies,
        "probability_setosa": prediction.ProbabilitySetosa,
        "probability_versicolor": prediction.ProbabilityVersicolor,
        "probability_virginica": prediction.ProbabilityVirginica,
        "model_name": prediction.ModelName,
        "kernel": prediction.Kernel,

        # API & Frontend contract fields
        "id": str(prediction.PredictionID),
        "timestamp": prediction.CreatedAt.isoformat() if prediction.CreatedAt else None,
        "features": {
            "sepal_length": prediction.SepalLength,
            "sepal_width": prediction.SepalWidth,
            "petal_length": prediction.PetalLength,
            "petal_width": prediction.PetalWidth,
        },
        "probabilities": {
            "setosa": round(p_setosa * (100 if p_setosa <= 1.0 else 1), 1),
            "versicolor": round(p_versicolor * (100 if p_versicolor <= 1.0 else 1), 1),
            "virginica": round(p_virginica * (100 if p_virginica <= 1.0 else 1), 1),
        },
        "confidence": round(max_prob * (100 if max_prob <= 1.0 else 1), 1),
        "model": prediction.ModelName or "SVM",
        "kernel_display": k_disp,
    }
