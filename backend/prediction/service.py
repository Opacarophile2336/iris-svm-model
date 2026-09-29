"""Prediction service — business logic for storing, querying, deleting, and exporting Iris predictions.

Integrates with prediction_repository using SQLAlchemy ORM Sessions.
Does NOT commit transactions directly; transaction management (commit/rollback)
is left to the caller/controller layer.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, List, Optional, Union

from sqlalchemy.orm import Session

from database.models import Prediction
from database.repositories import user_repository
from prediction import prediction_repository


def _resolve_user_id(db: Session, user_identity: Union[int, str], raise_if_not_found: bool = False) -> Optional[int]:
    """Helper to resolve username or user_id to actual integer user_id.

    Returns None if user is not found in database (e.g. admin or unregistered user).
    If raise_if_not_found is True, raises ValueError instead.
    """
    if isinstance(user_identity, str):
        user = user_repository.find_user(db, user_identity)
        if not user:
            if raise_if_not_found:
                raise ValueError(f"User '{user_identity}' not found.")
            return None
        return user["user_id"]
    return user_identity


def create_prediction(
    db: Session,
    user_id: Union[int, str],
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
    """Create and record a prediction record in SQL Server via prediction_repository."""
    actual_user_id = _resolve_user_id(db, user_id, raise_if_not_found=True)
    return prediction_repository.create_prediction(
        db=db,
        user_id=actual_user_id,
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
        predicted_species=predicted_species,
        probability_setosa=probability_setosa,
        probability_versicolor=probability_versicolor,
        probability_virginica=probability_virginica,
        model_name=model_name,
        kernel=kernel,
        created_at=created_at,
    )


# Compatibility aliases
save_prediction = create_prediction
add_prediction = create_prediction


def delete_prediction(
    db: Session,
    prediction_id: int,
    user_identity: Union[int, str],
    is_admin: bool = False,
) -> Optional[bool]:
    """Delete a prediction record with ownership enforcement.

    Returns:
        True: Successfully deleted.
        False: Record exists but belongs to another user (unauthorized for non-admin).
        None: Record not found.
    """
    actual_user_id = _resolve_user_id(db, user_identity, raise_if_not_found=False)
    if is_admin:
        return prediction_repository.delete_prediction(
            db=db,
            prediction_id=prediction_id,
            user_id=actual_user_id or 0,
            is_admin=True,
        )
    if actual_user_id is None:
        return False
    return prediction_repository.delete_prediction(
        db=db,
        prediction_id=prediction_id,
        user_id=actual_user_id,
        is_admin=False,
    )


def clear_user_history(
    db: Session,
    user_identity: Union[int, str],
) -> int:
    """Delete all prediction records belonging to the current user."""
    actual_user_id = _resolve_user_id(db, user_identity, raise_if_not_found=False)
    if actual_user_id is None:
        return 0
    return prediction_repository.clear_user_history(
        db=db,
        user_id=actual_user_id,
    )


def get_predictions_by_user(
    db: Session,
    user_id: Union[int, str],
    limit: Optional[int] = None,
) -> List[Prediction]:
    """Retrieve predictions for a given user (by user_id or username) via repository."""
    return prediction_repository.get_predictions_by_user(
        db=db,
        user_id=user_id,
        limit=limit,
    )


def get_all_predictions(
    db: Session,
    limit: Optional[int] = None,
) -> List[Prediction]:
    """Retrieve all predictions for admin view via repository."""
    return prediction_repository.get_all_predictions(
        db=db,
        limit=limit,
    )


def get_user_history(
    db: Session,
    user_id: Union[int, str],
    limit: Optional[int] = None,
) -> List[dict[str, Any]]:
    """Retrieve user prediction history as serializable dictionaries (backward compatible)."""
    records = get_predictions_by_user(db, user_id, limit=limit)
    return [prediction_repository.prediction_to_dict(p) for p in records]


def get_all_history(
    db: Session,
    limit: Optional[int] = None,
) -> List[dict[str, Any]]:
    """Retrieve all prediction history as serializable dictionaries for admin (backward compatible)."""
    records = get_all_predictions(db, limit=limit)
    return [prediction_repository.prediction_to_dict(p) for p in records]


def get_user_history_paginated(
    db: Session,
    user_identity: Union[int, str],
    page: int = 1,
    page_size: int = 20,
    species: Optional[str] = None,
    kernel: Optional[str] = None,
    model: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> dict[str, Any]:
    actual_user_id = _resolve_user_id(db, user_identity, raise_if_not_found=False)
    if actual_user_id is None:
        return {
            "items": [],
            "page": page,
            "page_size": page_size,
            "total": 0,
            "total_pages": 1,
        }
    records, total = prediction_repository.query_predictions(
        db=db,
        user_id=actual_user_id,
        species=species,
        kernel=kernel,
        model=model,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size,
    )
    items = [prediction_repository.prediction_to_dict(p) for p in records]
    total_pages = math.ceil(total / page_size) if page_size > 0 else 1
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": max(1, total_pages),
    }


def get_all_history_paginated(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    species: Optional[str] = None,
    kernel: Optional[str] = None,
    model: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    username_search: Optional[str] = None,
) -> dict[str, Any]:
    """Retrieve paginated, filtered, and searchable prediction history across all users for admin."""
    records, total = prediction_repository.query_predictions(
        db=db,
        user_id=None,
        species=species,
        kernel=kernel,
        model=model,
        start_date=start_date,
        end_date=end_date,
        username_search=username_search,
        page=page,
        page_size=page_size,
    )
    items = [prediction_repository.prediction_to_dict(p) for p in records]
    total_pages = math.ceil(total / page_size) if page_size > 0 else 1
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": max(1, total_pages),
    }


def get_history_for_export(
    db: Session,
    user_identity: Optional[Union[int, str]] = None,
    is_admin: bool = False,
    species: Optional[str] = None,
    kernel: Optional[str] = None,
    model: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    username_search: Optional[str] = None,
) -> List[dict[str, Any]]:
    """Retrieve all matching records for CSV or Excel export."""
    target_user_id = None
    if not is_admin:
        if user_identity is None:
            raise ValueError("user_identity required for non-admin export.")
        target_user_id = _resolve_user_id(db, user_identity, raise_if_not_found=False)
        if target_user_id is None:
            return []

    records, _ = prediction_repository.query_predictions(
        db=db,
        user_id=target_user_id,
        species=species,
        kernel=kernel,
        model=model,
        start_date=start_date,
        end_date=end_date,
        username_search=username_search if is_admin else None,
        page=None,
        page_size=None,
    )
    return [prediction_repository.prediction_to_dict(p) for p in records]
