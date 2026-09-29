"""Datasets API router — accessible to all authenticated users."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from auth.dependencies import require_user
from datasets.iris_dataset import load_iris_data, get_quality_stats, get_feature_dictionary

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("/iris")
def iris_data(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=150),
    current=Depends(require_user),
):
    """Return paginated Iris dataset records."""
    data = load_iris_data()
    records = data["records"]
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "total": data["total"],
        "page": page,
        "page_size": page_size,
        "class_distribution": data["class_distribution"],
        "records": records[start:end],
    }


@router.get("/quality")
def data_quality(current=Depends(require_user)):
    """Return data quality statistics."""
    return get_quality_stats()


@router.get("/dictionary")
def feature_dictionary(current=Depends(require_user)):
    """Return feature dictionary/metadata."""
    return get_feature_dictionary()
