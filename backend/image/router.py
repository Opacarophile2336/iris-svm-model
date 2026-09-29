"""Image API router — returns species image URLs and dynamic specimen showcase."""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from auth.dependencies import require_user
from image.species_image import (
    get_species_image,
    get_specimen_gallery,
    get_specimen_showcase,
)
import config

router = APIRouter(prefix="/image", tags=["Species Images"])


@router.get("/species/{species_name}")
def species_image(
    species_name: str,
    exclude: Optional[str] = Query(None, description="Optional image ID or URL to exclude for rotation"),
    current=Depends(require_user),
):
    """Retrieve image for an Iris species. species_name format: 'Iris setosa' etc."""
    # Normalize case-insensitively against config.IRIS_CLASSES
    raw_lower = species_name.strip().lower().replace("_", " ")
    matched = None
    for cls_name in config.IRIS_CLASSES:
        if raw_lower == cls_name.lower() or raw_lower == cls_name.lower().replace("iris ", ""):
            matched = cls_name
            break

    if not matched:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown species '{species_name}'. Supported: {config.IRIS_CLASSES}",
        )

    result = get_species_image(matched, exclude_id=exclude)
    return result


@router.get("/showcase/{species_name}")
def specimen_showcase(
    species_name: str,
    count: int = Query(3, ge=1, le=5, description="Number of distinct specimens to showcase"),
    exclude: Optional[str] = Query(None, description="Comma-separated IDs of recently shown images to avoid"),
    current=Depends(require_user),
):
    """Retrieve dynamic diverse specimen showcase for predicted species.

    Guarantees:
    - All specimens belong strictly to the predicted species.
    - Zero duplicates within the showcase.
    - History-aware rotation avoiding recently shown IDs.
    """
    raw_lower = species_name.strip().lower().replace("_", " ")
    matched = None
    for cls_name in config.IRIS_CLASSES:
        if raw_lower == cls_name.lower() or raw_lower == cls_name.lower().replace("iris ", ""):
            matched = cls_name
            break

    if not matched:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown species '{species_name}'. Supported: {config.IRIS_CLASSES}",
        )

    exclude_list = [x.strip() for x in exclude.split(",") if x.strip()] if exclude else []
    return get_specimen_showcase(matched, count=count, exclude_ids=exclude_list)


@router.get("/gallery")
def specimen_gallery_all(
    species: Optional[str] = None,
    current=Depends(require_user),
):
    """Retrieve authentic botanical specimen gallery for all or filtered species."""
    if species:
        raw_lower = species.strip().lower().replace("_", " ")
        matched = None
        for cls_name in config.IRIS_CLASSES:
            if raw_lower == cls_name.lower() or raw_lower == cls_name.lower().replace("iris ", ""):
                matched = cls_name
                break
        if not matched:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown species '{species}'. Supported: {config.IRIS_CLASSES}",
            )
        return get_specimen_gallery(matched)

    return get_specimen_gallery(None)


@router.get("/gallery/{species_name}")
def specimen_gallery_by_species(
    species_name: str,
    current=Depends(require_user),
):
    """Retrieve authentic botanical specimen gallery for a specific Iris species."""
    raw_lower = species_name.strip().lower().replace("_", " ")
    matched = None
    for cls_name in config.IRIS_CLASSES:
        if raw_lower == cls_name.lower() or raw_lower == cls_name.lower().replace("iris ", ""):
            matched = cls_name
            break

    if not matched:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown species '{species_name}'. Supported: {config.IRIS_CLASSES}",
        )

    return get_specimen_gallery(matched)
