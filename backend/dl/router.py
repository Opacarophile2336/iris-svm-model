"""Deep Learning API router."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from auth.dependencies import require_user
from dl.dl_service import get_dl_status, DLPipeline

router = APIRouter(prefix="/dl", tags=["Deep Learning"])


@router.get("/status")
def dl_status(current=Depends(require_user)):
    """Return DL module status and pipeline stages."""
    return get_dl_status()


@router.get("/pipeline-info")
def pipeline_info(current=Depends(require_user)):
    """Return DL pipeline configuration."""
    pipeline = DLPipeline()
    return {
        "data_config": pipeline.data_config.__dict__,
        "preprocessing_config": pipeline.preprocessing_config.__dict__,
        "model_config": pipeline.model_config.__dict__,
        "training_config": pipeline.training_config.__dict__,
        "pipeline_stages": [
            {"stage": "Data", "description": "Load and validate dataset"},
            {"stage": "Preprocessing", "description": "Normalize, split, encode"},
            {"stage": "Model", "description": "Build neural network architecture"},
            {"stage": "Training", "description": "Train model with optimizer"},
            {"stage": "Evaluation", "description": "Compute accuracy, F1, confusion matrix"},
            {"stage": "Prediction", "description": "Run inference on new samples"},
        ],
    }
