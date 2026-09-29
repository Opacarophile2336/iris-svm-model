"""Prediction history repository — JSON-based per-user storage."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

_lock = threading.Lock()


def _load() -> list[dict]:
    path = config.PREDICTION_HISTORY_PATH
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(records: list[dict]) -> None:
    path = config.PREDICTION_HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def add_prediction(
    username: str,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    predicted_species: str,
    probabilities: dict,
    confidence: float,
    model: str,
    kernel: str,
    kernel_display: str,
) -> dict:
    """Append a new prediction record and return it."""
    with _lock:
        records = _load()
        record = {
            "id": str(uuid.uuid4())[:8].upper(),
            "username": username,
            "timestamp": datetime.now().isoformat(),
            "features": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
            },
            "predicted_species": predicted_species,
            "probabilities": probabilities,
            "confidence": confidence,
            "model": model,
            "kernel": kernel,
            "kernel_display": kernel_display,
        }
        records.append(record)
        _save(records)
        return record


def get_user_history(username: str) -> list[dict]:
    """Return prediction history for a specific user (newest first)."""
    with _lock:
        records = _load()
    user_records = [r for r in records if r.get("username") == username]
    return list(reversed(user_records))


def get_all_history() -> list[dict]:
    """Return all prediction history (admin use, newest first)."""
    with _lock:
        records = _load()
    return list(reversed(records))
