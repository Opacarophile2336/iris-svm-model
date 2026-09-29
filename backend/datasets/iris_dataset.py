"""Iris dataset service — load, quality, dictionary."""
from __future__ import annotations

import numpy as np
from sklearn import datasets


def load_iris_data() -> dict:
    """Return full Iris dataset as list of records."""
    iris = datasets.load_iris()
    records = []
    for i, (row, label) in enumerate(zip(iris.data, iris.target)):
        species_map = {0: "Iris setosa", 1: "Iris versicolor", 2: "Iris virginica"}
        records.append({
            "id": i + 1,
            "sepal_length": float(row[0]),
            "sepal_width": float(row[1]),
            "petal_length": float(row[2]),
            "petal_width": float(row[3]),
            "species": species_map[label],
            "class_id": int(label),
        })
    return {
        "total": len(records),
        "records": records,
        "class_distribution": {
            "Iris setosa": int(np.sum(iris.target == 0)),
            "Iris versicolor": int(np.sum(iris.target == 1)),
            "Iris virginica": int(np.sum(iris.target == 2)),
        },
    }


def get_quality_stats() -> dict:
    """Return quality statistics for the Iris dataset."""
    iris = datasets.load_iris()
    X = iris.data
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    stats = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        stats[name] = {
            "count": int(len(col)),
            "missing": 0,
            "mean": round(float(np.mean(col)), 4),
            "std": round(float(np.std(col)), 4),
            "min": round(float(np.min(col)), 4),
            "q25": round(float(np.percentile(col, 25)), 4),
            "median": round(float(np.median(col)), 4),
            "q75": round(float(np.percentile(col, 75)), 4),
            "max": round(float(np.max(col)), 4),
            "missing_pct": 0.0,
        }

    return {
        "total_records": int(X.shape[0]),
        "total_features": int(X.shape[1]),
        "missing_values": 0,
        "missing_pct": 0.0,
        "feature_stats": stats,
        "class_balance": "balanced (50 per class)",
    }


def get_feature_dictionary() -> list[dict]:
    """Return feature dictionary/metadata."""
    return [
        {
            "feature": "sepal_length",
            "display_name": "Sepal Length",
            "unit": "cm",
            "description": "Length of the sepal (the outer parts of the flower).",
            "range": "4.3 – 7.9",
            "type": "continuous",
        },
        {
            "feature": "sepal_width",
            "display_name": "Sepal Width",
            "unit": "cm",
            "description": "Width of the sepal.",
            "range": "2.0 – 4.4",
            "type": "continuous",
        },
        {
            "feature": "petal_length",
            "display_name": "Petal Length",
            "unit": "cm",
            "description": "Length of the petal (the inner colorful parts of the flower).",
            "range": "1.0 – 6.9",
            "type": "continuous",
        },
        {
            "feature": "petal_width",
            "display_name": "Petal Width",
            "unit": "cm",
            "description": "Width of the petal.",
            "range": "0.1 – 2.5",
            "type": "continuous",
        },
        {
            "feature": "species",
            "display_name": "Species",
            "unit": "",
            "description": "Target class: Iris setosa, Iris versicolor, or Iris virginica.",
            "range": "3 classes",
            "type": "categorical",
        },
    ]
