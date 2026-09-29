"""Batch file parsing and export service for Prediction Studio.

Supports:
- Reading CSV, TXT, XLSX dataset files
- Normalizing feature columns safely
- Validating numeric values and ranges
- Generating formatted XLSX and CSV reports with 3-kernel predictions & real accuracies
"""
from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import pandas as pd

# Target feature keys
TARGET_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

FEATURE_DISPLAY = {
    "sepal_length": "Sepal Length (cm)",
    "sepal_width": "Sepal Width (cm)",
    "petal_length": "Petal Length (cm)",
    "petal_width": "Petal Width (cm)",
}

# Aliases for robust, safe column mapping
COLUMN_ALIASES = {
    "sepal_length": [
        "sepal_length", "sepallength", "sepal length", "sepal.length",
        "sepal_length_cm", "sepal length (cm)", "sepal length cm", "sepallengthcm", "sl"
    ],
    "sepal_width": [
        "sepal_width", "sepalwidth", "sepal width", "sepal.width",
        "sepal_width_cm", "sepal width (cm)", "sepal width cm", "sepalwidthcm", "sw"
    ],
    "petal_length": [
        "petal_length", "petallength", "petal length", "petal.length",
        "petal_length_cm", "petal length (cm)", "petal length cm", "petallengthcm", "pl"
    ],
    "petal_width": [
        "petal_width", "petalwidth", "petal width", "petal.width",
        "petal_width_cm", "petal width (cm)", "petal width cm", "petalwidthcm", "pw"
    ],
}

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
MAX_ROWS = 5000  # Max batch size limit


def _clean_header(name: str) -> str:
    """Normalize string for fuzzy alias matching."""
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-_\.\(\)]+", "", s)
    return s


def map_columns(headers: list[str]) -> dict[str, str]:
    """Map raw file headers to standard target features.

    Returns:
        dict: {target_feature: original_header_name}
    Raises:
        ValueError if any of the 4 features cannot be mapped confidently.
    """
    cleaned_map = {_clean_header(h): h for h in headers}
    mapped = {}

    for target, aliases in COLUMN_ALIASES.items():
        found = False
        for alias in aliases:
            cleaned_alias = _clean_header(alias)
            if cleaned_alias in cleaned_map:
                mapped[target] = cleaned_map[cleaned_alias]
                found = True
                break
        if not found:
            # Check if any header starts with target (e.g., 'sepal length ...')
            for c_head, orig_head in cleaned_map.items():
                if c_head.startswith(_clean_header(target)):
                    mapped[target] = orig_head
                    found = True
                    break

    missing = [FEATURE_DISPLAY[f] for f in TARGET_FEATURES if f not in mapped]
    if missing:
        raise ValueError(
            f"Could not find required columns: {', '.join(missing)}. "
            f"Please ensure your file has columns for Sepal Length, Sepal Width, Petal Length, and Petal Width."
        )

    return mapped


def parse_batch_file(file_bytes: bytes, filename: str) -> tuple[list[dict], list[str]]:
    """Parse CSV, TXT, or XLSX file safely and extract valid numeric sample rows.

    Returns:
        (valid_rows, warning_messages)
    """
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError("File size exceeds 10 MB limit. Please upload a smaller file.")

    ext = filename.lower().split(".")[-1]
    df = None

    try:
        if ext == "xlsx":
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        elif ext == "csv":
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
        elif ext == "txt":
            # Attempt sniffing delimiter, fallback to common separators
            content = file_bytes.decode("utf-8", errors="replace")
            try:
                df = pd.read_csv(io.StringIO(content), sep=None, engine="python")
            except Exception:
                for delim in ["\t", ",", ";", r"\s+"]:
                    try:
                        df = pd.read_csv(io.StringIO(content), sep=delim, engine="python")
                        if len(df.columns) >= 4:
                            break
                    except Exception:
                        continue
                if df is None:
                    raise ValueError("Could not determine delimiter for TXT file. Please use comma, tab, or semicolon.")
        else:
            raise ValueError(f"Unsupported file extension '.{ext}'. Supported formats are: .csv, .txt, .xlsx")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")

    if df is None or df.empty:
        raise ValueError("The uploaded file contains no data.")

    if len(df) > MAX_ROWS:
        raise ValueError(f"File contains {len(df)} rows, exceeding maximum limit of {MAX_ROWS} rows.")

    col_map = map_columns(list(df.columns))

    warnings = []
    valid_rows = []

    for idx, row in df.iterrows():
        row_num = idx + 2  # 1-indexed plus header row
        try:
            sl = float(row[col_map["sepal_length"]])
            sw = float(row[col_map["sepal_width"]])
            pl = float(row[col_map["petal_length"]])
            pw = float(row[col_map["petal_width"]])

            # Range & NaN validation
            vals = [sl, sw, pl, pw]
            if any(np.isnan(v) or np.isinf(v) or v <= 0 or v > 50 for v in vals):
                warnings.append(f"Row {row_num} skipped: values must be positive numbers <= 50.")
                continue

            valid_rows.append({
                "sepal_length": round(sl, 2),
                "sepal_width": round(sw, 2),
                "petal_length": round(pl, 2),
                "petal_width": round(pw, 2),
            })
        except (ValueError, TypeError):
            warnings.append(f"Row {row_num} skipped: non-numeric measurement values.")

    if not valid_rows:
        raise ValueError("No valid Iris measurement rows found in the uploaded file.")

    return valid_rows, warnings


def generate_batch_excel(results: list[dict], metrics: dict) -> bytes:
    """Generate a professionally styled XLSX workbook with predictions and model metrics."""
    wb = openpyxl.Workbook()

    # Sheet 1: Predictions
    ws_pred = wb.active
    ws_pred.title = "Batch Predictions"

    # Styling constants
    navy_fill = PatternFill(start_color="1F3864", end_color="1F3864", fill_type="solid")
    soft_fill = PatternFill(start_color="F2F4F8", end_color="F2F4F8", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    bold_font = Font(name="Calibri", size=10, bold=True)
    regular_font = Font(name="Calibri", size=10)
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    headers = [
        "Row #",
        "Sepal Length (cm)",
        "Sepal Width (cm)",
        "Petal Length (cm)",
        "Petal Width (cm)",
        "RBF Prediction",
        "RBF Confidence (%)",
        "Linear Prediction",
        "Linear Confidence (%)",
        "Poly Prediction",
        "Poly Confidence (%)",
        "Sigmoid Prediction",
        "Sigmoid Confidence (%)",
        "MLP Prediction",
        "MLP Confidence (%)",
        "Consensus",
    ]

    ws_pred.append(headers)
    for col_idx in range(1, len(headers) + 1):
        cell = ws_pred.cell(row=1, column=col_idx)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ws_pred.row_dimensions[1].height = 28

    for row_idx, r in enumerate(results, start=2):
        row_data = [
            r["row_number"],
            r["sepal_length"],
            r["sepal_width"],
            r["petal_length"],
            r["petal_width"],
            r.get("rbf_prediction", "—"),
            f"{r.get('rbf_confidence', 0.0)}%",
            r.get("linear_prediction", "—"),
            f"{r.get('linear_confidence', 0.0)}%",
            r.get("poly_prediction", "—"),
            f"{r.get('poly_confidence', 0.0)}%",
            r.get("sigmoid_prediction", "—"),
            f"{r.get('sigmoid_confidence', 0.0)}%",
            r.get("mlp_prediction", "—"),
            f"{r.get('mlp_confidence', 0.0)}%",
            r.get("consensus", "—"),
        ]
        ws_pred.append(row_data)

        fill = soft_fill if row_idx % 2 == 0 else None
        for col_idx in range(1, len(row_data) + 1):
            cell = ws_pred.cell(row=row_idx, column=col_idx)
            cell.font = regular_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if fill:
                cell.fill = fill

    # Auto-fit column widths
    for col in ws_pred.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws_pred.column_dimensions[col_letter].width = max(max_len + 4, 14)

    # Sheet 2: Model Evaluation Metrics
    ws_metrics = wb.create_sheet(title="Model Evaluation Summary")

    metric_headers = [
        "Classifier / Model",
        "Test Accuracy (%)",
        "5-Fold CV Accuracy (%)",
        "CV Std (±%)",
        "Precision (%)",
        "Recall (%)",
        "F1-Score (%)",
    ]
    ws_metrics.append(metric_headers)
    for col_idx in range(1, len(metric_headers) + 1):
        cell = ws_metrics.cell(row=1, column=col_idx)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    ws_metrics.row_dimensions[1].height = 26

    for row_idx, k in enumerate(["rbf", "linear", "poly", "sigmoid", "mlp"], start=2):
        m = metrics.get(k, {})
        row_data = [
            m.get("kernel_display", k.upper()),
            f"{m.get('test_accuracy', 0.0)}%",
            f"{m.get('cv_accuracy', 0.0)}%",
            f"±{m.get('cv_std', 0.0)}%",
            f"{m.get('precision', 0.0)}%",
            f"{m.get('recall', 0.0)}%",
            f"{m.get('f1', 0.0)}%",
        ]
        ws_metrics.append(row_data)
        for col_idx in range(1, len(row_data) + 1):
            cell = ws_metrics.cell(row=row_idx, column=col_idx)
            cell.font = regular_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")

    for col in ws_metrics.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws_metrics.column_dimensions[col_letter].width = max(max_len + 4, 16)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def generate_batch_csv(results: list[dict]) -> str:
    """Generate a clean CSV string of batch prediction results for all 5 classifiers."""
    df = pd.DataFrame([
        {
            "Row": r["row_number"],
            "Sepal Length (cm)": r["sepal_length"],
            "Sepal Width (cm)": r["sepal_width"],
            "Petal Length (cm)": r["petal_length"],
            "Petal Width (cm)": r["petal_width"],
            "RBF Prediction": r.get("rbf_prediction", ""),
            "RBF Confidence (%)": r.get("rbf_confidence", 0.0),
            "Linear Prediction": r.get("linear_prediction", ""),
            "Linear Confidence (%)": r.get("linear_confidence", 0.0),
            "Poly Prediction": r.get("poly_prediction", ""),
            "Poly Confidence (%)": r.get("poly_confidence", 0.0),
            "Sigmoid Prediction": r.get("sigmoid_prediction", ""),
            "Sigmoid Confidence (%)": r.get("sigmoid_confidence", 0.0),
            "MLP Prediction": r.get("mlp_prediction", ""),
            "MLP Confidence (%)": r.get("mlp_confidence", 0.0),
            "Consensus": r.get("consensus", ""),
        }
        for r in results
    ])
    return df.to_csv(index=False)
