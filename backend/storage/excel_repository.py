"""Excel-based user account repository.

Manages E:\\8_Project 2026\\HMNC_PRO\\user_accounts.xlsx.
Never overwrites existing data; only appends new users.
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import openpyxl
from openpyxl import Workbook, load_workbook

import config

# Thread-lock to prevent concurrent Excel corruption
_lock = threading.Lock()

HEADERS = ["username", "password_hash", "created_at", "last_login", "role", "status"]


def init_excel_file() -> None:
    """Create user_accounts.xlsx with correct headers if it does not exist."""
    path: Path = config.USER_ACCOUNTS_PATH
    if path.exists():
        return  # Never overwrite existing file
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Users"
    ws.append(HEADERS)
    wb.save(path)


def _load_workbook_safe() -> tuple[Workbook, openpyxl.worksheet.worksheet.Worksheet]:
    path: Path = config.USER_ACCOUNTS_PATH
    if not path.exists():
        init_excel_file()
    wb = load_workbook(path)
    ws = wb["Users"] if "Users" in wb.sheetnames else wb.active
    return wb, ws


def find_user(username: str) -> Optional[dict]:
    """Return user dict if found, else None. Case-sensitive match."""
    with _lock:
        try:
            wb, ws = _load_workbook_safe()
        except Exception:
            return None
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] == username:
                return dict(zip(HEADERS, row))
        return None


def create_user(username: str, password_hash: str) -> dict:
    """Append a new USER row and save. Returns the new user dict."""
    with _lock:
        wb, ws = _load_workbook_safe()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [username, password_hash, now, None, "USER", "active"]
        ws.append(row)
        wb.save(config.USER_ACCOUNTS_PATH)
        return dict(zip(HEADERS, row))


def update_last_login(username: str) -> None:
    """Update the last_login timestamp for an existing user."""
    with _lock:
        try:
            wb, ws = _load_workbook_safe()
        except Exception:
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for row in ws.iter_rows(min_row=2):
            if row[0].value == username:
                row[3].value = now  # last_login column
                break
        wb.save(config.USER_ACCOUNTS_PATH)


def list_users() -> list[dict]:
    """Return all users as a list of dicts (admin use)."""
    with _lock:
        try:
            wb, ws = _load_workbook_safe()
        except Exception:
            return []
        results = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0]:
                results.append(dict(zip(HEADERS, row)))
        return results
