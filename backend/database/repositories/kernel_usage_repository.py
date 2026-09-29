"""Repository for KernelUsageHistory database operations."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Union

from sqlalchemy.orm import Session

import config
from database.models import KernelUsageHistory, User


def record_kernel_usage(
    db: Session,
    user_id: int,
    kernel: str,
    used_at: Optional[datetime] = None,
) -> KernelUsageHistory:
    """Record a kernel usage event in SQL Server for a given user.

    Validates that the kernel is one of the allowed kernels (rbf, linear, poly, sigmoid).
    Uses db.flush() and db.refresh() without committing.
    """
    kernel_clean = kernel.strip().lower()
    if kernel_clean not in [k.lower() for k in config.ALLOWED_KERNELS]:
        raise ValueError(
            f"Invalid or unsupported kernel '{kernel}'. Allowed kernels: {config.ALLOWED_KERNELS}"
        )

    record = KernelUsageHistory(
        UserID=user_id,
        Kernel=kernel_clean.upper(),
        UsedAt=used_at or datetime.utcnow(),
    )
    db.add(record)
    db.flush()
    db.refresh(record)
    return record


def get_kernel_usage_by_user(
    db: Session,
    user_id: Union[int, str],
    limit: Optional[int] = None,
) -> List[KernelUsageHistory]:
    """Retrieve kernel usage history for a user, newest first (user isolation)."""
    if isinstance(user_id, str):
        query = (
            db.query(KernelUsageHistory)
            .join(User, KernelUsageHistory.UserID == User.UserID)
            .filter(User.Username == user_id)
            .order_by(KernelUsageHistory.UsedAt.desc())
        )
    else:
        query = (
            db.query(KernelUsageHistory)
            .filter(KernelUsageHistory.UserID == user_id)
            .order_by(KernelUsageHistory.UsedAt.desc())
        )

    if limit is not None and limit > 0:
        query = query.limit(limit)

    return query.all()
