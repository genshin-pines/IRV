from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.models.alert_event import AlertEvent, AlertStatus
from backend.schemas.alerts import AlertCreate, AlertUpdate


def create_alert(db: Session, payload: AlertCreate) -> AlertEvent:
    alert = AlertEvent(**payload.model_dump())
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def get_alert(db: Session, alert_id: int) -> AlertEvent | None:
    return db.get(AlertEvent, alert_id)


def list_alerts(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    level: str | None = None,
    status: str | None = None,
    source_module: str | None = None,
) -> tuple[list[AlertEvent], int]:
    stmt = select(AlertEvent)
    count_stmt = select(func.count()).select_from(AlertEvent)
    filters = []
    if level:
        filters.append(AlertEvent.level == level.upper())
    if status:
        filters.append(AlertEvent.status == status.upper())
    if source_module:
        filters.append(AlertEvent.source_module == source_module)
    for condition in filters:
        stmt = stmt.where(condition)
        count_stmt = count_stmt.where(condition)
    total = db.scalar(count_stmt) or 0
    items = db.scalars(
        stmt.order_by(AlertEvent.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    ).all()
    return list(items), total


def update_alert(db: Session, alert_id: int, payload: AlertUpdate) -> AlertEvent | None:
    alert = get_alert(db, alert_id)
    if alert is None:
        return None
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(alert, key, value)
    alert.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return alert


def acknowledge_alert(db: Session, alert_id: int, ack_user: str) -> AlertEvent | None:
    alert = get_alert(db, alert_id)
    if alert is None:
        return None
    alert.status = AlertStatus.ACKNOWLEDGED.value
    alert.ack_user = ack_user
    alert.ack_time = datetime.now(timezone.utc)
    alert.updated_at = alert.ack_time
    db.commit()
    db.refresh(alert)
    return alert


def delete_alert(db: Session, alert_id: int) -> bool:
    alert = get_alert(db, alert_id)
    if alert is None:
        return False
    db.delete(alert)
    db.commit()
    return True


def get_alert_stats(db: Session) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = db.scalar(select(func.count()).select_from(AlertEvent).where(AlertEvent.created_at >= today_start)) or 0
    unread_count = db.scalar(select(func.count()).select_from(AlertEvent).where(AlertEvent.status == AlertStatus.UNREAD.value)) or 0
    error_count = db.scalar(select(func.count()).select_from(AlertEvent).where(AlertEvent.level == "ERROR")) or 0
    critical_count = db.scalar(select(func.count()).select_from(AlertEvent).where(AlertEvent.level == "CRITICAL")) or 0

    by_module_rows = db.execute(select(AlertEvent.source_module, func.count()).group_by(AlertEvent.source_module)).all()
    by_module = {module or "system": count for module, count in by_module_rows}

    trend = []
    for offset in range(3, -1, -1):
        start = (now - timedelta(hours=offset)).replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
        count = db.scalar(
            select(func.count()).select_from(AlertEvent).where(AlertEvent.created_at >= start, AlertEvent.created_at < end)
        ) or 0
        trend.append({"hour": start.isoformat(), "count": count})

    return {
        "today_count": today_count,
        "unread_count": unread_count,
        "error_count": error_count,
        "critical_count": critical_count,
        "by_module": by_module,
        "trend_4h": trend,
    }


def cleanup_old_alerts(db: Session, days: int = 30) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    old_alerts = db.scalars(select(AlertEvent).where(AlertEvent.created_at < cutoff)).all()
    count = len(old_alerts)
    for alert in old_alerts:
        db.delete(alert)
    db.commit()
    return count
