from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.models.plate_record import PlateRecord


def save_plate_records(db: Session, plates: list[dict[str, Any]], *, source: str, source_type: str) -> int:
    """Persist accepted results; one result per plate code per recognition request."""
    best: dict[str, dict[str, Any]] = {}
    for plate in plates:
        code = str(plate.get("plate_code") or plate.get("code") or "").strip()
        if not code or code == "???":
            continue
        candidate = dict(plate)
        candidate["plate_code"] = code
        if code not in best or float(candidate.get("confidence", candidate.get("conf", 0)) or 0) > float(best[code].get("confidence", best[code].get("conf", 0)) or 0):
            best[code] = candidate

    for plate in best.values():
        bbox = plate.get("bbox") or []
        db.add(PlateRecord(
            plate_code=plate["plate_code"],
            plate_color=str(plate.get("plate_color") or plate.get("color") or "unknown"),
            confidence=float(plate.get("confidence", plate.get("conf", 0)) or 0),
            source=source[:512],
            source_type=source_type,
            time_sec=plate.get("time_sec"),
            bbox=json.dumps(bbox, ensure_ascii=False),
        ))
    db.commit()
    return len(best)


def list_plate_records(
    db: Session, *, page: int, page_size: int, plate_code: str | None = None,
    source_type: str | None = None, start_time: datetime | None = None, end_time: datetime | None = None,
) -> tuple[list[PlateRecord], int]:
    filters = []
    if plate_code:
        filters.append(PlateRecord.plate_code.contains(plate_code.strip().upper()))
    if source_type:
        filters.append(PlateRecord.source_type == source_type)
    if start_time:
        filters.append(PlateRecord.recognized_at >= start_time)
    if end_time:
        filters.append(PlateRecord.recognized_at <= end_time)
    total = db.scalar(select(func.count()).select_from(PlateRecord).where(*filters)) or 0
    items = db.scalars(
        select(PlateRecord).where(*filters).order_by(PlateRecord.recognized_at.desc()).offset((page - 1) * page_size).limit(page_size)
    ).all()
    return list(items), total


def serialize_plate_record(record: PlateRecord) -> dict[str, Any]:
    try:
        bbox = json.loads(record.bbox or "[]")
    except json.JSONDecodeError:
        bbox = []
    return {
        "id": record.id, "plate_code": record.plate_code, "plate_color": record.plate_color,
        "confidence": round(record.confidence, 4), "source": record.source,
        "source_type": record.source_type, "time_sec": record.time_sec, "bbox": bbox,
        "recognized_at": record.recognized_at.isoformat(),
    }
