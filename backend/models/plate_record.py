from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PlateRecord(Base):
    """An accepted plate recognition result retained for history queries."""

    __tablename__ = "plate_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    plate_code: Mapped[str] = mapped_column(String(32), index=True)
    plate_color: Mapped[str] = mapped_column(String(32), default="unknown")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(512), default="", index=True)
    source_type: Mapped[str] = mapped_column(String(32), default="image", index=True)
    time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox: Mapped[str] = mapped_column(String(128), default="")
    recognized_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
