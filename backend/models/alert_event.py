from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class AlertLevel(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertStatus(StrEnum):
    UNREAD = "UNREAD"
    READ = "READ"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AlertEvent(Base):
    __tablename__ = "alert_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    level: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    source_module: Mapped[str] = mapped_column(String(64), index=True, default="system")
    affected_modules: Mapped[str] = mapped_column(String(255), default="")
    ai_generated: Mapped[bool] = mapped_column(default=False)
    fingerprint: Mapped[str] = mapped_column(String(128), default="", index=True)
    webhook_markdown: Mapped[str] = mapped_column(Text, default="")
    acknowledged: Mapped[bool] = mapped_column(default=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str] = mapped_column(String(128), default="")
    ttl_minutes: Mapped[int] = mapped_column(Integer, default=60)
    raw_log: Mapped[str] = mapped_column(Text, default="")
    llm_summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), index=True, default=AlertStatus.UNREAD.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    ack_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ack_user: Mapped[str | None] = mapped_column(String(128), nullable=True)
