from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CustomGestureBinding(Base):
    __tablename__ = "custom_gesture_bindings"
    __table_args__ = (
        UniqueConstraint("user_id", "gesture_key", name="uq_custom_gesture_user_key"),
        UniqueConstraint("user_id", "display_name", name="uq_custom_gesture_user_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("auth_users.id"), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(64), nullable=False)
    gesture_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action_code: Mapped[str] = mapped_column(String(64), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    trigger_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


