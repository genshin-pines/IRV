from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.user_preference import UserPreference


def get_preference(db: Session, user_id: int, key: str) -> str | None:
    row = db.scalar(
        select(UserPreference).where(UserPreference.user_id == user_id, UserPreference.key == key)
    )
    return row.value if row else None


def get_all_preferences(db: Session, user_id: int) -> dict[str, str]:
    rows = db.scalars(
        select(UserPreference).where(UserPreference.user_id == user_id)
    ).all()
    return {row.key: row.value for row in rows}


def set_preference(db: Session, user_id: int, key: str, value: str) -> UserPreference:
    row = db.scalar(
        select(UserPreference).where(UserPreference.user_id == user_id, UserPreference.key == key)
    )
    if row:
        row.value = value
        row.updated_at = datetime.now(timezone.utc)
    else:
        row = UserPreference(user_id=user_id, key=key, value=value)
        db.add(row)
    db.commit()
    db.refresh(row)
    return row


def batch_set_preferences(db: Session, user_id: int, prefs: dict[str, str]) -> list[UserPreference]:
    results = []
    for key, value in prefs.items():
        results.append(set_preference(db, user_id, key, value))
    return results


def delete_preference(db: Session, user_id: int, key: str) -> bool:
    row = db.scalar(
        select(UserPreference).where(UserPreference.user_id == user_id, UserPreference.key == key)
    )
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True
