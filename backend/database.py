from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import DATABASE_URL


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from backend.models.alert_event import AlertEvent  # noqa: F401
    from backend.models.auth_user import AuthCode, AuthUser  # noqa: F401
    from backend.models.custom_gesture_binding import CustomGestureBinding  # noqa: F401
    from backend.models.music_track import MusicTrack  # noqa: F401
    from backend.models.user_preference import UserPreference  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_alert_columns_for_sqlite()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_alert_columns_for_sqlite() -> None:
    if not DATABASE_URL.startswith("sqlite"):
        return
    inspector = inspect(engine)
    if "alert_events" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("alert_events")}
    expected = {
        "raw_log": "TEXT",
        "llm_summary": "TEXT",
        "status": "VARCHAR(32) DEFAULT 'UNREAD'",
        "updated_at": "DATETIME",
        "ack_time": "DATETIME",
        "ack_user": "VARCHAR(128)",
    }
    with engine.begin() as conn:
        for name, ddl_type in expected.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE alert_events ADD COLUMN {name} {ddl_type}"))
