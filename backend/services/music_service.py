from __future__ import annotations

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from backend.models.music_track import MusicTrack


def list_tracks(db: Session, user_id: int | None = None) -> list[MusicTrack]:
    stmt = select(MusicTrack)
    if user_id is not None:
        stmt = stmt.where(MusicTrack.user_id == user_id)
    else:
        stmt = stmt.where(MusicTrack.user_id.is_(None))
    return list(db.scalars(stmt.order_by(MusicTrack.sort_order, MusicTrack.id)).all())


def add_track(db: Session, title: str, artist: str = "", duration_sec: int = 0, source_url: str = "", user_id: int | None = None) -> MusicTrack:
    max_order = db.scalar(select(func.coalesce(func.max(MusicTrack.sort_order), -1))) or -1
    track = MusicTrack(
        title=title,
        artist=artist,
        duration_sec=duration_sec,
        source_url=source_url,
        user_id=user_id,
        sort_order=max_order + 1,
    )
    db.add(track)
    db.commit()
    db.refresh(track)
    return track


def delete_track(db: Session, track_id: int) -> bool:
    track = db.get(MusicTrack, track_id)
    if track is None:
        return False
    db.delete(track)
    db.commit()
    return True


def reorder_tracks(db: Session, track_ids: list[int]) -> None:
    for index, track_id in enumerate(track_ids):
        track = db.get(MusicTrack, track_id)
        if track:
            track.sort_order = index
    db.commit()


def seed_default_tracks(db: Session) -> list[MusicTrack]:
    existing = db.scalar(select(func.count()).select_from(MusicTrack)) or 0
    if existing > 0:
        return list_tracks(db)
    defaults = [
        ("City Lights", "Lofi Beats", 195, ""),
        ("Night Drive", "Chillwave", 210, ""),
        ("Morning Dew", "Ambient", 180, ""),
        ("Highway", "Electronic", 200, ""),
        ("Sunset Boulevard", "Jazz Hop", 225, ""),
        ("星空", "轻音乐", 240, ""),
        ("夏日微风", "Acoustic", 172, ""),
        ("出发", "Rock", 188, ""),
    ]
    for title, artist, duration, url in defaults:
        add_track(db, title=title, artist=artist, duration_sec=duration, source_url=url)
    return list_tracks(db)
