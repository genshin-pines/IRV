from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.services.music_service import add_track, delete_track, list_tracks, reorder_tracks, seed_default_tracks


router = APIRouter(prefix="/api/music", tags=["music"])


def response(data=None, message: str = "success", ok: bool = True) -> dict:
    return {"ok": ok, "data": data, "message": message, "trace_id": datetime.now().strftime("%Y%m%d-") + uuid4().hex[:8]}


class AddTrackRequest(BaseModel):
    title: str
    artist: str = ""
    duration_sec: int = 0
    source_url: str = ""


class ReorderRequest(BaseModel):
    track_ids: list[int] = Field(min_length=1)


def _track_to_dict(track) -> dict[str, Any]:
    return {
        "id": track.id,
        "title": track.title,
        "artist": track.artist,
        "duration_sec": track.duration_sec,
        "source_url": track.source_url,
        "sort_order": track.sort_order,
    }


@router.get("/tracks")
def api_list_tracks(db: Session = Depends(get_db)):
    tracks = list_tracks(db)
    if not tracks:
        tracks = seed_default_tracks(db)
    return response([_track_to_dict(t) for t in tracks])


@router.post("/tracks")
def api_add_track(payload: AddTrackRequest, db: Session = Depends(get_db)):
    track = add_track(
        db,
        title=payload.title,
        artist=payload.artist,
        duration_sec=payload.duration_sec,
        source_url=payload.source_url,
    )
    return response(_track_to_dict(track))


@router.delete("/tracks/{track_id}")
def api_delete_track(track_id: int, db: Session = Depends(get_db)):
    if not delete_track(db, track_id):
        raise HTTPException(status_code=404, detail="track not found")
    return response({"deleted": True})


@router.put("/tracks/reorder")
def api_reorder_tracks(payload: ReorderRequest, db: Session = Depends(get_db)):
    reorder_tracks(db, payload.track_ids)
    tracks = list_tracks(db)
    return response([_track_to_dict(t) for t in tracks])
