from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from backend.services.plate_service import (
    publish_plate_events,
    recognize_image_bytes,
    recognize_stream,
    recognize_video_bytes,
)

router = APIRouter(prefix="/api/plate", tags=["plate"])


def response(data=None, message: str = "success", ok: bool = True) -> dict:
    return {"ok": ok, "data": data, "message": message, "trace_id": datetime.now().strftime("%Y%m%d-") + uuid4().hex[:8]}


class StreamRequest(BaseModel):
    rtsp_url: str = Field(..., examples=["rtsp://10.126.59.120:8554/live/live1"])
    duration_sec: float = Field(8, ge=1, le=60)
    sample_interval: float = Field(0.5, ge=0.1, le=10)


@router.post("/recognize-image")
async def api_recognize_image(file: UploadFile = File(...)):
    try:
        data = recognize_image_bytes(await file.read(), filename=file.filename or "upload")
        await publish_plate_events(data.get("plates", []), camera_id=file.filename or "upload")
        return response(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/recognize-video")
async def api_recognize_video(file: UploadFile = File(...), interval: float = Query(0.5, ge=0.1, le=10)):
    try:
        data = recognize_video_bytes(await file.read(), filename=file.filename or "video", interval=interval)
        await publish_plate_events(data.get("plates", []), camera_id=file.filename or "video")
        return response(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/recognize-stream")
async def api_recognize_stream(payload: StreamRequest):
    try:
        data = recognize_stream(payload.rtsp_url, payload.duration_sec, payload.sample_interval)
        await publish_plate_events(data.get("plates", []), camera_id=payload.rtsp_url)
        return response(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
