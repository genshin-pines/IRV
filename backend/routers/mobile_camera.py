from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.schemas.common import ok
from backend.services.camera_service import latest_mjpeg_generator
from backend.services.mobile_camera_service import connect_source, disconnect_source, probe_source, status
from backend.services.mobile_camera_service import get_connected_source


router = APIRouter(prefix="/api/mobile-camera", tags=["mobile-camera"])


class MobileCameraRequest(BaseModel):
    source_url: str = Field(min_length=1, max_length=2048)


@router.get("/status")
def api_status():
    return ok(status())


@router.post("/probe")
def api_probe(payload: MobileCameraRequest):
    try:
        return ok(probe_source(payload.source_url), message="手机视频源连接正常")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/connect")
def api_connect(payload: MobileCameraRequest):
    try:
        return ok(connect_source(payload.source_url), message="手机视频源已连接")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/disconnect")
def api_disconnect():
    disconnect_source()
    return ok(status(), message="手机视频源已断开")


@router.get("/stream")
def api_stream(
    fps: float = Query(8, ge=1, le=25),
    width: int = Query(960, ge=320, le=1920),
    quality: int = Query(80, ge=40, le=95),
):
    source_url = get_connected_source()
    if not source_url:
        raise HTTPException(status_code=409, detail="手机视频源未连接")
    camera = {"id": "phone_cam", "name": "手机摄像头", "rtsp_url": source_url}
    return StreamingResponse(
        latest_mjpeg_generator(camera, fps=fps, width=width, quality=quality),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
