from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse

from backend.services.camera_service import get_camera, list_cameras, mjpeg_generator, read_snapshot

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("/health")
def api_camera_health():
    """摄像头连通性健康检查"""
    from backend.services.camera_health_service import health_status
    return {"ok": True, "data": health_status(), "message": "success"}


@router.get("")
def api_list_cameras():
    return {"ok": True, "data": list_cameras(), "message": "success"}


@router.get("/{camera_id}")
def api_get_camera(camera_id: str):
    camera = get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return {"ok": True, "data": camera, "message": "success"}


@router.get("/{camera_id}/snapshot.jpg")
def api_snapshot(camera_id: str, width: int = Query(1280, ge=320, le=2560), quality: int = Query(90, ge=40, le=95)):
    camera = get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return Response(read_snapshot(camera, width=width, quality=quality), media_type="image/jpeg")


@router.get("/{camera_id}/stream")
def api_stream(camera_id: str, fps: float = Query(8, ge=1, le=25), width: int = Query(960, ge=320, le=1920), quality: int = Query(80, ge=40, le=95)):
    camera = get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return StreamingResponse(mjpeg_generator(camera, fps=fps, width=width, quality=quality), media_type="multipart/x-mixed-replace; boundary=frame")
