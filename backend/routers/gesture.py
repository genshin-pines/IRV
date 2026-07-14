from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services.gesture_service import (
    drain_messages,
    gesture_status,
    list_available_cameras,
    latest_frame,
    latest_raw_jpeg,
    map_event_to_vehicle,
    recognize_frame_bytes,
    start_gesture_stream,
    stop_gesture_stream,
)
from backend.services.log_service import write_log
from backend.models.auth_user import AuthUser
from backend.routers.custom_gestures import get_authenticated_driver

router = APIRouter(prefix="/api/gesture", tags=["gesture"])


def response(data=None, message: str = "success", ok: bool = True) -> dict:
    return {"ok": ok, "data": data, "message": message, "trace_id": datetime.now().strftime("%Y%m%d-") + uuid4().hex[:8]}


class GestureStartRequest(BaseModel):
    src_url: str | None = Field(default="rtsp://10.126.59.120:8554/live/live1")
    use_webcam: bool = False
    camera_index: int = Field(default=0, ge=0, le=10)
    mirror: bool = False
    enable_rtsp: bool = True


class GestureEvent(BaseModel):
    source: str = "driver"
    gesture_type: str
    gesture_label: str = ""
    confidence: float = 1.0
    stable: bool = True
    command: dict | None = None


@router.post("/start")
def api_start(payload: GestureStartRequest, user: AuthUser = Depends(get_authenticated_driver)):
    try:
        return response(start_gesture_stream(
            src_url=payload.src_url,
            use_webcam=payload.use_webcam,
            camera_index=payload.camera_index,
            mirror=payload.mirror,
            user_id=user.id,
            enable_rtsp=payload.enable_rtsp,
        ))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/cameras")
def api_list_cameras(user: AuthUser = Depends(get_authenticated_driver)):
    del user
    return response(list_available_cameras())


@router.post("/stop")
def api_stop():
    return response(stop_gesture_stream())


@router.get("/status")
def api_status():
    return response(gesture_status())


@router.get("/messages")
def api_messages(limit: int = Query(20, ge=1, le=100)):
    return response(drain_messages(limit=limit))


@router.post("/driver/recognize-frame")
async def api_driver_frame(file: UploadFile = File(...)):
    try:
        data = recognize_frame_bytes(await file.read(), filename=file.filename or "frame")
        return response(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/traffic-police/recognize-frame")
async def api_traffic_police_frame(file: UploadFile = File(...)):
    try:
        data = recognize_frame_bytes(await file.read(), filename=file.filename or "frame")
        data["note"] = "当前接入 web_gesture 动态手势引擎；交警8类规则可在此接口继续扩展。"
        return response(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/events")
async def api_gesture_event(event: GestureEvent):
    command = event.command or map_event_to_vehicle(event.model_dump())
    write_log("gesture", "INFO", f"gesture event source={event.source} type={event.gesture_type} confidence={event.confidence} stable={event.stable} command={command}")
    if event.confidence < 0.98:
        write_log("gesture", "WARNING", f"gesture confidence low source={event.source} type={event.gesture_type} confidence={event.confidence}")

    # 推送车主手势到融合引擎（三路感知 EventBus）
    try:
        from backend.services.alert_service import get_event_bus
        from fusion.perception_event import PerceptionEvent, Module

        bus = get_event_bus()
        if bus is not None:
            pe = PerceptionEvent.from_gesture(
                gesture_type=event.gesture_type,
                gesture_name=event.gesture_label or event.gesture_type,
                confidence=event.confidence,
                module=Module.DRIVER_GESTURE,
            )
            await bus.publish(pe)
    except Exception:
        pass

    return response({**event.model_dump(), "command": command})


@router.get("/video-feed")
def api_video_feed(raw: bool = Query(False)):
    async def generate():
        while True:
            if raw:
                jpeg = latest_raw_jpeg()
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            else:
                frame = latest_frame()
                if frame is not None:
                    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.websocket("/ws")
async def gesture_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            messages = drain_messages(limit=20)
            for message in messages:
                await ws.send_json(message)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
