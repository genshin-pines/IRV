from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services.camera_service import get_camera
from backend.services.log_service import write_log
from backend.services.traffic_police_service import (
    drain_messages,
    latest_frame,
    recognize_frame_bytes,
    start_stream,
    status as traffic_status,
    stop_stream,
    summarize_gesture_frame,
)


router = APIRouter(prefix="/api/traffic-police", tags=["traffic-police"])


def response(data=None, message: str = "success", ok: bool = True) -> dict:
    return {"ok": ok, "data": data, "message": message, "trace_id": datetime.now().strftime("%Y%m%d-") + uuid4().hex[:8]}


class TrafficStartRequest(BaseModel):
    src_url: str | None = Field(default=None)
    use_webcam: bool = False


class DriverAssistAnalyzeRequest(BaseModel):
    camera_id: str = "live1"
    scene: str = Field(default="normal", examples=["normal", "traffic_police", "camera_disconnect", "near_collision"])


@router.get("/status")
def api_status():
    return response(traffic_status())


@router.post("/start")
def api_start(payload: TrafficStartRequest):
    try:
        return response(start_stream(src_url=payload.src_url, use_webcam=payload.use_webcam))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/stop")
def api_stop():
    return response(stop_stream())


@router.get("/messages")
def api_messages(limit: int = Query(20, ge=1, le=100)):
    return response(drain_messages(limit=limit))


@router.post("/recognize-frame")
async def api_recognize_frame(file: UploadFile = File(...)):
    try:
        data = recognize_frame_bytes(await file.read(), filename=file.filename or "frame")
        data["driver_advice"] = summarize_gesture_frame(data.get("frame"))
        return response(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/video-feed")
def api_video_feed():
    async def generate():
        while True:
            frame = latest_frame()
            if frame is not None:
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/driver-assist/analyze")
async def api_driver_assist_analyze(payload: DriverAssistAnalyzeRequest):
    camera = get_camera(payload.camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")

    scene = payload.scene
    if scene == "traffic_police":
        level = "WARNING"
        title = "车外摄像头检测到交警指挥场景"
        advice = "请关注前方交警手势，可切换交警手势识别模式辅助判断。"
    elif scene == "camera_disconnect":
        level = "ERROR"
        title = "车外行车记录仪画面异常"
        advice = "行车记录仪画面中断，请检查外部摄像头或 RTSP 视频源。"
    elif scene == "near_collision":
        level = "CRITICAL"
        title = "车外辅助检测到近距离风险"
        advice = "建议立即减速并保持安全距离。"
    else:
        level = "INFO"
        title = "车外行车记录仪巡检正常"
        advice = "车外画面已接入，可用于 Agent 预警、车牌辅助和交警手势识别。"

    message = f"driver assist scene={scene} camera={camera['id']} name={camera['name']} advice={advice}"
    write_log("camera" if level in {"ERROR", "CRITICAL"} else "system", level, message)

    # 推送交警手势场景到融合引擎（三路感知 EventBus）
    try:
        from backend.services.alert_service import get_event_bus
        from fusion.perception_event import PerceptionEvent, Module

        bus = get_event_bus()
        if bus is not None:
            pe = PerceptionEvent.from_gesture(
                gesture_type=payload.scene,
                gesture_name=title,
                confidence=0.85,
                module=Module.TRAFFIC_GESTURE,
                camera_id=payload.camera_id,
            )
            await bus.publish(pe)
    except Exception:
        pass

    return response(
        {
            "camera": camera,
            "scene": scene,
            "level": level,
            "title": title,
            "advice": advice,
            "agent_hint": "已写入日志收集器；事件驱动 Agent 会立即分析，必要时生成 /api/alerts 告警。",
        }
    )
