from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime
from uuid import uuid4

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.services.camera_service import get_camera
from backend.services.log_service import write_log
from backend.services.optimized_traffic_service import (
    create_capture as create_optimized_capture,
    create_session as create_optimized_session,
    get_runtime as get_optimized_runtime,
    is_available as optimized_available,
    resolve_upload as resolve_optimized_upload,
    save_upload as save_optimized_upload,
)
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
logger = logging.getLogger(__name__)


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
    data = traffic_status()
    available, reason = optimized_available()
    data["optimized"] = {"available": available, "reason": reason, "model": "BiLSTM multi-video"}
    return response(data)


@router.post("/optimized-video")
async def api_optimized_video(file: UploadFile = File(...)):
    try:
        return response(await save_optimized_upload(file), message="优化交警视频已就绪")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.websocket("/optimized-live")
async def api_optimized_live(ws: WebSocket):
    await ws.accept()
    source = None
    cap = None
    try:
        request = await ws.receive_json()
        source = resolve_optimized_upload(str(request.get("video_id", "")))
        if source is None:
            await ws.send_json({"type": "error", "message": "本地视频不存在或已失效"})
            return
        await ws.send_json({"type": "status", "message": "正在加载优化 BiLSTM 模型"})
        runtime = await get_optimized_runtime()
        session = create_optimized_session(runtime)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            await ws.send_json({"type": "error", "message": "无法打开本地视频"})
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        frame_duration = 1.0 / max(source_fps, 1.0)
        next_frame_at = asyncio.get_running_loop().time()
        frame_index = 0
        await ws.send_json({"type": "status", "message": "优化交警手势实时识别已启动"})
        while True:
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok or frame is None:
                break
            annotated, gesture = await asyncio.to_thread(session.process, frame)
            height, width = annotated.shape[:2]
            if width > 1280:
                scale = 1280 / width
                annotated = cv2.resize(annotated, (1280, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
            encoded, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not encoded:
                continue
            frame_index += 1
            await ws.send_json(
                {
                    "type": "frame",
                    "frame": base64.b64encode(jpeg).decode("ascii"),
                    "gesture": gesture,
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "progress": round(frame_index / max(total_frames, 1), 4),
                }
            )
            next_frame_at += frame_duration
            await asyncio.sleep(max(0.0, next_frame_at - asyncio.get_running_loop().time()))
        await ws.send_json({"type": "ended", "frames": frame_index})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("optimized traffic video stream failed")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if cap is not None:
            cap.release()
        if source is not None:
            source.unlink(missing_ok=True)


@router.websocket("/optimized-camera-live")
async def api_optimized_camera_live(ws: WebSocket):
    await ws.accept()
    capture = None
    try:
        request = await ws.receive_json()
        await ws.send_json({"type": "status", "message": "正在加载优化 BiLSTM 模型"})
        runtime = await get_optimized_runtime()
        session = create_optimized_session(runtime)
        capture = create_optimized_capture(
            camera_index=int(request.get("camera_index", 0)),
            source_url=request.get("source_url") or None,
        )
        await asyncio.to_thread(capture.start)
        await ws.send_json({"type": "status", "message": "优化交警手势摄像头识别已启动"})
        sequence = 0
        processed = 0
        started = asyncio.get_running_loop().time()
        while capture.running:
            sequence, frame = await asyncio.to_thread(capture.latest, sequence, 1.0)
            if frame is None:
                continue
            annotated, gesture = await asyncio.to_thread(session.process, frame)
            height, width = annotated.shape[:2]
            if width > 1280:
                scale = 1280 / width
                annotated = cv2.resize(annotated, (1280, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
            encoded, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not encoded:
                continue
            processed += 1
            elapsed = max(asyncio.get_running_loop().time() - started, 1e-3)
            await ws.send_json(
                {
                    "type": "frame",
                    "frame": base64.b64encode(jpeg).decode("ascii"),
                    "gesture": gesture,
                    "frame_index": processed,
                    "processing_fps": round(processed / elapsed, 1),
                    "dropped_frames": max(0, sequence - processed),
                }
            )
        if capture.error:
            await ws.send_json({"type": "error", "message": capture.error})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if capture is not None:
            await asyncio.to_thread(capture.stop)


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
