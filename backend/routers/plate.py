from __future__ import annotations

import asyncio
import base64
import logging
import queue
import threading
import time
from datetime import datetime
from uuid import uuid4

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.services.log_service import write_log
from backend.services.plate_service import (
    publish_plate_events,
    recognize_image_bytes,
    recognize_stream,
    recognize_video_bytes,
)
from backend.services.local_video_service import LocalVideoManager, delete_video, resolve_video, save_upload, warmup_models

router = APIRouter(prefix="/api/plate", tags=["plate"])
logger = logging.getLogger(__name__)


def response(data=None, message: str = "success", ok: bool = True) -> dict:
    return {"ok": ok, "data": data, "message": message, "trace_id": datetime.now().strftime("%Y%m%d-") + uuid4().hex[:8]}


class StreamRequest(BaseModel):
    rtsp_url: str = Field(..., examples=["rtsp://10.126.59.120:8554/live/live1"])
    duration_sec: float = Field(8, ge=1, le=60)
    sample_interval: float = Field(0.5, ge=0.1, le=10)


@router.post("/recognize-image")
async def api_recognize_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, recognize_image_bytes, contents, file.filename or "upload")
        await publish_plate_events(data.get("plates", []), camera_id=file.filename or "upload")
        return response(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/recognize-video")
async def api_recognize_video(file: UploadFile = File(...), interval: float = Query(0.5, ge=0.1, le=10)):
    try:
        contents = await file.read()
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, recognize_video_bytes, contents, file.filename or "video", interval)
        await publish_plate_events(data.get("plates", []), camera_id=file.filename or "video")
        return response(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/local-video")
async def api_upload_local_video(file: UploadFile = File(...)):
    try:
        return response(await save_upload(file), message="本地视频已就绪")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.websocket("/local-video/live")
async def api_local_video_live(ws: WebSocket):
    await ws.accept()
    manager = LocalVideoManager()
    frame_queue: queue.Queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()
    video_id = ""

    def reader() -> None:
        try:
            while not stop_event.is_set() and manager.running:
                frame, plates, inference_ms = manager.read_frame()
                if frame is None:
                    break
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    continue
                payload = {
                    "type": "frame",
                    "frame": base64.b64encode(jpeg).decode("ascii"),
                    "plates": plates,
                    "inference_ms": inference_ms,
                    "frame_index": manager.frame_index,
                    "total_frames": manager.total_frames,
                    "progress": round(manager.frame_index / max(manager.total_frames, 1), 4),
                }
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(payload)
                time.sleep(max(0.001, 1.0 / max(manager.fps, 1.0)))
        except Exception as exc:
            manager.error = f"本地视频处理失败：{exc}"
            manager.running = False
            logger.exception("local video reader failed")

    try:
        message = await ws.receive_json()
        video_id = str(message.get("video_id", ""))
        path = resolve_video(video_id)
        if path is None:
            await ws.send_json({"type": "error", "message": "本地视频不存在或已失效"})
            return
        await ws.send_json({"type": "status", "stage": "warming", "message": "正在加载车辆与车牌识别模型"})
        await asyncio.to_thread(warmup_models)
        await ws.send_json({"type": "status", "stage": "opening", "message": "模型已就绪，正在打开视频"})
        if not manager.open(path):
            write_log("plate", "ERROR", f"plate video failed: cannot open video filename={path.name}")
            await ws.send_json({"type": "error", "message": "无法打开本地视频"})
            return

        await ws.send_json({"type": "status", "stage": "running", "connected": True, "message": "视频识别已启动"})
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        published_codes: set[str] = set()
        while manager.running or not frame_queue.empty():
            try:
                payload = frame_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            await ws.send_json(payload)
            new_plates = []
            for plate in payload["plates"]:
                code = plate.get("code", "")
                if code and code != "???" and code not in published_codes:
                    published_codes.add(code)
                    new_plates.append(
                        {
                            "plate_code": code,
                            "confidence": plate.get("conf", 0.0),
                            "plate_color": plate.get("color", "未知"),
                            "plate_type": -1,
                            "bbox": plate.get("bbox", []),
                        }
                    )
            await publish_plate_events(new_plates, camera_id=f"local:{path.name}")
        if manager.error:
            await ws.send_json({"type": "error", "message": manager.error})
        else:
            await ws.send_json({"type": "ended", "frames": manager.frame_index})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("local video websocket failed")
        try:
            await ws.send_json({"type": "error", "message": f"本地视频处理失败：{exc}"})
        except Exception:
            pass
    finally:
        stop_event.set()
        manager.close()
        delete_video(video_id)


@router.post("/recognize-stream")
async def api_recognize_stream(payload: StreamRequest):
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, recognize_stream, payload.rtsp_url, payload.duration_sec, payload.sample_interval)
        await publish_plate_events(data.get("plates", []), camera_id=payload.rtsp_url)
        return response(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
